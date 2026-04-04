import io
import pickle
import logging
import os
from pathlib import Path
from typing import Callable, Any, List, Optional

import lmdb
import av
import numpy as np
import torch
from huggingface_hub import snapshot_download

from stable_worldmodel.data.dataset import Dataset


def download_minestudio_datasets(dataset_prefixes: Optional[List[str]] = None, local_dir: str = '/content/data'):
    """
    独立的数据集下载函数。
    - 如果不填入 dataset_prefixes (如 None 或空列表)，则下载演示数据集 (10xx) 的一小部分 (使用 allow_patterns)。
    - 如果填入了列表 (例如 ['10xx', '9xx', '6xx'])，则下载列表中对应的完整数据集 (不使用 allow_patterns)。
    """
    downloaded_dirs = []
    if not dataset_prefixes:
        logging.info("未提供数据集列表，默认下载演示数据集 (10xx) 的一小部分...")
        repo_id = "CraftJarvis/minestudio-data-10xx-v110"
        target_dir = os.path.join(local_dir, '10xx')
        allow_patterns = [
            "action/part-???/**",
            "image/part-???/**"
        ]
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=target_dir,
            allow_patterns=allow_patterns
        )
        downloaded_dirs.append(target_dir)
    else:
        logging.info(f"检测到指定的数据集列表 {dataset_prefixes}，准备下载完整数据...")
        for prefix in dataset_prefixes:
            repo_id = f"CraftJarvis/minestudio-data-{prefix}-v110"
            target_dir = os.path.join(local_dir, '10xx')
            logging.info(f"正在下载完整数据集: {repo_id} 到 {target_dir}...")
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=target_dir
                # 填入了列表，要求下载完整数据集，因此不传入 allow_patterns
            )
            downloaded_dirs.append(target_dir)
    return downloaded_dirs


class LMDBDecoupledDataset(Dataset):
    """
    针对 MineStudio 数据集优化的解耦 LMDB 加载器。
    支持跨版本（6xx-10xx）聚合、多模态对齐以及懒加载。
    """

    def __init__(
        self,
        base_path: str | Path = '/content/data',
        frameskip: int = 1,
        num_steps: int = 1,
        load_data: List[str] = ['action', 'image'],
        transform: Optional[Callable[[dict], dict]] = None,
        chunk_size: int = 32
    ) -> None:
        self.base_path = Path(base_path)
        self.chunk_size = chunk_size
        self.load_data = load_data
        
        # 1. 构建全局索引与模态对齐 (内部处理 num_frames 和过滤)
        self.global_map, self.episode_names = self._build_alignment()

        if not self.episode_names:
            raise ValueError(f"未能在 {self.base_path} 下找到满足要求的完整数据集。")

        # 2. 准备基类所需的 lengths 数组 (每一集的 num_frames)
        lengths = []
        for name in self.episode_names:
            # 默认取第一个模态的长度
            primary_mod = self.load_data[0]
            lengths.append(self.global_map[name][primary_mod]['length'])

        lengths = np.array(lengths, dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        
        # 3. 初始化进程内 LMDB 句柄缓存
        self._envs: dict[str, lmdb.Environment] = {}

        # 4. 初始化基类：这将建立全局帧索引，支持 __getitem__ 的切片
        super().__init__(lengths, offsets, frameskip, num_steps, transform)

    def _build_alignment(self) -> tuple[dict, list[str]]:
        """扫描目录、提取 num_frames 并执行模态过滤"""
        logging.info("🌍 正在扫描 LMDB 目录并构建跨版本索引...")
        temp_map = {}
        prefixes = ["6xx", "7xx", "8xx", "9xx", "10xx"]

        for p1 in prefixes:
            prefix_path = self.base_path / p1
            if not prefix_path.exists(): continue
            
            for modality in self.load_data:
                mod_path = prefix_path / modality
                if not mod_path.exists(): continue
                
                for p3 in mod_path.iterdir():
                    if "part-" not in p3.name or not p3.is_dir(): continue
                    
                    try:
                        # 临时打开 LMDB 读取元数据
                        env = lmdb.open(str(p3), readonly=True, lock=False)
                        with env.begin() as txn:
                            infos = pickle.loads(txn.get(b'__chunk_infos__'))
                            for info in infos:
                                ep_id = info['episode']
                                temp_map.setdefault(ep_id, {})
                                # 存储模态具体位置和帧长度 (基于你发现的 num_frames)
                                temp_map[ep_id][modality] = {
                                    'prefix': p1,
                                    'part': p3.name,
                                    'idx': info['episode_idx'],
                                    'length': info.get('num_frames', 0)
                                }
                    env.close()
                    except Exception as e:
                        logging.warning(f"跳过损坏的分片 {p3}: {e}")
                        env.close()
                        

        # 执行模态补全过滤：必须包含所有请求的 keys
        required_mods = set(self.load_data)
        final_map = {
            k: v for k, v in temp_map.items() 
            if required_mods.issubset(v.keys())
        }
        
        episode_names = sorted(list(final_map.keys()))
        logging.info(f"✅ 索引构建完成。可用 Episode: {len(episode_names)}")
        return final_map, episode_names

    def _get_or_open_env(self, prefix: str, modality: str, part: str) -> lmdb.Environment:
        """懒加载环境句柄，确保多进程安全"""
        env_key = f"{prefix}_{modality}_{part}"
        if env_key not in self._envs:
            path = str(self.base_path / prefix / modality / part)
            self._envs[env_key] = lmdb.open(path, readonly=True, lock=False, max_readers=256)
        return self._envs[env_key]

    def _decode_video_chunk(self, chunk_bytes: bytes) -> np.ndarray:
        """使用 PyAV 将视频字节流转换为 RGB 数组"""
        with io.BytesIO(chunk_bytes) as f:
            with av.open(f) as container:
                stream = container.streams.video[0]
                frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
        return np.stack(frames)
    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        """
        核心加载方法：
        - 仅对 'image' 进行视频解码、frameskip 和维度转换 [T,C,H,W]。
        - 对其他模态（如 action）仅进行数据提取、字典展平与维度对齐，不进行采样变动。
        """
        ep_name = self.episode_names[ep_idx]
        ep_info = self.global_map[ep_name]

        # 1. 确定 Chunk 范围
        start_chunk = start // self.chunk_size
        end_chunk = (end - 1) // self.chunk_size
        
        result_steps = {}

        for modality in self.load_data:
            meta = ep_info[modality]
            env = self._get_or_open_env(meta['prefix'], modality, meta['part'])
            
            chunk_data_list = []
            with env.begin() as txn:
                for c_offset in range(start_chunk * self.chunk_size, (end_chunk + 1) * self.chunk_size, self.chunk_size):
                    key = f"({meta['idx']}, {c_offset})".encode('utf-8')
                    raw_bytes = txn.get(key)
                    if raw_bytes is None: break
                    
                    # --- 核心逻辑分歧点 ---
                    if modality == 'image':
                        # 只有图像进行视频解码
                        decoded = self._decode_video_chunk(raw_bytes)
                        chunk_data_list.append(decoded)
                    else:
                        # 其他模态（如 action）使用 pickle 加载
                        data = pickle.loads(raw_bytes)
                        if isinstance(data, dict):
                            # 处理字典：展平并强制升维以保证 concatenate 成功
                            step_acts = []
                            for k in sorted(data.keys()):
                                val = np.array(data[k])
                                if val.ndim == 1: val = np.expand_dims(val, axis=-1)
                                step_acts.append(val)
                            chunk_data_list.append(np.concatenate(step_acts, axis=-1))
                        else:
                            val = np.array(data)
                            if val.ndim == 1: val = np.expand_dims(val, axis=-1)
                            chunk_data_list.append(val)
            
            # 合并块
            full_array = np.concatenate(chunk_data_list, axis=0)
            
            # 计算局部切片索引
            local_start = start - start_chunk * self.chunk_size
            local_end = local_start + (end - start)
            
            # --- 结果处理分歧点 ---
            if modality == 'image':
                # 【变动项】：执行 frameskip 和维度置换
                sliced = full_array[local_start : local_end : self.frameskip]
                tensor = torch.from_numpy(sliced)
                if tensor.ndim == 4: # [T, H, W, C] -> [T, C, H, W]
                    tensor = tensor.permute(0, 3, 1, 2)
                result_steps['pixels'] = tensor
            else:
                # 【不变项】：不执行 frameskip，原样返回（仅转为 Tensor）
                sliced = full_array[local_start : local_end]
                result_steps[modality] = torch.from_numpy(sliced).float()

        return self.transform(result_steps) if self.transform else result_steps

    # ------------------ 适配 stable-worldmodel 的接口 ------------------

    def get_col_data(self, col: str) -> np.ndarray:
        """获取全量列数据（通常用于计算动作均值/方差）"""
        if col == 'pixels': raise MemoryError("禁止一次性加载全量像素数据。")
        all_data = []
        for i in range(len(self.episode_names)):
            # 直接提取全长数据
            data_dict = self._load_slice(i, 0, self.lengths[i])
            # 注意：stable-world 需要 action 而不是 load_data 里的名字
            key = 'pixels' if col == 'pixels' else 'action'
            all_data.append(data_dict[key].numpy())
        return np.concatenate(all_data, axis=0)

    def get_dim(self, col: str) -> int:
        """探测列维度"""
        sample = self._load_slice(0, 0, 1)
        data = sample['pixels' if col == 'pixels' else 'action']
        return np.prod(data.shape[1:]).item() if data.ndim > 1 else 1

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        """按全局索引获取单步数据"""
        if isinstance(row_idx, int): row_idx = [row_idx]
        res = {'pixels': [], 'action': []}
        for idx in row_idx:
            ep_idx = int(np.searchsorted(self.offsets, idx, side='right')) - 1
            local_start = idx - self.offsets[ep_idx]
            step = self._load_slice(ep_idx, local_start, local_start + 1)
            res['pixels'].append(step['pixels'][0])
            res['action'].append(step['action'][0])
        return {k: np.stack(v) for k, v in res.items()}

    def merge_col(self, source: list[str] | str, target: str, dim: int = -1) -> None:
        pass # LMDB 只读，不支持运行时合并
import cv2
import torch
import numpy as np

def visualize_dataset_slice(dataset, ep_idx, out_name="dataset_verify.mp4"):
    """
    直接通过调用 dataset._load_slice 来渲染视频。
    用于检查 frameskip、维度变换和动作对齐是否符合预期。
    """
    # 1. 获取该 Episode 的总长度
    ep_len = dataset.lengths[ep_idx]
    ep_name = dataset.episode_names[ep_idx]
    
    print(f"🧐 正在检查数据集切片: {ep_name}")
    print(f"   - 总帧数 (原始): {ep_len}")
    print(f"   - 跳帧步长 (frameskip): {dataset.frameskip}")

    # 2. 调用 dataset 内部的加载逻辑
    # 获取该 Episode 的全部数据（经过了 transform 和维度转换）
    data = dataset._load_slice(ep_idx, 0, ep_len)
    
    pixels = data['pixels'] # 预期形状: [T, C, H, W]
    actions = data['action'] # 预期形状: [T_raw, D] 或者是采样后的
    
    # 3. 准备视频写入器
    # 注意：T 的长度受 frameskip 影响
    T, C, H, W = pixels.shape
    out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (W, H))

    print(f"   - 渲染帧数 (采样后): {T}")
    print(f"   - 动作维度: {actions.shape[-1]}")

    for i in range(T):
        # --- 图像还原 ---
        # 逆转换: [C, H, W] -> [H, W, C]
        img_tensor = pixels[i].permute(1, 2, 0)
        
        # 如果像素被归一化到了 [0, 1] 或 [-1, 1]，需要还原到 [0, 255]
        img_np = img_tensor.cpu().numpy()
        if img_np.max() <= 1.1: # 自动检测归一化
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
            
        # RGB -> BGR (OpenCV)
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # --- 动作还原 ---
        # 找到对应时间点的动作。注意：图像是 T，动作通常是 T * frameskip
        # 这里直接取 i * frameskip 对应的原始动作
        act_idx = i # 如果你在 _load_slice 里对 action 也做了采样，直接用 i
        # 但通常 stable-worldmodel 期望 action 也是对齐采样后的，
        # 如果你的 _load_slice 没给 action 做采样，这里需要手动对齐：
        # act_val = actions[i * dataset.frameskip].numpy() 
        act_val = actions[i].numpy() # 假设已经对齐
        
        act_text = f"Step {i} | Act: {np.round(act_val[:4], 3)}..."

        # --- 绘制信息层 ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, 25), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.putText(frame, act_text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        out.write(frame)

    out.release()
    print(f"✅ 验证视频已生成: {out_name}")

# visualize_dataset_slice(dataset, ep_idx=0, out_name="check_alignment.mp4")