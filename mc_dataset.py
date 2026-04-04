import time
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MineStudioInMemoryDataset(Dataset):
    """
    专为大内存机器设计的极致性能 Dataset。
    一次性将整个 HDF5 文件吞入内存，彻底消除磁盘 I/O 瓶颈。
    """
    def __init__(self, h5_file_path: str, transform=None):
        super().__init__()
        self.h5_file_path = h5_file_path
        self.transform = transform
        
        print(f"🚀 开始将全量数据加载到内存中，请稍候...")
        print(f"📁 目标文件: {self.h5_file_path}")
        start_time = time.time()
        
        # ==========================================
        # 核心逻辑：全量装载进 RAM (使用 [:] 语法)
        # 警告：这里保持 uint8 和 float16，千万不要全局转换为 float32！
        # ==========================================
        with h5py.File(h5_file_path, 'r') as f:
            # 读取图像数据 (N, 4, 3, 224, 224), 格式: uint8
            self.pixels = f['pixels'][:]
            
            # 读取离散动作 (N, 15, 20), 格式: uint8
            self.binary_actions = f['binary_actions'][:]
            
            # 读取相机动作 (N, 15, 2), 格式: float16
            self.camera_actions = f['camera_actions'][:]
            
        self.num_samples = len(self.pixels)
        end_time = time.time()
        
<<<<<<< HEAD
        # 计算加载到内存中的实际大小 (GB)
        mem_gb = (self.pixels.nbytes + self.binary_actions.nbytes + self.camera_actions.nbytes) / (1024**3)
        
        print(f"✅ 加载完成! 耗时: {end_time - start_time:.2f} 秒.")
        print(f"📊 总样本数: {self.num_samples}")
        print(f"💾 内存占用: 约 {mem_gb:.2f} GB (原生格式)")
=======
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
        ep_info = self.global_map[self.episode_names[ep_idx]]
        start_chunk, end_chunk = start // self.chunk_size, (end - 1) // self.chunk_size
        result_steps = {}
>>>>>>> 43295cb10682989cb8280d9b358184228337e3a5

<<<<<<< HEAD
    def __len__(self):
        return self.num_samples
=======
        for modality in self.load_data:
            meta = ep_info[modality]
            env = self._get_or_open_env(meta['prefix'], modality, meta['part'])
            chunk_data_list = []

            with env.begin() as txn:
                for c_offset in range(start_chunk * self.chunk_size, (end_chunk + 1) * self.chunk_size, self.chunk_size):
                    raw_bytes = txn.get(f"({meta['idx']}, {c_offset})".encode('utf-8'))
                    if not raw_bytes: break
                    
                    if modality == 'image':
                        chunk_data_list.append(self._decode_video_chunk(raw_bytes))
                    else:
                        data = pickle.loads(raw_bytes)
                        if isinstance(data, dict):
                            vals = [np.expand_dims(data[k], -1) if np.ndim(data[k]) == 1 else np.array(data[k]) for k in sorted(data.keys())]
                            chunk_data_list.append(np.concatenate(vals, axis=-1))
                        else:
                            val = np.array(data)
                            chunk_data_list.append(np.expand_dims(val, -1) if val.ndim == 1 else val)

            # 兜底：如果完全没有数据，回退到开头
            if not chunk_data_list:
                return self._load_slice(ep_idx, 0, end - start)

            full_array = np.concatenate(chunk_data_list, axis=0)
            local_start = start % self.chunk_size
            local_end = local_start + (end - start)

            # 切片与维度处理
            if modality == 'image':
                tensor = torch.from_numpy(full_array[local_start : local_end : self.frameskip])
                if tensor.ndim == 4: 
                    tensor = tensor.permute(0, 3, 1, 2)
                target_len = (end - start + self.frameskip - 1) // self.frameskip
                out_key = 'pixels'
            else:
                tensor = torch.from_numpy(full_array[local_start : local_end]).float()
                target_len = end - start
                out_key = modality
>>>>>>> 43295cb10682989cb8280d9b358184228337e3a5

<<<<<<< HEAD
    def __getitem__(self, idx):
=======
            # 强制对齐 (Padding)：解决 DataLoader not resizable 错误
            if tensor.shape[0] < target_len:
                padding = tensor[-1:].repeat(target_len - tensor.shape[0], *([1] * (tensor.ndim - 1)))
                tensor = torch.cat([tensor, padding], dim=0)

            result_steps[out_key] = tensor

        return self.transform(result_steps) if self.transform else result_steps

    # ------------------ 适配 stable-worldmodel 的接口 ------------------

    def get_col_data(self, col: str) -> np.ndarray:
>>>>>>> 43295cb10682989cb8280d9b358184228337e3a5
        """
        单条数据的获取逻辑。
        因为数据已经在内存中，这里的索引操作速度接近光速。
        """
        # 1. 内存极速索引
        px_np = self.pixels[idx]           # (4, 3, 224, 224)
        bin_act_np = self.binary_actions[idx]  # (15, 20)
        cam_act_np = self.camera_actions[idx]  # (15, 2)
        
<<<<<<< HEAD
        # 2. 转换为 PyTorch Tensor 并执行升维类型转换 (Type Casting)
        # 图像转为 float32 并归一化到 [0, 1] 区间（视觉模型通用做法）
        pixels_tensor = torch.from_numpy(px_np).float() / 255.0
=======
        # --- 抽样策略配置 ---
        # 经验：对于 110 维动作，抽取 500 个序列或约 20 万帧已足够精准
        max_stats_episodes = 3
>>>>>>> 43295cb10682989cb8280d9b358184228337e3a5
        
<<<<<<< HEAD
        # 动作全部转为 float32 并拼接，形成统一的 22 维动作向量
        bin_tensor = torch.from_numpy(bin_act_np).float()
        cam_tensor = torch.from_numpy(cam_act_np).float()
        action_tensor = torch.cat([bin_tensor, cam_tensor], dim=-1) # (15, 22)
=======
        all_indices = np.arange(len(self.episode_names))
        if len(all_indices) > max_stats_episodes:
            # 随机挑选索引，保证统计分布的代表性
            sample_indices = np.random.choice(all_indices, max_stats_episodes, replace=False)
            logging.info(f"📊 数据集规模较大 ({len(all_indices)} eps)，正在随机抽取 {max_stats_episodes} 个序列进行统计...")
        else:
            sample_indices = all_indices
            logging.info(f"📊 正在对全量数据集 ({len(all_indices)} eps) 进行统计...")

        all_data = []
        for i in sample_indices:
            try:
                # 调用我们写好的 _load_slice，它已经处理了动作展平
                data_dict = self._load_slice(i, 0, self.lengths[i])
                
                # 兼容性处理：脚本可能请求 'action'，我们也确保返回正确键名
                key = 'pixels' if col == 'pixels' else 'action'
                if key not in data_dict:
                    # 如果键名不匹配，尝试按原始 load_data 寻找
                    key = col if col in data_dict else list(data_dict.keys())[0]
                
                all_data.append(data_dict[key].numpy())
                
                # 每读取 1 个打印一次进度，避免刷屏
                if len(all_data) % 1 == 0:
                    print(f"   [统计进度] 已读取 {len(all_data)} / {len(sample_indices)} 个序列...")
                    
            except Exception as e:
                logging.warning(f"统计读取跳过 {self.episode_names[i]}: {e}")

        # 合并数据
        combined_data = np.concatenate(all_data, axis=0)
        logging.info(f"✅ 统计数据收集完毕，总计样本数: {combined_data.shape[0]} 帧")
        return combined_data

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
>>>>>>> 43295cb10682989cb8280d9b358184228337e3a5
        
        batch = {
            "pixels": pixels_tensor,
            "action": action_tensor
        }
        
        # 如果你传入了 torchvision 等 Transform，可以在这里应用
        if self.transform:
            batch = self.transform(batch)
            
        return batch