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
        
        # 计算加载到内存中的实际大小 (GB)
        mem_gb = (self.pixels.nbytes + self.binary_actions.nbytes + self.camera_actions.nbytes) / (1024**3)
        
        print(f"✅ 加载完成! 耗时: {end_time - start_time:.2f} 秒.")
        print(f"📊 总样本数: {self.num_samples}")
        print(f"💾 内存占用: 约 {mem_gb:.2f} GB (原生格式)")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        单条数据的获取逻辑。
        因为数据已经在内存中，这里的索引操作速度接近光速。
        """
        # 1. 内存极速索引
        px_np = self.pixels[idx]           # (4, 3, 224, 224)
        bin_act_np = self.binary_actions[idx]  # (15, 20)
        cam_act_np = self.camera_actions[idx]  # (15, 2)
        
        # 2. 转换为 PyTorch Tensor 并执行升维类型转换 (Type Casting)
        # 图像转为 float32 并归一化到 [0, 1] 区间（视觉模型通用做法）
        pixels_tensor = torch.from_numpy(px_np).float() / 255.0
        
        # 动作全部转为 float32 并拼接，形成统一的 22 维动作向量
        bin_tensor = torch.from_numpy(bin_act_np).float()
        cam_tensor = torch.from_numpy(cam_act_np).float()
        action_tensor = torch.cat([bin_tensor, cam_tensor], dim=-1) # (15, 22)
        
        batch = {
            "pixels": pixels_tensor,
            "action": action_tensor
        }
        
        # 如果你传入了 torchvision 等 Transform，可以在这里应用
        if self.transform:
            batch = self.transform(batch)
            
        return batch