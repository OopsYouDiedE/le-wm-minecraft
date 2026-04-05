import os
import torch
import hydra
import wandb
import numpy as np
import time
from pathlib import Path
from omegaconf import OmegaConf, open_dict

import stable_pretraining as spt
import stable_worldmodel as swm
import lightning as pl

# 导入原始定义的模块
from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_img_preprocessor
from minestudio_inmemory_dataset import MineStudioInMemoryDataset

def load_wandb_model(artifact_path, model_instance, device="cuda"):
    """
    从 WandB 下载并加载指定的模型权重
    """
    print(f"正在从 WandB 下载 Artifact: {artifact_path} ...")
    run = wandb.init(project="eval_test", job_type="evaluation")
    artifact = run.use_artifact(artifact_path, type='model')
    artifact_dir = artifact.download()
    
    ckpt_files = list(Path(artifact_dir).glob("*.ckpt"))
    if not ckpt_files:
        ckpt_files = list(Path(artifact_dir).glob("*.pth"))
        
    if ckpt_files:
        print(f"找到权重文件: {ckpt_files[0]}")
        checkpoint = torch.load(ckpt_files[0], map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model.model."): 
                new_state_dict[k.replace("model.model.", "")] = v
            elif k.startswith("model."):
                new_state_dict[k.replace("model.", "")] = v
            else:
                new_state_dict[k] = v
                
        model_instance.load_state_dict(new_state_dict, strict=False)
        print("权重加载成功！")
    else:
        raise FileNotFoundError("未在 Artifact 中找到有效的权重文件。")
    
    run.finish()
    return model_instance

@hydra.main(version_base=None, config_path="./config/eval", config_name="minecraft")
def run_test(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. 数据准备 ---
    # 由于数据集本身已打散，我们直接加载
    print("正在加载已打散的全内存数据集...")
    h5_path = cfg.data.dataset.get("h5_file_path", "data_0000.h5")
    dataset = MineStudioInMemoryDataset(h5_file_path=h5_path, transform=None)
    
    cam_data = torch.from_numpy(dataset.camera_actions).float()
    cam_mean = cam_data.mean(dim=(0, 1))
    cam_std = cam_data.std(dim=(0, 1))
    frameskip = cfg.data.dataset.get('frameskip', 5)
    print('cam_mean',cam_mean,'\ncam_std',cam_std)

if __name__ == "__main__":
    run_test()