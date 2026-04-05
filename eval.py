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

    class ActionPreProcessor:
        def __init__(self, mean, std, history_size, f_skip):
            self.mean = mean
            self.std = std
            self.history_size = history_size
            self.f_skip = f_skip

        def __call__(self, batch):
            action = batch["action"].clone()
            action[..., -2:] = (action[..., -2:] - self.mean) / (self.std + 1e-6)
            # 重塑为 (history_size, frameskip * 22) -> (3, 110)
            action = action.contiguous().view(self.history_size, self.f_skip * action.shape[-1])
            batch["action"] = action
            return batch

    transforms = [
        get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.eval.img_size),
        ActionPreProcessor(cam_mean, cam_std, cfg.wm.history_size, frameskip)
    ]
    dataset.transform = spt.data.transforms.Compose(*transforms)

    # --- 2. 模型构建 ---
    print("构建模型结构...")
    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )
    
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = frameskip * 22

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    predictor_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    ).to(device)

    # --- 3. 加载指定 Artifact 权重 ---
    artifact_path = "oopsyoudied88-aaa/my_train_true1/model-njde34z4:v33"
    world_model = load_wandb_model(artifact_path, world_model, device=device)
    world_model.eval()
    world_model.requires_grad_(False)

    # --- 4. 初始化 Solver ---
    with open_dict(cfg):
        if "plan_config" not in cfg:
            cfg.plan_config = {
                "horizon": 3,
                "num_samples": 256, # 增加采样数以获得更好效果
                "temperature": 0.1,
            }
        if "solver" not in cfg:
            cfg.solver = {
                "_target_": "stable_worldmodel.solver.MPPISolver",
                "horizon": cfg.plan_config.horizon,
                "num_samples": cfg.plan_config.num_samples,
                "action_dim": effective_act_dim,
            }

    print(f"初始化 Solver: {cfg.solver._target_}")
    solver = hydra.utils.instantiate(cfg.solver, model=world_model)

    # --- 5. 执行规划测试 ---
    print("\n--- 开始基于打散数据的规划测试 ---")
    num_test_samples = 5
    # 因为数据集已打散，我们直接取前 N 个样本即可，或者继续随机采样
    indices = np.arange(num_test_samples) 
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
            
            # A. 编码样本获取 Latents
            output = world_model.encode(batch)
            full_emb = output["emb"]  # (1, 50, D)
            
            # B. 准备规划输入
            ctx_len = cfg.wm.history_size # 3
            initial_obs_emb = full_emb[:, :ctx_len] # (1, 3, D)
            
            # 提取真实的目标特征 (例如 ctx_len 之后第 3 步的特征) 作为规划目标
            target_step = ctx_len + cfg.plan_config.horizon - 1
            goal_latent = full_emb[:, target_step] # (1, D)
            
            print(f"\n[测试 {i+1}] 样本索引: {idx}")
            
            # C. 调用 Solver 获取最优动作
            # 这里的 solve 逻辑取决于 stable_worldmodel 具体实现
            # 我们假设 solver 需要目标特征来计算 cost
            start_time = time.time()
            try:
                # 传入初始 latent 和目标 latent 进行规划
                best_actions = solver.solve(initial_obs_emb, goal=goal_latent)
                duration = time.time() - start_time
                
                # D. 验证规划结果
                # 用最优动作跑一遍模型预测
                planned_act_emb = world_model.action_encoder(best_actions.unsqueeze(0))
                predicted_latents = world_model.predict(initial_obs_emb, planned_act_emb)
                
                # 最后一帧预测与目标的距离
                final_predicted_latent = predicted_latents[:, -1]
                dist_to_goal = (final_predicted_latent - goal_latent).pow(2).mean()
                
                # 原始数据集动作产生的距离 (Baseline)
                gt_actions = batch["action"] # (1, 3, 110)
                gt_act_emb = world_model.action_encoder(gt_actions)
                gt_predicted_latents = world_model.predict(initial_obs_emb, gt_act_emb)
                gt_dist = (gt_predicted_latents[:, -1] - goal_latent).pow(2).mean()

                print(f"  - 规划耗时: {duration:.4f}s")
                print(f"  - 规划预测 MSE (到目标): {dist_to_goal.item():.6f}")
                print(f"  - 数据集动作 MSE (Baseline): {gt_dist.item():.6f}")
                
                if dist_to_goal < gt_dist:
                    print("  - 结果: 规划动作比数据集原始动作更接近目标特征。")
                else:
                    print("  - 结果: 数据集动作表现更好（或规划器采样不足）。")

            except Exception as e:
                print(f"  - 规划过程出错: {e}")

    print("\n测试脚本执行完毕。")

if __name__ == "__main__":
    run_test()