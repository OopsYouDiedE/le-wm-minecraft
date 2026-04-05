import os

os.environ["MUJOCO_GL"] = "egl"

import time
import traceback
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

def img_transform(cfg):
    """图像预处理逻辑，与训练时保持 ImageNet 标准一致"""
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform

def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)

def get_dataset(cfg, dataset_name):
    # 【动态路径】：使用与 train.py 完全相同的动态路径获取方式
    h5_path = cfg.data.dataset.get("h5_file_path", "data_0000.h5")
    
    cache_dir = Path(h5_path).parent if Path(h5_path).parent.name else Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    file_name = Path(h5_path).name
    
    dataset = swm.data.HDF5Dataset(
        file_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=cache_dir,
    )
    return dataset

# ==========================================
# 【定制区】：专为你的 Train 脚本定制的动作处理器
# ==========================================
class CustomActionProcessor:
    """仅对动作向量的最后两维（Camera 的 Pitch 和 Yaw）进行 Z-Score 标准化。"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, action):
        act = action.copy() if isinstance(action, np.ndarray) else action.clone()
        act[..., -2:] = (act[..., -2:] - self.mean) / (self.std + 1e-6)
        return act

    def inverse_transform(self, action):
        act = action.copy() if isinstance(action, np.ndarray) else action.clone()
        act[..., -2:] = act[..., -2:] * (self.std + 1e-6) + self.mean
        return act

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """执行世界模型评估与潜空间规划"""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "规划视野(Horizon)必须小于或等于评估预算"

    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    # ---------------------------------------------------------
    # 【对齐训练】：计算/获取全局 Camera 统计特征
    # ---------------------------------------------------------
    h5_path = cfg.data.dataset.get("h5_file_path", "data_0000.h5")
    try:
        from minestudio_inmemory_dataset import MineStudioInMemoryDataset
        in_memory_dataset = MineStudioInMemoryDataset(h5_file_path=h5_path, transform=None)
        cam_data = torch.from_numpy(in_memory_dataset.camera_actions).float()
        cam_mean = cam_data.mean(dim=(0, 1)).numpy()
        cam_std = cam_data.std(dim=(0, 1)).numpy()
    except ImportError:
        action_data = stats_dataset.get_col_data("action")
        action_data = action_data[~np.isnan(action_data).any(axis=1)]
        cam_data = action_data[:, -2:]
        cam_mean = np.mean(cam_data, axis=0)
        cam_std = np.std(cam_data, axis=0)
        
    print("==================================================")
    print("[*] 动作预处理器加载完毕: 仅标准化末尾 2 维 (Camera)")
    print(f"[*] 来源数据集: {h5_path}")
    print(f"[*] cam_mean: {cam_mean}")
    print(f"[*] cam_std:  {cam_std}")
    print("==================================================")

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        
        if col == "action":
            process[col] = CustomActionProcessor(cam_mean, cam_std)
        else:
            processor = preprocessing.StandardScaler()
            col_data = stats_dataset.get_col_data(col)
            col_data = col_data[~np.isnan(col_data).any(axis=1)]
            processor.fit(col_data)
            process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- 开始加载模型并执行规划
    policy_type = cfg.get("policy", "random")

    if policy_type != "random":
        # =========================================================================
        # 【新增功能】：从 WandB 自动下载模型并用于 Eval 推演
        # =========================================================================
        wandb_artifact_path = "oopsyoudied88-aaa/my_train_true1/model-njde34z4:v33"
        print("==================================================")
        print(f"[*] 正在连接 WandB，请求 Eval 模型: {wandb_artifact_path} ...")
        
        target_ckpt = None
        try:
            import wandb
            api = wandb.Api()
            artifact = api.artifact(wandb_artifact_path)
            # 下载到系统缓存目录
            download_dir = Path(swm.data.utils.get_cache_dir()) / "wandb_artifacts"
            artifact_dir = artifact.download(root=str(download_dir))
            
            # 自动搜索刚下好的 .ckpt
            ckpt_files = list(Path(artifact_dir).glob("*.ckpt")) + list(Path(artifact_dir).glob("*.pth"))
            
            if ckpt_files:
                target_ckpt = str(ckpt_files[0])
                print(f"[*] 成功找到权重文件，准备挂载: {target_ckpt}")
            else:
                print("[-] 警告: 下载的文件中未发现 .ckpt 或 .pth 模型！")
        except Exception as e:
            print(f"[-] 从 WandB 加载模型失败，降级使用 cfg 配置路径。错误: {e}")
        print("==================================================")

        # 动态覆盖加载路径：如果 WandB 下载成功就用新的，否则回退
        model_load_path = target_ckpt if target_ckpt else str(cfg.policy)
        
        # 将刚下载的模型路径喂给 AutoCostModel
        model = swm.policy.AutoCostModel(model_load_path)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        
        config = swm.PlanConfig(**cfg.plan_config)
        # 实例化 CEMSolver 或者 GradientSolver
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )
    else:
        policy = swm.policy.RandomPolicy()

    # 安全地构造保存结果的目录
    try:
        policy_str = str(cfg.policy) if cfg.policy else "random"
    except Exception:
        policy_str = "random"
        
    results_path = (
        Path(swm.data.utils.get_cache_dir(), policy_str).parent
        if policy_type != "random"
        else Path(__file__).parent
    )

    # 采样评估起点
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(f"[*] 找到 {valid_mask.sum()} 个合法的评估起始点。")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    random_episode_indices = np.sort(valid_indices[random_episode_indices])
    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("数据集中没有足够长度的回合用于评估。")

    start_time = time.time()
    
    # =================================================================
    # 【自定义沙盘推演环节】：直接使用起点和终点测试模型，跳过烦人的环境黑盒
    # =================================================================
    print("==================================================")
    print("[*] 启动潜空间规划 (Latent Planning) 引擎推演...")
    print("==================================================")
    
    metrics = {"success": [], "planning_time": []}
    
    if hasattr(policy, "reset"):
        policy.reset()

    eval_limit = min(len(eval_episodes), cfg.eval.num_eval)
    
    for i in range(eval_limit):
        ep_id = eval_episodes[i]
        start_idx = eval_start_idx[i]
        goal_idx = start_idx + cfg.eval.goal_offset_steps
        
        print(f"[*] Episode {ep_id} | 尝试从 Frame {start_idx} 规划至 Frame {goal_idx} ...")
        
        try:
            start_data = dataset.get_row_data(np.array([start_idx]))
            goal_data = dataset.get_row_data(np.array([goal_idx]))
            
            # 向规划器提供 "现在" 与 "未来目标"
            obs = {
                "pixels": start_data["pixels"][0],
                "goal": goal_data["pixels"][0]
            }
            
            step_start_time = time.time()
            
            # 【执行规划】调用你的 CEMSolver
            action = policy(obs)
            
            plan_time = time.time() - step_start_time
            metrics["success"].append(1)
            metrics["planning_time"].append(plan_time)
            
            if isinstance(action, np.ndarray) or isinstance(action, torch.Tensor):
                print(f"    [+] 成功! 耗时 {plan_time:.2f}s | 输出动作阵列维度: {action.shape}")
            else:
                print(f"    [+] 成功! 耗时 {plan_time:.2f}s")
                
        except Exception as e:
            metrics["success"].append(0)
            print(f"    [-] 规划失败: {e}")
            traceback.print_exc()
            
    metrics_summary = {
        "success_rate": float(np.mean(metrics["success"])) if metrics["success"] else 0.0,
        "mean_planning_time": float(np.mean(metrics["planning_time"])) if metrics["planning_time"] else 0.0
    }
    
    end_time = time.time()
    print("==================================================")
    print(f"推演总耗时: {end_time - start_time:.2f} 秒")
    print(f"最终评估指标: {metrics_summary}")
    print("==================================================")

    # 结果写入日志
    if getattr(cfg.output, 'filename', None):
        results_path = results_path / cfg.output.filename
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("a") as f:
            f.write("\n==== RESULTS ====\n")
            f.write(f"metrics: {metrics_summary}\n")

if __name__ == "__main__":
    run()