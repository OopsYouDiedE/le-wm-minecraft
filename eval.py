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
    # 【修改逻辑】：使用与 train.py 完全相同的动态路径获取方式
    h5_path = cfg.data.dataset.get("h5_file_path", "data_0000.h5")
    
    # 动态解析缓存目录与文件名
    cache_dir = Path(h5_path).parent if Path(h5_path).parent.name else Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    file_name = Path(h5_path).name
    
    dataset = swm.data.HDF5Dataset(
        file_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=cache_dir,
    )
    return dataset

# ==========================================
# 【核心修复区】：专为你的 Train 脚本定制的动作处理器
# ==========================================
class CustomActionProcessor:
    """
    仅对动作向量的最后两维（Camera 的 Pitch 和 Yaw）进行标准化。
    兼容 sklearn 的 transform 和 inverse_transform API，供 Solver 规划时调用。
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, action):
        # 规划器前向预测时调用
        act = action.copy() if isinstance(action, np.ndarray) else action.clone()
        act[..., -2:] = (act[..., -2:] - self.mean) / (self.std + 1e-6)
        return act

    def inverse_transform(self, action):
        # 规划器将预测的动作发送给真实 Minecraft 环境前调用
        act = action.copy() if isinstance(action, np.ndarray) else action.clone()
        act[..., -2:] = act[..., -2:] * (self.std + 1e-6) + self.mean
        return act

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """执行世界模型评估与潜空间规划"""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "规划视野(Horizon)必须小于或等于评估预算"

    # 【已被移除】：移除了所有与 swm.World 相关的初始化和 cfg 注入逻辑
    # 现已完全脱离黑盒环境依赖

    # 创建图像转换器
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    # ---------------------------------------------------------
    # 【对齐训练逻辑】：动态读取数据集获取全局统计算法
    # ---------------------------------------------------------
    h5_path = cfg.data.dataset.get("h5_file_path", "data_0000.h5")
    try:
        # 引入你写的全内存类，保证底层矩阵读取的均值与训练时一模一样
        from minestudio_inmemory_dataset import MineStudioInMemoryDataset
        in_memory_dataset = MineStudioInMemoryDataset(h5_file_path=h5_path, transform=None)
        
        cam_data = torch.from_numpy(in_memory_dataset.camera_actions).float()
        # 转换为 numpy，方便 solver 进行 ndarray 计算
        cam_mean = cam_data.mean(dim=(0, 1)).numpy()
        cam_std = cam_data.std(dim=(0, 1)).numpy()
    except ImportError:
        # 如果 eval 脚本所在环境无法导入该类，则降级使用常规抽取计算
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
            # 注入计算好的处理器
            process[col] = CustomActionProcessor(cam_mean, cam_std)
        else:
            # 对于其他状态向量（如果有），保留原有的 StandardScaler
            processor = preprocessing.StandardScaler()
            col_data = stats_dataset.get_col_data(col)
            col_data = col_data[~np.isnan(col_data).any(axis=1)]
            processor.fit(col_data)
            process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- 开始加载模型并执行规划
    policy = cfg.get("policy", "random")

    if policy != "random":
        # 加载我们在 WandB 跑出来的预测器模型
        model = swm.policy.AutoCostModel(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        
        config = swm.PlanConfig(**cfg.plan_config)
        # 这里实例化的就是 CEMSolver 或者 GradientSolver
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )
    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
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
    # 【核心沙盘推演环节】：自定义离线规划循环，替代 world.evaluate_from_dataset
    # =================================================================
    print("==================================================")
    print("[*] 已脱离 swm.World，启动完全自定义的离线规划推演...")
    print("==================================================")
    
    metrics = {"success": [], "planning_time": []}
    
    if hasattr(policy, "reset"):
        policy.reset()

    # 仅取前几个用于快速验证规划器是否跑通
    eval_limit = min(len(eval_episodes), cfg.eval.num_eval)
    
    for i in range(eval_limit):
        ep_id = eval_episodes[i]
        start_idx = eval_start_idx[i]
        goal_idx = start_idx + cfg.eval.goal_offset_steps
        
        print(f"[*] 正在评估 第 {i+1}/{eval_limit} 个片段 (Episode: {ep_id}, Start: {start_idx} -> Goal: {goal_idx})")
        
        try:
            # 从 HDF5 提取图像数据
            start_data = dataset.get_row_data(np.array([start_idx]))
            goal_data = dataset.get_row_data(np.array([goal_idx]))
            
            # 构造 Policy 需要的 obs 字典
            # 这里假设提取出来的 pixels 能够直接兼容 transform 的要求
            obs = {
                "pixels": start_data["pixels"][0],
                "goal": goal_data["pixels"][0]
            }
            
            step_start_time = time.time()
            
            # 执行潜空间规划！这句代码会调用 CEM 或者 Gradient Solver
            action = policy(obs)
            
            plan_time = time.time() - step_start_time
            metrics["success"].append(1)
            metrics["planning_time"].append(plan_time)
            
            # 输出规划结果 (只打印部分维度防止刷屏)
            if isinstance(action, np.ndarray) or isinstance(action, torch.Tensor):
                print(f"    [+] 规划成功! 耗时: {plan_time:.2f}s, 求解器输出动作形状: {action.shape}")
            else:
                print(f"    [+] 规划成功! 耗时: {plan_time:.2f}s, 输出类型: {type(action)}")
                
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
    print(f"最终评估指标: {metrics_summary}")
    print("==================================================")

    # 保存结果日志
    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")
        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")
        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics_summary}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")

if __name__ == "__main__":
    run()