import os

os.environ["MUJOCO_GL"] = "egl"

import time
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
    # 【修改逻辑】：强制指向你实际的数据集路径
    cache_dir = Path("/content/data")
    dataset = swm.data.HDF5Dataset(
        "data_0001.h5",
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

    # 创建世界环境
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

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
    # 【对齐训练逻辑】：直接读取 /content/data/data_0001.h5 获取全局统计算法
    # ---------------------------------------------------------
    try:
        # 引入你写的全内存类，保证底层矩阵读取的均值与训练时一模一样
        from minestudio_inmemory_dataset import MineStudioInMemoryDataset
        h5_file_path = "/content/data/data_0001"
        in_memory_dataset = MineStudioInMemoryDataset(h5_file_path=h5_file_path, transform=None)
        
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
    print(f"[*] 来源数据集: /content/data/data_0001.h5")
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

    world.set_policy(policy)

    start_time = time.time()
    
    # 【核心沙盘推演环节】
    # 脚本会自动把 (start_idx + goal_offset_steps) 的画面作为目标(Goal Image)喂给求解器
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video_path=results_path,
    )
    end_time = time.time()
    
    print(metrics)

    # 保存结果日志
    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")
        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")
        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")

if __name__ == "__main__":
    run()