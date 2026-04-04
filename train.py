import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_img_preprocessor, ModelObjectCallBack
# 【改动 1】移除旧依赖，直接导入我们写的全内存极速 Dataset
from minestudio_inmemory_dataset import MineStudioInMemoryDataset

def setup_wandb_login():
    """自动从 Colab Secrets 或环境变量中读取 WandB Key 并登录"""
    api_key = None
    
    # 1. 尝试从 Colab 的 Secrets 中读取
    try:
        from google.colab import userdata
        api_key = userdata.get('WANDB_API_KEY')
        if api_key:
            print("✅ 成功从 Colab Secrets 读取 WANDB_API_KEY")
    except ImportError:
        pass  # 非 Colab 环境
    except Exception:
        pass  # Secret 不存在或其他异常
        
    # 2. 尝试从环境变量读取
    if not api_key:
        api_key = os.environ.get('WANDB_API_KEY')
        if api_key:
            print("✅ 成功从环境变量读取 WANDB_API_KEY")
            
    # 3. 执行登录或降级为手动
    if api_key:
        wandb.login(key=api_key)
    else:
        print("⚠️ 未自动检测到 WANDB_API_KEY。若尚未登录，WandB 稍后会阻塞程序并提示您手动输入。")

def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size    # 值为 3
    lambd = cfg.loss.sigreg.weight

    # 1. 预处理
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)

    emb = output["emb"]              # 形状: (B, 50, D)
    act_emb = output["act_emb"]

    # 2. 提取历史 (Context)
    ctx_emb = emb[:, :ctx_len]       # 取第 0, 1, 2 帧
    ctx_act = act_emb[:, :ctx_len]

    # 3. 预测未来
    # 模型输出 pred_emb 形状为 (B, 3, D)，因为它是基于 ctx_len 生成的
    pred_emb = self.model.predict(ctx_emb, ctx_act) 

    # 4. 【核心修复】提取对应的目标值 (Label)
    # 原代码: tgt_emb = emb[:, n_preds:] -> 导致切出 49 帧
    # 修正后: 从历史结束的地方开始切，长度与预测步数一致
    pred_steps = pred_emb.shape[1]   # 动态获取预测步数 (3)
    tgt_emb = emb[:, ctx_len : ctx_len + pred_steps] # 取第 3, 4, 5 帧

    # 5. 计算损失 (此时维度均为 3，完美匹配)
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################
    
    # 【改动 2】取消所有多余的预处理、下载流程，一步到位吞入内存
    h5_path = cfg.data.dataset.get("h5_file_path", "data_0000.h5")
    dataset = MineStudioInMemoryDataset(h5_file_path=h5_path, transform=None)
    
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    # 【改动 3】极速计算 Camera 全局归一化参数 (利用全量内存读取的优势，耗时 < 0.1s)
    cam_data = torch.from_numpy(dataset.camera_actions).float()
    cam_mean = cam_data.mean(dim=(0, 1))
    cam_std = cam_data.std(dim=(0, 1))
    
    frameskip = cfg.data.dataset.get('frameskip', 5)
    
    # 【核心改动】专属的动作空间处理器
    class ActionPreProcessor:
        def __init__(self, mean, std, history_size, f_skip):
            self.mean = mean
            self.std = std
            self.history_size = history_size
            self.f_skip = f_skip

        def __call__(self, batch):
            action = batch["action"].clone() # Shape: (15, 22)
            
            # A. 仅对 Camera (最后的 2 个维度) 实施 Z-Score 归一化
            action[..., -2:] = (action[..., -2:] - self.mean) / (self.std + 1e-6)
            
            # B. 时序动作打包 (Action Chunking): 
            # 将扁平的 (15帧, 22维) 重塑为 Transformer 视角的 (3步, 5*22=110维)
            action = action.contiguous().view(self.history_size, self.f_skip * action.shape[-1])
            
            batch["action"] = action
            return batch

    transforms.append(ActionPreProcessor(cam_mean, cam_std, cfg.wm.history_size, frameskip))
    
    with open_dict(cfg):
        # 强制接管环境参数：写死我们现有的硬核指标，废除多余的自动获取逻辑
        cfg.wm.action_dim = 22
        cfg.data.dataset.frameskip = frameskip
        
    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    # DataLoader 中可以直接开启 shuffle，无任何 I/O 阻塞
    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    
    # 完美对齐： 5 * 22 = 110，这与我们 Transform 中重塑的末位维度一致
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        setup_wandb_login()
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()