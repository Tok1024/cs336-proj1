import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cs336_basics.model import TransformerLM


@dataclass
class TrainConfig:
    train_tokens_path: str
    valid_tokens_path: str
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 8
    num_heads: int = 8
    d_ff: int = 1344
    rope_theta: float = 10000.0
    batch_size: int = 32
    total_iters: int = 2000
    eval_interval: int = 200
    eval_batches: int = 20
    max_learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_iters: int = 200
    cosine_cycle_iters: int = 2000
    max_grad_norm: float = 1.0
    seed: int = 42
    device: str = "cpu"
    checkpoint_path: str = "checkpoints/latest.pt"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_token_array(path: str) -> np.ndarray:
    # TODO: 任务1
    # 按你的数据格式读取为1D token id数组
    # 可以先支持.npy，再扩展到.bin等格式
    raise NotImplementedError("TODO: 完成load_token_array")


def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: 任务2
    # 复用你在adapters里的run_get_batch逻辑
    raise NotImplementedError("TODO: 完成get_batch")


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # TODO: 任务3
    # 复用你在adapters里的run_cross_entropy逻辑
    raise NotImplementedError("TODO: 完成cross_entropy_loss")


def clip_gradients(parameters, max_l2_norm: float) -> None:
    # TODO: 任务4
    # 复用你在adapters里的run_gradient_clipping逻辑
    raise NotImplementedError("TODO: 完成clip_gradients")


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    # TODO: 任务5
    # 复用你在adapters里的run_get_lr_cosine_schedule逻辑
    raise NotImplementedError("TODO: 完成get_lr_cosine_schedule")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out_path: str,
) -> None:
    # TODO: 任务6
    # 复用你在adapters里的run_save_checkpoint逻辑
    raise NotImplementedError("TODO: 完成save_checkpoint")


def load_checkpoint(
    src_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    # TODO: 任务7
    # 复用你在adapters里的run_load_checkpoint逻辑
    raise NotImplementedError("TODO: 完成load_checkpoint")


def evaluate(
    model: TransformerLM,
    valid_data: np.ndarray,
    cfg: TrainConfig,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(cfg.eval_batches):
            x, y = get_batch(valid_data, cfg.batch_size, cfg.context_length, cfg.device)
            logits = model(x)
            logits = logits.reshape(-1, logits.shape[-1])
            y = y.reshape(-1)
            losses.append(cross_entropy_loss(logits, y).item())
    model.train()
    return float(np.mean(losses))


def build_model(cfg: TrainConfig) -> TransformerLM:
    model = TransformerLM(
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        num_layers=cfg.num_layers,
        rope_theta=cfg.rope_theta,
    )
    return model.to(cfg.device)


def train(cfg: TrainConfig) -> None:
    # 1) 准备
    set_seed(cfg.seed)
    train_data = load_token_array(cfg.train_tokens_path)
    valid_data = load_token_array(cfg.valid_tokens_path)

    # 2) 模型与优化器
    model = build_model(cfg)

    # TODO: 任务8
    # 用你自己的AdamW类替换这里
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.max_learning_rate)

    # 3) 训练主循环
    start_iter = 0
    for it in range(start_iter, cfg.total_iters):
        # 3.1 动态学习率
        lr = get_lr_cosine_schedule(
            it=it,
            max_learning_rate=cfg.max_learning_rate,
            min_learning_rate=cfg.min_learning_rate,
            warmup_iters=cfg.warmup_iters,
            cosine_cycle_iters=cfg.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 3.2 取batch并前向
        x, y = get_batch(train_data, cfg.batch_size, cfg.context_length, cfg.device)
        logits = model(x)

        # 3.3 计算loss
        logits = logits.reshape(-1, logits.shape[-1])
        y = y.reshape(-1)
        loss = cross_entropy_loss(logits, y)

        # 3.4 反向与更新
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_gradients(model.parameters(), cfg.max_grad_norm)
        optimizer.step()

        # 3.5 周期评估与保存
        if (it + 1) % cfg.eval_interval == 0:
            val_loss = evaluate(model, valid_data, cfg)
            print(f"iter={it + 1} train_loss={loss.item():.4f} val_loss={val_loss:.4f} lr={lr:.6e}")
            Path(cfg.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, it + 1, cfg.checkpoint_path)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Training entry for CS336 assignment")
    parser.add_argument("--train_tokens_path", type=str, required=True)
    parser.add_argument("--valid_tokens_path", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--total_iters", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--max_learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--cosine_cycle_iters", type=int, default=2000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/latest.pt")

    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    config = parse_args()
    train(config)
