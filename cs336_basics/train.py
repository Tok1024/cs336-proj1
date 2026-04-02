import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cs336_basics.model import TransformerLM


# 对齐实验要求的任务地图（先过单测，再接入train主循环）:
# A1 -> tests/test_data.py::test_get_batch
# A2 -> tests/test_nn_utils.py::test_cross_entropy
# A3 -> tests/test_nn_utils.py::test_gradient_clipping
# A4 -> tests/test_optimizer.py::test_get_lr_cosine_schedule
# A5 -> tests/test_optimizer.py::test_adamw
# A6 -> tests/test_serialization.py::test_checkpointing
# A7 -> 训练主循环集成（本文件train/evaluate）

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
    # TODO(A7-准备): 按你的数据格式读取为1D token id数组
    # 建议先支持.npy；若你有.bin可再扩展。
    # 要求: 返回dtype可用于torch.long，且shape为(N,)。
    raise NotImplementedError("TODO: 完成load_token_array")


def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO(A1): 与tests/test_data.py对齐
    # 1) 起点均匀采样于[0, len(dataset)-context_length)
    # 2) x.shape == y.shape == (batch_size, context_length)
    # 3) y始终是x右移一位（逐元素满足 y = x + 1 在该测试构造下）
    # 4) 返回torch.long并放到device
    raise NotImplementedError("TODO: 完成get_batch")


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # TODO(A2): 与tests/test_nn_utils.py::test_cross_entropy对齐
    # 输入形状: [B, V] 与 [B]；输出标量平均loss。
    # import pdb; pdb.set_trace()
    # 1. 普通写法，转换到概率 prob 空间，指数容易溢出！
    # logits = logits - logits.max(dim=-1, keepdim=True).values
    # exp = torch.exp(logits)
    # probs = exp[torch.arange(logits.shape[0]), targets] / exp.sum(dim=-1) # 这里我们的targets需要给最后一位取元素
    # neg_log = - torch.log(probs)
    # nlh = neg_log.mean()
    # 2. 聪明的写法，把log和指数合并一下, 在logits空间计算
    logits = logits - logits.max(dim=-1, keepdim=True).values
    correct_logits = logits[torch.arange(logits.shape[0]), targets] # (B)
    log_sum = torch.log(torch.exp(logits).sum(dim=-1)) # (B)
    nlh = -(correct_logits - log_sum).mean()
    return nlh

def clip_gradients(parameters, max_l2_norm: float) -> None:
    # TODO(A3): 与tests/test_nn_utils.py::test_gradient_clipping对齐
    # 行为应与torch.nn.utils.clip_grad_norm_一致（忽略grad=None参数）。
    raise NotImplementedError("TODO: 完成clip_gradients")


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    # TODO(A4): 与tests/test_optimizer.py::test_get_lr_cosine_schedule对齐
    # 1) warmup段: 线性从0升到max_learning_rate
    # 2) 余弦段: 从max衰减到min
    # 3) 余弦结束后: 固定min_learning_rate
    raise NotImplementedError("TODO: 完成get_lr_cosine_schedule")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out_path: str,
) -> None:
    # TODO(A6-保存): 与tests/test_serialization.py::test_checkpointing对齐
    # 保存model/optimizer/iteration三项状态。
    raise NotImplementedError("TODO: 完成save_checkpoint")


def load_checkpoint(
    src_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    # TODO(A6-加载): 与tests/test_serialization.py::test_checkpointing对齐
    # 正确恢复model与optimizer状态，并返回iteration。
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

    # TODO(A5): 用你在adapters里实现的AdamW替换这里
    # 要求与tests/test_optimizer.py::test_adamw行为一致。
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
