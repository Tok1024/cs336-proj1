from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups: # 这个 param_groups 是参数组，代表模型不同模块的参数可以有不同的学习率
            lr = group["lr"]  # Get the learning rate.
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                
                # Update weight tensor in-place.
                p.data -= lr / math.sqrt(t + 1) * grad
                
                state["t"] = t + 1  # Increment iteration number.
            
            
        
        return loss
    
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, lamd=1e-2, epsilon=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "beta1": beta1, "beta2": beta2, "lamd": lamd, "epsilon": epsilon}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups: # 这个 param_groups 是参数组，代表模型不同模块的参数可以有不同的学习率
            lr = group["lr"]  # Get the learning rate.
            beta1 = group["beta1"] # 一阶动量平滑系数
            beta2 = group["beta2"] # 二阶动量平滑系数
            lamd = group["lamd"] # weight decay
            epsilon = group["epsilon"] # 分母
            
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # 1. 取state
                state = self.state[p]  # state 是一个map，从参数param映射到字典
                t = state.get("t", 1)  # Get iteration number from the state, or initial value.
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                
                # 2. 更新动量
                m = beta1 * m + (1 - beta1) * m
                v = beta2 * v + (1 - beta1) * v
                
                # 3. 更新参数
                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= alpha_t * m / (torch.sqrt(v) + epsilon)
                
                # 4. weight decay(正则化)
                p.data -= lr * lamd * p.data
                
                # 5. 更新状态
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        
        return loss

# 初始化参数和优化器
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
target = torch.diag(torch.ones(10))
opt = SGD([weights], lr=10)

# 训练循环
for t in range(100):
    opt.zero_grad()  # Reset the gradients for all learnable parameters.
    loss = ((weights - target)**2).mean()  # Compute a scalar loss value.
    print(f"Step {t:3d} | Loss: {loss.cpu().item():.6f}")
    loss.backward()  # Run backward pass, which computes gradients.
    opt.step()  # Run optimizer step.
