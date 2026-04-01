import torch
import torch.nn as nn
import math
from einops import einsum, rearrange, reduce

# Transformer快速复习地图（建议按这个顺序看）:
# 1) Embedding: token_id -> 向量
# 2) TransformerBlock: Norm -> Attention/FFN -> 残差
# 3) 堆叠多个Block后再做Norm + 输出到词表logits


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        # 初始化权重
        super().__init__()
        # 权重 W 存储为原矩阵(dout, din)，而非转置. 使用时右乘x，如 W @ x
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype)) 
        self.init_param()
        
        
    def init_param(self):
        
        mean = 0
        d_out, d_in = self.weight.shape
        std = math.sqrt(2/(d_in + d_out))
        nn.init.trunc_normal_(self.weight, mean=mean, std=std, a=-3*std, b=3*std)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # pytorch中必须把向量视为行向量
        # return x @ self.weight.T
        # 但是对于einsum就无所谓
        # (复习-挖空): 补全线性层的einsum维度映射
        return einsum(self.weight, x, 'o i, b s i -> b s o')
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.embeddings = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self.d_model = embedding_dim
        self.init_embeddings()
        
    def init_embeddings(self):
        mean = 0
        std = 1
        nn.init.trunc_normal_(self.embeddings, mean=mean, std=std, a=-3, b=3)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # (复习-挖空): 根据token_ids做embedding查表
        return self.embeddings[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.gain = nn.Parameter(torch.randn(d_model, device=device, dtype=dtype) / math.sqrt(d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        in_dtype = x.dtype
        x = x.float()

        # TODO(复习-挖空): 补全RMSNorm核心计算（平方、归一化因子、缩放）
        # 要求: 结果转回输入dtype
        rms = torch.sqrt(((x**2).mean(dim=-1, keepdim=True) + self.eps))
        out = x * self.gain / rms
        return out.to(in_dtype)
    
def SiLU(x: torch.Tensor):
    # TODO(复习-挖空): 写出SiLU定义
    return torch.sigmoid(x) * x
    
class SwiGLU(nn.Module):
    def __init__(self, d_model, dff=None):
        super().__init__()
        if not dff:
            self.dff = 8  * d_model // 3
        else:
            self.dff = dff
        self.w1 = Linear(d_model, self.dff)
        self.w2 = Linear(self.dff, d_model)
        self.w3 = Linear(d_model, self.dff)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(复习-挖空): 补全SwiGLU前向
        # 提示: 两个分支 + 门控 + 输出投影
        gated = SiLU(self.w1(x))
        projected = self.w3(x)
        return self.w2(gated * projected)
    
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0
        self.d_k = d_k
        self.device = device
        
        # 构建逆频率
        # TODO(复习): 试着先自己写，再和这一行对照
        inv_freqs = theta ** (- torch.arange(0, d_k, 2, dtype=torch.float32) / self.d_k)
        
        # 构建频率和位置
        pos = torch.arange(0, max_seq_len, dtype=torch.float32) # (max_seq_len)
        
        # 构建sin和cos
        # 我们需要的sin和cos的形状是什么?
        # 需要能和 x 进行计算, 那么他们都是2d向量
        freqs = pos.unsqueeze(1) @ inv_freqs.unsqueeze(0)
        # 轮椅写法
        # freqs = einsum(pos, inv_freqs, 's, d -> s d')

        cos = torch.cos(freqs) # seq d/2
        sin = torch.sin(freqs) # seq d/2
        self.cos: torch.Tensor
        self.sin: torch.Tensor
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # import pdb; pdb.set_trace()
        # 分组
        x_even = x[..., 0::2] # b, s, d/2
        x_odd = x[..., 1::2]

        # 获取 sin/cos
        cos = self.cos[token_positions] # s, d/2
        sin = self.sin[token_positions]
        
        # 进行旋转
        # TODO(复习-挖空): 补全旋转公式，并按偶/奇位置写回输出
        y_even = x_even * cos - x_odd * sin
        y_odd = x_even * sin + x_odd * cos
        
        y = torch.zeros(x.shape, device=self.device, dtype=torch.float32)
        y[..., 0::2] = y_even
        y[..., 1::2] = y_odd
        
        return y
    
def softmax(in_features: torch.Tensor, dim: int):
    # 写出数值稳定版softmax
    mx = in_features.max(dim=dim, keepdim=True).values
    exp = torch.exp(in_features - mx)
    return exp / exp.sum(dim=dim, keepdim=True)
    

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None) -> torch.Tensor:
    # import pdb; pdb.set_trace()
    # TODO(复习-挖空): 补全缩放点积注意力
    _, _, seq_q, d_k = query.shape # b, h, s, d
    _, _, seq_k, _ = key.shape # b, h, s, d

    pre_sm_attn = query @ key.transpose(-1, -2) / math.sqrt(d_k)

    if mask is not None:
        pre_sm_attn = pre_sm_attn.masked_fill(mask == False, float('-inf'))
    
    attn = softmax(pre_sm_attn, dim=-1)
    
    return attn @ value

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, d_model)
        self.Wv = Linear(d_model, d_model)
        self.Wo = Linear(d_model, d_model)
        self.rope = rope
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        token_positions = torch.arange(seq_len, device=x.device)

        # 流程总览: x -> QKV投影 -> 分头 -> (可选RoPE) -> 因果注意力 -> 合并头 -> Wo
        # TODO(复习-挖空): 按上面的流程补完整个前向
        raise NotImplementedError("TODO: 完成MultiHeadSelfAttention.forward")
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, theta=10000.0, max_seq_len=1024):
        super().__init__()
        self.rope = RoPE(theta, d_model//num_heads, max_seq_len)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, self.rope)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln1 = RMSNorm(d_model=d_model)
        self.ln2 = RMSNorm(d_model=d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(复习-挖空): 按Pre-Norm残差结构补全
        raise NotImplementedError("TODO: 完成TransformerBlock.forward")
    
class TransformerLM(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, vocab_size:int, context_length:int, num_layers:int, rope_theta:float):
        super().__init__()
        self.embd = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, theta=rope_theta, max_seq_len=context_length) for i in range(num_layers)])
        self.ln = RMSNorm(d_model=d_model)
        self.output_embd = Linear(d_model, vocab_size)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # import pdb; pdb.set_trace()
        # TODO(复习-挖空): 按主流程补全语言模型前向
        # 注意: forward输出logits，不在这里做softmax
        raise NotImplementedError("TODO: 完成TransformerLM.forward")