import re, pathlib, textwrap, math

fp = pathlib.Path(__file__).resolve().parents[1] / "models" / "conditional_diffusion_model.py"
src = fp.read_text(encoding="utf-8")
orig = src

# 规范 CrossAttention：以 query_dim 为准；q/k/v -> heads*head_dim，再 reshape 成 [B,H,L,D]
new_cls = textwrap.dedent('''
class CrossAttention(nn.Module):
    """
    Standard multi-head cross-attention:
      - query_dim 决定 head_dim（默认 query_dim // heads，不整除则 ceil）
      - to_q / to_k / to_v : Linear(·, heads * head_dim)
      - 前向兼容输入形状 [B, L, C] 或 [B, C, L]，输出布局与输入一致
    """
    def __init__(self, query_dim: int, cond_dim: int, heads: int = 8, head_dim: int | None = None, bias: bool = False):
        super().__init__()
        self.heads = int(heads)
        if head_dim is None:
            self.head_dim = query_dim // self.heads if (query_dim % self.heads) == 0 else math.ceil(query_dim / self.heads)
        else:
            self.head_dim = int(head_dim)
        inner = self.heads * self.head_dim

        self.to_q = nn.Linear(query_dim, inner, bias=bias)
        self.to_k = nn.Linear(cond_dim, inner, bias=bias)
        self.to_v = nn.Linear(cond_dim, inner, bias=bias)
        self.out  = nn.Linear(inner, query_dim, bias=bias)

    @staticmethod
    def _ensure_seq_last(x: torch.Tensor, expected_channels: int) -> tuple[torch.Tensor, bool]:
        # 返回 (seq_last_tensor, was_channels_first)
        if x.dim() != 3:
            raise ValueError(f"CrossAttention expects 3D tensor, got {x.shape}")
        if x.size(-1) == expected_channels:
            return x, False  # [B, L, C]
        if x.size(1) == expected_channels:
            return x.transpose(1, 2), True  # [B, C, L] -> [B, L, C]
        # 回退：假设已是 [B, L, C]
        return x, False

    def forward(self, h: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        import torch
        B = h.size(0)
        # 统一到 [B, L, C]
        h_seq, was_cf = self._ensure_seq_last(h, expected_channels=self.to_q.in_features)
        c_seq, _      = self._ensure_seq_last(condition, expected_channels=self.to_k.in_features)

        Lq, Lk = h_seq.size(1), c_seq.size(1)
        H, D   = self.heads, self.head_dim
        inner  = H * D

        # 线性 -> [B, L, inner] -> [B, H, L, D]
        q = self.to_q(h_seq).view(B, Lq, H, D).permute(0, 2, 1, 3)
        k = self.to_k(c_seq).view(B, Lk, H, D).permute(0, 2, 1, 3)
        v = self.to_v(c_seq).view(B, Lk, H, D).permute(0, 2, 1, 3)

        # 注意力
        scale = (D ** -0.5)
        attn  = torch.matmul(q, k.transpose(-2, -1)) * scale   # [B, H, Lq, Lk]
        attn  = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)                              # [B, H, Lq, D]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, Lq, inner)  # [B, Lq, inner]
        out = self.out(out)                                      # [B, Lq, query_dim]

        # 返回与输入一致的布局
        return out.transpose(1, 2) if was_cf else out
''').lstrip()

# 1) 定位并替换原 CrossAttention 类体
cls_pat = re.compile(r'\nclass\s+CrossAttention\s*\(nn\.Module\)\s*:\s*\n(?:.*?\n)*?(?=\nclass\s+|\Z)', re.S)
if not cls_pat.search(src):
    raise SystemExit("未找到原始 CrossAttention 类定义，请把该类附近 80 行代码发我以定制匹配。")

src = cls_pat.sub("\n" + new_cls + "\n", src, count=1)

# 2) 移除旧实现中依赖的 self.scale（如果有调用处），新实现内部已使用运行时 head_dim
# 不强行删除其它地方代码，仅将可能的 'self.scale' 使用替换为局部计算（保险处理）
src = re.sub(r'\bself\.scale\b', '(self.head_dim ** -0.5)', src)

# 3) 写回并备份
bak = fp.with_suffix('.py.bak_xattn_v2')
bak.write_text(orig, encoding='utf-8')
fp.write_text(src, encoding='utf-8')
print("✅ CrossAttention v2 patched")
print("Backup:", bak)
