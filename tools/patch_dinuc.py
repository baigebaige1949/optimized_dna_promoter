# tools/patch_dinuc.py
import re, pathlib

P = pathlib.Path(__file__).resolve().parent.parent / "models" / "transformer_predictor.py"
S = P.read_text(encoding="utf-8")
ORIG = S
changed = []

def already_patched(src: str) -> bool:
    return ("maintainable fusion: concat dinuc" in src) or ("self.enable_dinuc" in src and "self.proj = None" in src)

# 1) 在 DNAEmbedding.__init__ 注入开关与惰性投影属性（放在 dropout 初始化后）
m = re.search(r'(class\s+DNAEmbedding\(nn\.Module\):[\s\S]*?def\s+__init__\([\s\S]*?\):[\s\S]*?self\.dropout[^\n]*\n)', S)
if m and "self.enable_dinuc" not in m.group(0):
    inject = (
        "    # ---- maintainable: dinuc fusion controls ----\n"
        "    self.enable_dinuc = getattr(config, 'enable_dinuc_features', True)\n"
        "    self.proj = None  # nn.Linear(...), lazy init in forward\n"
    )
    S = S[:m.end(1)] + inject + S[m.end(1):]
    changed.append("A) inject flags in __init__")

# 2) 移除无效的 zeros_like 行（下一行会覆盖它）
S2, n = re.subn(
    r'^\s*dinuc_embeds_expanded\s*=\s*torch\.zeros_like\(embeddings\[:,\s*:-1,\s*:embeddings\.size\(-1\)//4\]\)\s*\n',
    '', S, flags=re.M
)
if n:
    S = S2
    changed.append("B) remove redundant zeros_like line")

# 3) 用“拼接 + 线性投影”替换从“# 拼接特征”到 LayerNorm 之间的老逻辑
start = re.search(r'\n\s*#\s*拼接特征[^\n]*\n', S)
end = re.search(r'\n\s*#\s*应用LayerNorm和Dropout[^\n]*\n|\n\s*embeddings\s*=\s*self\.layer_norm\(', S)
if start and end and start.start() < end.start() and "maintainable fusion: concat dinuc" not in S:
    before = S[:start.end()]
    after = S[end.start():]
    fusion = (
        "    # ---- maintainable fusion: concat dinuc -> project back to hidden_size ----\n"
        "    if self.enable_dinuc and dinuc_embeds_padded is not None:\n"
        "        L = embeddings.size(1)\n"
        "        if dinuc_embeds_padded.size(1) < L:\n"
        "            pad_len = L - dinuc_embeds_padded.size(1)\n"
        "            dinuc_embeds_padded = torch.cat([\n"
        "                dinuc_embeds_padded,\n"
        "                torch.zeros(dinuc_embeds_padded.size(0), pad_len, dinuc_embeds_padded.size(-1),\n"
        "                            device=dinuc_embeds_padded.device, dtype=dinuc_embeds_padded.dtype)\n"
        "            ], dim=1)\n"
        "        elif dinuc_embeds_padded.size(1) > L:\n"
        "            dinuc_embeds_padded = dinuc_embeds_padded[:, :L, :]\n"
        "\n"
        "        concat = torch.cat([embeddings, dinuc_embeds_padded], dim=-1)  # [B, L, H + D]\n"
        "        in_features = concat.size(-1)\n"
        "        out_features = embeddings.size(-1)\n"
        "        if (getattr(self, 'proj', None) is None\n"
        "            or not hasattr(self.proj, 'in_features')\n"
        "            or self.proj.in_features != in_features\n"
        "            or self.proj.out_features != out_features):\n"
        "            self.proj = nn.Linear(in_features, out_features, bias=True).to(concat.device)\n"
        "\n"
        "        embeddings = self.proj(concat)\n"
        "\n"
    )
    S = before + fusion + after
    changed.append("C) replace concat block with concat+linear projection")

# 写回
if changed:
    bak = P.with_suffix(".py.bak_dinuc")
    bak.write_text(ORIG, encoding="utf-8")
    P.write_text(S, encoding="utf-8")
    print("✅ patch applied to", P)
    for c in changed:
        print(" -", c)
    print("Backup:", bak)
else:
    if already_patched(S):
        print("ℹ️ already patched. Nothing to do.")
    else:
        print("ℹ️ patterns not found. Your file may differ around the target region.")
        print("   请把这段输出给我：sed -n '260,460p' models/transformer_predictor.py")
