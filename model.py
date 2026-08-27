# DinoV2ViT: clean ViT + 4 register tokens that loads Meta's `dinov2_vit{s,b,g}14_reg`
# pretrained weights via state_dict (no xformers, no dinov2 codebase imports).
# Attention runs on `F.scaled_dot_product_attention` so we get FlashAttention-2
# on H100 bf16 with no third-party kernel dependency. Module names below match
# Meta's checkpoint key layout exactly, so `load_dinov2_pretrained(model)` does
# a strict load.
#
# DINOHead is the small MLP + weight-normed classifier used by train.py for the
# DINO CLS self-distillation loss. It is intentionally trivial
# (~15 lines) so we have zero runtime dependency on the dinov2 codebase.

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.pos_embed_sincos import RotaryEmbeddingDinoV3, apply_rot_embed_cat
from torchvision import transforms


# (dim, depth, heads, pretrain_grid, ffn, pos_has_cls, weight URL[, registers]) for each supported variant.
# Probe readout selection. NANOPATH_PROBE_LAYERS is a comma-separated list of 1-INDEXED blocks
# whose normalized CLS tokens the classification probes concatenate; NANOPATH_PROBE_SEG picks the
# segmentation feature map ("dense" = last four blocks guided-upsampled to 32x32, "plain" = the
# final block at the native 16x16 grid). probe.py rebuilds the model as DinoV2ViT(variant=...) and
# can be passed nothing else, so train.py exports both from cfg["model"] and the probe subprocess
# inherits them. Defaults reproduce this branch: 1-indexed 5,7,9,12 (the hardcoded 0-indexed
# 4,6,8,11) with the dense segmentation map.
# NANOPATH_PROBE_POOL adds spatially-pooled context to each tapped layer's classification vector:
# "cls" is CLS alone, "patch" appends the mean over that layer's patch tokens, "reg" appends the
# mean over its four register tokens. Combine with "+" (e.g. "cls+patch"); order follows cls, patch, reg.
def probe_readout():
    return (
        tuple(int(v) for v in os.environ.get("NANOPATH_PROBE_LAYERS", "5,7,9,12").split(",")),
        os.environ.get("NANOPATH_PROBE_SEG", "dense"),
        os.environ.get("NANOPATH_PROBE_POOL", "cls"),
    )


DINOV2_VARIANTS = {
    "dinov2_vits14_reg": (384, 12, 6, 37, "mlp", True, "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth"),
    "dinov2_vitb14_reg": (768, 12, 12, 37, "mlp", True, "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth"),
    "dinov2_vitg14_reg": (1536, 40, 24, 37, "swiglu", True, "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth"),
    # DINOv3 ViT-S/16. Positions come from axial RoPE applied to q/k inside attention rather than a
    # learned pos_embed, qkv carries no bias, and LayerNorm eps is 1e-5 not 1e-6. Weights are the
    # ungated timm mirror of facebook/dinov3-vits16-pretrain-lvd1689m; DINOv3 is LVD-1689M (natural
    # images, no pathology), so it is an allowed initialisation. Trailing slots: registers, patch, rope temp.
    "dinov3_vits16": (384, 12, 6, 0, "mlp", False, "timm/vit_small_patch16_dinov3.lvd1689m", 4, 16, 100.0),
}


def probe_transforms():
    # Default for Nanopath-trained checkpoints; baseline scripts override this in their request config.
    transform = transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.ToTensor()])
    # Keep the two return slots because probe.py separates tile-image and slide/patch-bag probes.
    return transform, transform


# Stochastic depth: keep_prob bernoulli on the residual branch, scaled to preserve mean.
class DropPath(nn.Module):
    def __init__(self, p): super().__init__(); self.p = float(p)
    def forward(self, x):
        if self.p == 0.0 or not self.training: return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.shape[0], 1, 1).bernoulli_(keep)
        return x * mask / keep


# Per-channel learnable scale on residual branches; matches Meta's `ls1.gamma`/`ls2.gamma`.
class LayerScale(nn.Module):
    def __init__(self, dim): super().__init__(); self.gamma = nn.Parameter(torch.ones(dim))
    def forward(self, x): return x * self.gamma


# FINO gradient gate: identity forward, scales the gradient by `scale` on backward. sign>0 encourages the
# encoder to predict a metadata factor (M+); sign<0 reverses the gradient to suppress it (M-, DANN-style).
class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale): ctx.scale = scale; return x
    @staticmethod
    def backward(ctx, g): return g * ctx.scale, None


# Attention with single qkv Linear + F.scaled_dot_product_attention (Flash-2 backend on H100 bf16).
class Attention(nn.Module):
    def __init__(self, dim, heads, qkv_bias=True, n_prefix=0):
        super().__init__()
        self.heads, self.n_prefix = heads, n_prefix
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True)

    # `rope` is DINOv3's (HW, 2*head_dim) sin/cos table; it rotates q/k for patch tokens only,
    # leaving the cls and register prefix unrotated (they have no spatial position).
    def forward(self, x, rope=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if rope is not None:
            n = self.n_prefix
            q = torch.cat([q[:, :, :n], apply_rot_embed_cat(q[:, :, n:], rope, half=True)], dim=2).type_as(v)
            k = torch.cat([k[:, :, :n], apply_rot_embed_cat(k[:, :, n:], rope, half=True)], dim=2).type_as(v)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        hidden = (int(hidden * 2 / 3) + 7) // 8 * 8
        self.w12 = nn.Linear(dim, 2 * hidden, bias=True)
        self.w3 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


# Standard pre-LN block: attn + ls1 + drop_path, then mlp + ls2 + drop_path.
class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, drop_path_p, ffn="mlp", eps=1e-6, qkv_bias=True, n_prefix=0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(dim, heads, qkv_bias=qkv_bias, n_prefix=n_prefix)
        self.ls1 = LayerScale(dim)
        self.drop_path1 = DropPath(drop_path_p)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = SwiGLU(dim, hidden) if ffn == "swiglu" else nn.Sequential()
        if ffn == "mlp":
            self.mlp.fc1 = nn.Linear(dim, hidden, bias=True)
            self.mlp.fc2 = nn.Linear(hidden, dim, bias=True)
        self.ls2 = LayerScale(dim)
        self.drop_path2 = DropPath(drop_path_p)

    def _ff(self, x): return self.mlp(x) if isinstance(self.mlp, SwiGLU) else self.mlp.fc2(F.gelu(self.mlp.fc1(x)))

    def forward(self, x, rope=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), rope)))
        x = x + self.drop_path2(self.ls2(self._ff(self.norm2(x))))
        return x


# ViT-S/B-14 with 4 register tokens; key layout matches Meta's DINOv2 register checkpoints
# (cls_token, register_tokens, pos_embed (1, 1+37^2, dim), mask_token (1, dim), patch_embed.proj,
# blocks.{i}.{norm1,norm2,attn.qkv,attn.proj,ls1,ls2,mlp.fc1,mlp.fc2}, norm).
# Pos embed is bicubically interpolated at runtime to the current patch grid.
# Meta DINOv2 includes a cls pos and uses 37x37 patches; variant_cfg can override this for probes.
class DinoV2ViT(nn.Module):
    def __init__(self, variant="dinov2_vits14_reg", drop_path_rate=0.0, variant_cfg=None):
        super().__init__()
        cfg = variant_cfg or DINOV2_VARIANTS[variant]
        dim, depth, heads, pretrain_grid, ffn, pos_has_cls, _ = cfg[:7]
        registers = cfg[7] if len(cfg) > 7 else 4
        mlp_ratio, patch = 4.0, cfg[8] if len(cfg) > 8 else 14
        rope_temp = cfg[9] if len(cfg) > 9 else None
        eps = 1e-5 if rope_temp else 1e-6
        self.variant = variant
        self.patch_size, self.registers, self.embed_dim = patch, registers, dim
        self._pretrain_grid, self._pos_has_cls = pretrain_grid, pos_has_cls
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, registers, dim))
        # RoPE variants carry no positional parameters at all; the sin/cos table is derived per grid.
        self.rope = RotaryEmbeddingDinoV3(dim // heads, temperature=rope_temp, normalize_coords="separate") if rope_temp else None
        self.pos_embed = None if rope_temp else nn.Parameter(torch.zeros(1, int(pos_has_cls) + pretrain_grid**2, dim))
        # DINOv3 ships no mask_token, so JEPA masking gets a freshly initialised one either way.
        self.mask_token = nn.Parameter(torch.zeros(1, dim))
        rates = [drop_path_rate * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList(
            Block(dim, heads, mlp_ratio, p, ffn=ffn, eps=eps, qkv_bias=rope_temp is None, n_prefix=1 + registers)
            for p in rates
        )
        self.norm = nn.LayerNorm(dim, eps=eps)

    # Bicubic resample of the checkpoint patch-pos grid to the current (h, w) grid.
    def _interpolate_pos_embed(self, h, w):
        cls_pos = self.pos_embed[:, :1] if self._pos_has_cls else None
        g = self._pretrain_grid
        patch_pos = self.pos_embed[:, int(self._pos_has_cls):].reshape(1, g, g, -1).permute(0, 3, 1, 2).float()
        # antialias=True matches Meta's default for DINOv2 `_reg` variants.
        patch_pos = F.interpolate(patch_pos, size=(h, w), mode="bicubic", align_corners=False, antialias=True)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1).to(self.pos_embed.dtype)
        return torch.cat([cls_pos, patch_pos], dim=1) if cls_pos is not None else patch_pos

    # Build [cls, registers, patches] tokens; masked patch positions are replaced by mask_token.
    def _prepare_tokens(self, x, masks=None):
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).expand_as(x), x)
        cls = self.cls_token.expand(B, -1, -1)
        regs = self.register_tokens.expand(B, -1, -1)
        if self.rope is not None:
            return torch.cat([cls, regs, x], dim=1)
        if self._pos_has_cls:
            x = torch.cat([cls, x], dim=1) + self._interpolate_pos_embed(h, w)
            return torch.cat([x[:, :1], regs, x[:, 1:]], dim=1)
        return torch.cat([cls, regs, x + self._interpolate_pos_embed(h, w)], dim=1)

    # Per-grid RoPE sin/cos table (None for the learned-pos_embed DINOv2 variants).
    def _rope_embed(self, x):
        if self.rope is None:
            return None
        return self.rope.get_embed([x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size]).to(x.dtype)

    # Returns the dict shape Meta's `forward_features` returns; used by train.py and probe.py.
    # `checkpoint=True` re-runs each block under torch.utils.checkpoint to trade compute for memory;
    # useful when the 1-GPU batch of 128 (2 globals + 8 locals) does not fit in 80 GB.
    def forward(self, x, masks=None, checkpoint=False):
        rope = self._rope_embed(x)
        x = self._prepare_tokens(x, masks)
        for blk in self.blocks:
            if checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, rope, use_reentrant=False)
            else:
                x = blk(x, rope)
        x = self.norm(x)
        return {
            "x_norm_clstoken": x[:, 0],
            "x_norm_regtokens": x[:, 1 : 1 + self.registers],
            "x_norm_patchtokens": x[:, 1 + self.registers :],
        }

    # Probe readouts fuse intermediate normalized tokens: denser patch detail for seg,
    # and strided-depth CLS features that are less tied to the final DINO head.
    def encode_image(self, x, checkpoint=False):
        if probe_readout()[1] == "plain":
            out = self(x, checkpoint=checkpoint)
            return torch.cat([out["x_norm_regtokens"], out["x_norm_patchtokens"]], dim=1)
        B, _, H, W = x.shape
        h, w, G = H // self.patch_size, W // self.patch_size, 32
        guide = x.mean(1, keepdim=True)
        guide = (guide - guide.amin((2, 3), keepdim=True)) / (guide.amax((2, 3), keepdim=True) - guide.amin((2, 3), keepdim=True) + 1e-6)
        rope = self._rope_embed(x)
        xt, feats = self._prepare_tokens(x), []
        for i, blk in enumerate(self.blocks):
            xt = torch.utils.checkpoint.checkpoint(blk, xt, rope, use_reentrant=False) if checkpoint and self.training else blk(xt, rope)
            if i >= len(self.blocks) - 4:
                feats.append(self.norm(xt)[:, 1:])
        fused = torch.cat(feats, -1)
        regs, patches = fused[:, :self.registers], fused[:, self.registers:]
        up = F.interpolate(patches.transpose(1, 2).reshape(B, patches.shape[-1], h, w).float(), size=(G, G), mode="bilinear", align_corners=False)
        guide_lr = F.interpolate(guide, size=(h, w), mode="area")
        guide_hr = F.interpolate(guide, size=(G, G), mode="area")
        w_range = torch.exp(-((guide_hr - F.interpolate(guide_lr, size=(G, G), mode="nearest")).abs() ** 2) / 0.02)
        blur = F.avg_pool2d(F.pad(up, (1, 1, 1, 1), mode="replicate"), 3, 1)
        dense = (up + (1 - w_range) * (up - blur)).flatten(2).transpose(1, 2).to(fused.dtype)
        return torch.cat([regs, dense], dim=1)

    # Blocks are cached by 1-indexed depth then gathered in order, so a depth may repeat (e.g.
    # 12,12,12,12 widens the feature without adding information -- the control for probe width).
    def probe_features(self, x):
        layers, _, pool = probe_readout()
        parts = pool.split("+")
        rope = self._rope_embed(x)
        xt, cache = self._prepare_tokens(x), {}
        for i, blk in enumerate(self.blocks, start=1):
            xt = blk(xt, rope)
            if i in set(layers):
                t = self.norm(xt)
                cache[i] = torch.cat(
                    [t[:, 0]]
                    + ([t[:, 1 + self.registers :].mean(1)] if "patch" in parts else [])
                    + ([t[:, 1 : 1 + self.registers].mean(1)] if "reg" in parts else []),
                    dim=-1,
                )
        return torch.cat([cache[i] for i in layers], dim=-1)


# Strict-load Meta's pretrained weights for the model's declared variant.
# Strict matches our key layout against Meta's; any drift fails loudly per AGENTS.md.
def load_dinov2_pretrained(model):
    source = DINOV2_VARIANTS[model.variant][6]
    if model.rope is None:
        state = torch.hub.load_state_dict_from_url(source, progress=False, map_location="cpu")
        model.load_state_dict(state, strict=True)
        return model
    # DINOv3: same block internals under different names, and no mask_token in the checkpoint.
    from huggingface_hub import hf_hub_download

    state = torch.load(hf_hub_download(source, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    state = {k.replace("gamma_1", "ls1.gamma").replace("gamma_2", "ls2.gamma"): v for k, v in state.items()}
    state["register_tokens"] = state.pop("reg_token")
    state["mask_token"] = model.mask_token.detach().clone()
    model.load_state_dict(state, strict=True)
    return model


# DINO projection head: 3-layer MLP (in -> hidden -> hidden -> bottleneck) + L2 norm +
# weight-normed Linear(bottleneck -> n_prototypes) with weight_g frozen at 1, matching the
# behaviour of dinov2.layers.DINOHead. Standalone reimplementation (no xformers, no fvcore).
class DINOHead(nn.Module):
    def __init__(self, in_dim, n_prototypes, hidden_dim=2048, bottleneck_dim=384, nlayers=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(nlayers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, n_prototypes, bias=False))
        # weight-norm under torch.nn.utils.parametrizations exposes `parametrizations.weight.original0/1`;
        # original0 is the magnitude vector (size n_prototypes). Freeze it at 1 to match dinov2's recipe.
        with torch.no_grad():
            self.last_layer.parametrizations.weight.original0.fill_(1.0)
        self.last_layer.parametrizations.weight.original0.requires_grad_(False)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


# I-JEPA predictor head: regresses EMA-teacher patch representations at masked target blocks from the student's
# block-masked patch tokens. FINO/JEPA-T option: n_cond>0 adds a learned per-class embedding (idx 0 = missing/-1)
# of a discrete metadata factor to every patch token, so the latent-regression target is metadata-aware
# (a dense-path alternative to CLS-token steering). n_cond=0 is plain I-JEPA.
class JEPAPredictor(nn.Module):
    def __init__(self, dim, depth=4, width=0, heads=6, n_cond=0):
        super().__init__()
        width = width or dim
        self.proj_in = nn.Linear(dim, width) if width != dim else nn.Identity()
        self.cond_emb = nn.Embedding(n_cond + 1, width) if n_cond else None
        self.blocks = nn.ModuleList(Block(width, heads, 4.0, 0.0) for _ in range(depth))
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.proj = nn.Linear(width, dim, bias=True)

    def forward(self, patch_tokens, cond=None):
        x = self.proj_in(patch_tokens)
        if self.cond_emb is not None and cond is not None:
            x = x + self.cond_emb(cond + 1).unsqueeze(1)  # broadcast factor embedding over patches; cond=-1 -> idx 0
        for blk in self.blocks:
            x = blk(x)
        return self.proj(self.norm(x))
