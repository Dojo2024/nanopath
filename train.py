# Continual DINOv2 pretraining on TCGA tiles (single-GPU). Three loss terms:
# DINO CLS self-distillation (Sinkhorn-Knopp centred teacher targets),
# I-JEPA patch-feature regression, and a KDE uniformity term on the
# L2-normalised CLS tokens. YAML drives the tunable knobs (backbone variant,
# LR + LR scheduler, drop path, layerwise decay, KDE weight + concentration,
# FLOP/sample budgets, batch size); other DINOv2 hyperparameters are hardcoded
# inline at their use sites.

import atexit
import contextlib
import fnmatch
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
from torchvision import transforms
from torchvision.transforms import functional as TF

from dataloader import TCGATileDataset, TILE_SIZE
from model import DINOHead, ViT, GradScale, JEPAPredictor, load_dinov2_pretrained
from probe import (
    completed_probe_summary,
    collect_probe_results,
    prepare_probe_state,
    probe_enabled,
    queue_probe_job,
)


# Prognostic sidecar head (cfg.prog). Gated attention pooling -- ABMIL (Ilse et al. 2018), the standard
# aggregator in computational pathology -- over the student's PATCH tokens, then a linear read-out of the
# per-patient prognostic scalars. Reading patch tokens rather than CLS is the point: the metadata gradient
# shapes the encoder, but the CLS stack that probe_features concatenates is never itself regressed onto
# metadata, which is the trade the FINO discrete branch pays. Predicting per-tile from a slide-level label
# is deliberate weak supervision; the attention gate is what lets the head ignore uninformative tiles.
class ProgHead(nn.Module):
    def __init__(self, dim, out_dim, hidden=256):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.value, self.gate, self.score = nn.Linear(dim, hidden), nn.Linear(dim, hidden), nn.Linear(hidden, 1)
        self.head = nn.Linear(dim, out_dim)

    def forward(self, tokens):
        z = self.norm(tokens)
        attn = self.score(torch.tanh(self.value(z)) * torch.sigmoid(self.gate(z))).softmax(dim=1)
        return self.head((attn * z).sum(1))


# Prefix every console line with wall time and job/process id so SLURM logs are easy to scan.
def console_prefix(): return f"{time.strftime('%H:%M:%S')} {os.environ.get('SLURM_JOB_ID', str(os.getpid()))}"


# Read the YAML recipe and fail before any GPU work if the parquet tile dataset is absent.
# expandvars is necessary to resolve `$USER` for checked-in configs.
def load_config():
    if len(sys.argv) < 2:
        raise ValueError("usage: python train.py <config.yaml> [output_dir=<path>]")
    cfg = yaml.safe_load(os.path.expandvars(Path(sys.argv[1]).read_text()))
    cfg["config_path"] = str(Path(sys.argv[1]).resolve())
    # Optional `key=value` overrides after the config; only output_dir is supported,
    # since it's the run identifier and routinely set per-submission from the CLI.
    for arg in sys.argv[2:]:
        key, _, value = arg.partition("=")
        if key != "output_dir":
            raise ValueError(f"unsupported override {arg!r}; only output_dir=<path> is supported")
        cfg["project"]["output_dir"] = os.path.expandvars(value)
    dataset_dir = Path(cfg["data"]["dataset_dir"])
    if not any(dataset_dir.glob("shard-*.parquet")):
        raise FileNotFoundError(
            f"No parquet shards (shard-*.parquet) under {dataset_dir}. Pull the 4M-tile "
            f"parquet dataset from medarc/nanopath on HF by running "
            f"`python prepare.py {cfg['config_path']} download=True`. Follow the data setup in "
            f"README.md before launching train.py."
        )
    return cfg


# Arm Labless before any GPU work so direct `python train.py ...` gets the same
# no-scope GitHub device login path as the SLURM launcher. Noninteractive runs
# train locally unless the launcher passed a preauthorized token file.
def maybe_arm_labless_autosubmit(cfg, repo_dir):
    token_path = os.environ.get("LABLESS_AUTOSUBMIT_FILE", "")
    eligible = (
        bool(cfg["probe"]["enabled"])
        and int(cfg["probe"]["count"]) > 0
        and int(cfg["train"]["max_train_samples"]) == 1_000_000
        and int(cfg["train"]["max_train_flops"]) == 1_000_000_000_000_000_000
    )
    if token_path:
        atexit.register(lambda p=Path(token_path): p.unlink(missing_ok=True))
        return token_path
    if not eligible:
        return ""
    if not sys.stdin.isatty():
        if not os.environ.get("SLURM_JOB_ID"):
            print(f"{console_prefix()} Labless  no interactive stdin; training will run without auto-submit.", flush=True)
        return ""
    print("This looks like a full Labless-eligible run. Leave either prompt blank to train without auto-submit.", flush=True)
    run_name = input("Labless run name (<=20 chars): ").strip()
    notes = input("Labless notes: ").strip()
    if not run_name or not notes or len(run_name) > 20:
        print("Labless auto-submit skipped; run name and notes are required, and run name must be <=20 chars.", flush=True)
        return ""
    token_path = str(Path(str(Path(cfg["project"]["output_dir"]).expanduser().resolve()) + ".labless_autosubmit.json"))
    status = subprocess.run(
        [sys.executable, str(repo_dir / "labless" / "submit_to_labless.py"), "login_only=true", f"token_output={token_path}", f"run_name={run_name}", f"notes={notes}"],
        cwd=repo_dir,
    ).returncode
    if status != 0:
        print("Labless login did not complete; training will run without auto-submit.", flush=True)
        Path(token_path).unlink(missing_ok=True)
        return ""
    os.environ["LABLESS_AUTOSUBMIT_FILE"] = token_path
    atexit.register(lambda p=Path(token_path): p.unlink(missing_ok=True))
    return token_path


def finish_labless_autosubmit(token_path, output_dir, repo_dir):
    token_file = Path(token_path) if token_path else None
    if token_file is None or not token_file.exists():
        return
    token = json.loads(token_file.read_text())
    status = subprocess.run(
        [
            sys.executable,
            str(repo_dir / "labless" / "submit_to_labless.py"),
            f"output_dir={output_dir.resolve()}",
            f"run_name={token['run_name']}",
            f"notes={token['notes']}",
            f"github_token_file={token_file}",
        ],
        cwd=repo_dir,
    ).returncode
    token_file.unlink(missing_ok=True)
    if status == 2:
        print(f"{console_prefix()} Labless  auto-submit skipped because the completed run did not satisfy submission restrictions.", flush=True)
    elif status != 0:
        raise SystemExit(status)


# Cosine schedule from `start` to `end` over fractional progress in [0, 1].
def cosine_schedule(start, end, frac):
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * min(1.0, max(0.0, frac))))


# Sinkhorn-Knopp centring across this batch, used for DINO teacher targets.
def sinkhorn(x, temp):
    q = torch.exp(x.float() / temp).t()
    b = q.shape[1]
    k = q.shape[0]
    q /= q.sum()
    for _ in range(3):
        q /= q.sum(1, keepdim=True) * k
        q /= q.sum(0, keepdim=True) * b
    return (q * b).t()


# Cross-entropy between teacher distribution and softmax(student / 0.1).
def dino_ce(student, teacher):
    return -(teacher * F.log_softmax(student / 0.1, dim=-1)).sum(-1).mean()


# KDE uniformity loss on L2-normalised CLS tokens.
def kde_loss(x, concentration):
    x = F.normalize(x, p=2, dim=-1)
    sim = concentration * (x @ x.T)
    sim.fill_diagonal_(-float("inf"))
    return torch.logsumexp(sim, dim=1).mean() - math.log(max(1, sim.shape[1] - 1))


# I-JEPA target mask: contiguous square blocks so the predictor must infer missing tissue context.
def make_block_mask(batch, grid, device, n_blocks=4, block_scale=0.10):
    masks = torch.zeros(batch, grid, grid, dtype=torch.bool, device=device)
    side = max(1, round(grid * block_scale ** 0.5))
    for i in range(batch):
        for _ in range(n_blocks):
            top = random.randint(0, grid - side)
            left = random.randint(0, grid - side)
            masks[i, top : top + side, left : left + side] = True
    masks = masks.flatten(1)
    idx = masks.flatten().nonzero().flatten()
    weights = (1 / masks.sum(-1).clamp(min=1)).unsqueeze(-1).expand_as(masks)[masks]
    return masks, idx, weights


# Round-3 idea A7 (CAPI): start each block CENTERED on the grid, then roll it by a uniform random
# (dy, dx) with wrap-around, instead of make_block_mask's clamped-random top-left corner. A
# clamped-random corner under-covers the grid edges (top/left can't exceed grid-side), biasing which
# positions get masked; a centered-then-rolled block covers every position with exactly equal
# probability once wrapped. "Mask the centre, predict the periphery" describes the unrolled starting
# placement -- averaged over the random roll it's a uniform-coverage block, same cost as before.
def make_inverse_block_mask(batch, grid, device, n_blocks=4, block_scale=0.10):
    masks = torch.zeros(batch, grid, grid, dtype=torch.bool, device=device)
    side = max(1, round(grid * block_scale ** 0.5))
    center = (grid - side) // 2
    base = torch.arange(side, device=device)
    for i in range(batch):
        for _ in range(n_blocks):
            dy, dx = random.randint(0, grid - 1), random.randint(0, grid - 1)
            rows = (center + dy + base) % grid
            cols = (center + dx + base) % grid
            masks[i, rows[:, None], cols[None, :]] = True
    masks = masks.flatten(1)
    idx = masks.flatten().nonzero().flatten()
    weights = (1 / masks.sum(-1).clamp(min=1)).unsqueeze(-1).expand_as(masks)[masks]
    return masks, idx, weights


# SiamJEPA: two weight-shared student views over DISJOINT context sets. `masks` marks the block region
# both views must predict; the remaining context is split in half, and each view additionally masks the
# other's half. Disjointness is the point -- it stops either branch copying tokens the other can see.
def split_context(masks):
    half = torch.rand(masks.shape, device=masks.device) < 0.5
    context = ~masks
    return masks | (context & half), masks | (context & ~half)


# AdamW parameter groups with layer-wise LR decay on the backbone:
# block i gets lr * layerwise_decay^(depth - 1 - i); patch_embed gets the deepest decay
# multiplied by patch_embed_lr_mult; biases and norms get no weight decay; the head's
# DINO final weight-norm last_layer parameters get an LR-freeze for the first dino.freeze_last_layer_fraction.
def build_param_groups(student_backbone, student_dino_head, student_predictor, layerwise_decay, patch_embed_lr_mult):
    depth = len(student_backbone.blocks)
    # Coalesce params that share (lr_mult, wd_mult, last_layer) into a single group each (~30 groups
    # instead of one-per-param), so AdamW's foreach path fuses the step across many tensors rather than
    # launching per-parameter kernels. Per-param lr/wd are unchanged, so the optimization is numerically identical.
    coalesced = {}
    modules = ((student_backbone, "backbone"), (student_dino_head, "dino_head"), (student_predictor, "jepa_predictor"))
    for module, kind in modules:
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            lr_mult = 1.0
            if kind == "backbone" and name.startswith("blocks."):
                lr_mult = layerwise_decay ** (depth - 1 - int(name.split(".")[1]))
            elif kind == "backbone" and name.startswith("patch_embed."):
                lr_mult = (layerwise_decay ** depth) * patch_embed_lr_mult
            wd_mult = 0.0 if name.endswith("bias") or "norm" in name or p.ndim < 2 else 1.0
            key = (lr_mult, wd_mult, "last_layer" in name)
            coalesced.setdefault(key, {"params": [], "lr_mult": lr_mult, "wd_mult": wd_mult, "last_layer": key[2]})["params"].append(p)
    return list(coalesced.values())


# EMA-update teacher modules from student modules with a single multiplicative decay.
# Params are fused into two _foreach kernels (mul then add) instead of a Python per-tensor loop;
# numerically identical (pt = pt*m + ps*(1-m) per tensor). Called under torch.no_grad() by the caller.
def update_ema(student_module, teacher_module, momentum):
    teacher_params, student_params = list(teacher_module.parameters()), list(student_module.parameters())
    torch._foreach_mul_(teacher_params, momentum)
    torch._foreach_add_(teacher_params, student_params, alpha=1 - momentum)
    for bs, bt in zip(student_module.buffers(), teacher_module.buffers()):
        bt.copy_(bs)


# Orchestrates one pretraining run: setup, train+probe loop, checkpoint, summary.
def main():
    cfg = load_config()
    repo_dir = Path(__file__).resolve().parent
    labless_autosubmit_file = maybe_arm_labless_autosubmit(cfg, repo_dir)
    train_cfg = cfg["train"]
    dino_cfg = cfg["dino"]
    # FINO metadata-guidance: select factors + signs (float; + encourage M+ / - suppress M-). fino_meta (built or
    # copied beside the dataset by prepare.py) holds per-factor barcode maps + cardinalities (n) / vector dims.
    fino_cfg = cfg["fino"] if (cfg.get("fino") or {}).get("enabled") else None
    fino_disc = [(f, float(s)) for f, s in fino_cfg.get("discrete", [])] if fino_cfg else []
    fino_cont = [(f, float(s)) for f, s in fino_cfg.get("continuous", [])] if fino_cfg else []
    fino_meta = json.loads((Path(cfg["data"]["dataset_dir"]) / "fino_meta.json").read_text()) if fino_cfg else {"n": {}, "cont_dim": {}}
    # FINO two-phase: freeze the backbone (except patch_embed) for the first this-fraction of the run so the DINO/JEPA
    # heads + metadata prototypes/predictors converge against a fixed target before they steer the encoder. 0 = off.
    freeze_backbone_frac = float(dino_cfg.get("freeze_backbone_fraction", 0.0))
    # JEPA-T: optionally condition the JEPA predictor on a discrete factor (must be in fino.discrete so its per-tile
    # label rides in the batch). cond_col indexes that factor's column in batch["meta_disc"].
    jepa_cond = fino_cfg.get("jepa_cond") if fino_cfg else None
    cond_col = [f for f, _ in fino_disc].index(jepa_cond) if jepa_cond else None
    save_every = train_cfg["save_every"]
    save_checkpoints = save_every is not None
    device = torch.device("cuda")
    random.seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])
    torch.manual_seed(train_cfg["seed"])
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    variant = cfg["model"]["type"]
    student_backbone = load_dinov2_pretrained(ViT(variant=variant, drop_path_rate=dino_cfg["drop_path_rate"])).to(device)
    # Idea 7: freeze the stem and the first N blocks. requires_grad=False -- not zeroing grads after the
    # fact -- is what actually removes the backward pass, and that only works if EVERY leaf below block N
    # is frozen: leaving cls_token/pos_embed/patch_embed trainable would force autograd to traverse all
    # blocks to reach them and save nothing. So this necessarily gives up patch_embed adaptation, which
    # the phase-1 freeze deliberately keeps; that trade is what the arm measures.
    freeze_blocks = int(dino_cfg.get("freeze_blocks", 0))
    if freeze_blocks:
        student_backbone.patch_embed.requires_grad_(False)
        for t in ("cls_token", "register_tokens", "pos_embed", "mask_token"):
            getattr(student_backbone, t).requires_grad_(False)
        for blk in student_backbone.blocks[:freeze_blocks]:
            blk.requires_grad_(False)
        n_frozen = sum(p.numel() for p in student_backbone.parameters() if not p.requires_grad)
        print(f"frozen: stem + blocks[:{freeze_blocks}] = {n_frozen/1e6:.1f}M of "
              f"{sum(p.numel() for p in student_backbone.parameters())/1e6:.1f}M params", flush=True)
    teacher_backbone = deepcopy(student_backbone)
    teacher_backbone.train(False)
    for p in teacher_backbone.parameters():
        p.requires_grad = False
    student_dino_head = DINOHead(student_backbone.embed_dim, 131072, dino_cfg["head_hidden_dim"], dino_cfg["head_bottleneck_dim"], 3).to(device)
    teacher_dino_head = deepcopy(student_dino_head)
    # Bootleg targets the teacher's hidden blocks; an empty list keeps the single final-layer target.
    jepa_taps = tuple(int(i) for i in dino_cfg["jepa_target_blocks"]) or (len(student_backbone.blocks),)
    jepa_lossfn = F.mse_loss if dino_cfg["jepa_loss"] == "l2" else F.smooth_l1_loss
    jepa_focal_gamma = float(dino_cfg["jepa_focal_gamma"])  # 0 keeps the plain per-token mean; >0 is scienceguru's focal JEPA
    student_predictor = JEPAPredictor(student_backbone.embed_dim, depth=int(dino_cfg["jepa_pred_depth"]), width=int(dino_cfg["jepa_pred_width"]), n_cond=(fino_meta["n"][jepa_cond] if jepa_cond else 0), scratch=int(dino_cfg["jepa_pred_scratch"]), n_targets=len(jepa_taps)).to(device)
    for p in teacher_dino_head.parameters():
        p.requires_grad = False
    backbone_activated_params = sum(p.numel() for p in student_backbone.parameters() if p.requires_grad)
    # FINO continuous-factor predictors (phi -> vector regressors); their params join the optimizer.
    predictors = {f: nn.Sequential(nn.Linear(student_backbone.embed_dim, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, fino_meta.get("cont_dim", {}).get(f, 1))).to(device) for f, _ in fino_cont}
    # AdamW param groups carry per-parameter LR/WD multipliers (LWD + patch_embed + biases-no-WD).
    # SiamJEPA (arXiv 2607.04044): second weight-shared student view on the disjoint context half, both
    # predicting the same hidden region, plus a stop-gradient alignment term between the two branches.
    # DEVIATION FROM THE PAPER: their alignment is a KL between a learned posterior (conditioned on both
    # branches) and prior (one branch) over a latent Z, with free-bits. That needs distribution heads and
    # reparameterisation; this uses a symmetric stop-gradient KL between the two branches' existing DINO
    # prototype distributions instead, so it adds NO parameters. Same role, simpler estimator.
    siam_kl_weight = float(dino_cfg.get("siam_kl_weight", 0.0))
    siam = bool(dino_cfg.get("siam", False)) or siam_kl_weight > 0
    # Gram anchoring (DINOv3): hold the student's patch-patch cosine-similarity matrix close to a frozen
    # EARLY teacher. Targets the measured dense decline (seg/prog/mut peak at 50% and fall). Taken at the
    # PRETRAINED trunk's last block, so with block expansion the fresh blocks stay free to learn instead of
    # being anchored to a checkpoint where they were still noise.
    gram_weight = float(dino_cfg.get("gram_weight", 0.0))
    gram_layer = int(dino_cfg.get("gram_layer", 12))
    gram_start = float(dino_cfg.get("gram_start_frac", 0.25))
    gram_taps = (gram_layer,) if gram_weight else ()
    gram_teacher = None
    # Round-3 idea A3: cross-site nearest-neighbour positives. A FIFO queue of (teacher CLS, site)
    # pairs from recent batches; each anchor is pulled toward its nearest QUEUE neighbour from a
    # DIFFERENT site (same-site entries masked out). Unlike the ruled-out site-repulsion arm, which
    # pushed same-site tiles apart indiscriminately and collapsed, this only ever pulls -- and only
    # pulls toward whatever the model already judges most similar, so it sculpts the CRoMa geometry
    # (reduce same-label/other-centre distance) without a supervised same-label signal. One [b, Q]
    # matmul per step; site labels are training-time-only (TCGA barcode), nothing at probe time.
    xsite_weight = float(dino_cfg.get("xsite_weight", 0.0))
    xsite_queue_size = int(dino_cfg.get("xsite_queue_size", 8192))
    xsite_queue_feat = xsite_queue_site = xsite_ptr = None
    if xsite_weight:
        xsite_queue_feat = F.normalize(torch.randn(xsite_queue_size, student_backbone.embed_dim, device=device), dim=-1)
        xsite_queue_site = torch.full((xsite_queue_size,), -1, dtype=torch.int64, device=device)
        xsite_ptr = [0]
    # Round-3 idea A4: per-site centering of DINO teacher targets. DINOv2 replaces classic DINO's
    # EMA-center with per-BATCH Sinkhorn-Knopp balancing (see sinkhorn() above), which has no
    # persistent center to swap out -- the adaptation here is ComBat's location step one stage
    # earlier, on the teacher CLS the head/Sinkhorn consume: an EMA running mean per TSS-site bucket
    # (384-d, not the 131072-d head output, to stay cheap) is subtracted before teacher_dino_head, so
    # a site's characteristic offset in representation space stops being learnable target signal.
    # Only the DINO-head branch reads the demeaned copy; t["cls"] itself is untouched for every other
    # consumer (JEPA target, FINO, gram anchor, xsite queue). Site labels are training-time-only.
    site_center = bool(dino_cfg.get("site_center", False))
    SITE_CENTER_BUCKETS = 256
    site_center_mu = site_center_seen = None
    if site_center:
        site_center_mu = torch.zeros(SITE_CENTER_BUCKETS, student_backbone.embed_dim, device=device)
        site_center_seen = torch.zeros(SITE_CENTER_BUCKETS, dtype=torch.bool, device=device)
    # Round-3 idea A5: stain-only sibling view. dataloader.py builds a third view sharing global
    # view 0's exact crop but with a strong, geometry-free stain perturbation; the student's patch
    # tokens for it are pulled toward the teacher's clean-view-0 patch tokens (cosine), which targets
    # dense/patch-level stain invariance directly rather than DINO's CLS-only two-global agreement.
    stain_weight = float(dino_cfg.get("stain_weight", 0.0))
    # BEST-RQ: a FIXED random projection + FIXED codebook turn raw masked patches into discrete targets,
    # so the smooth-L1 latent loss gains a classification term with no extra forward pass and no tokenizer.
    rq_weight = float(dino_cfg.get("rq_weight", 0.0))
    rq_head = rq_proj = rq_code = None
    if rq_weight:
        rq_dim, rq_codes = int(dino_cfg.get("rq_dim", 16)), int(dino_cfg.get("rq_codes", 8192))
        gen = torch.Generator().manual_seed(1234)
        rq_proj = torch.randn(3 * student_backbone.patch_size ** 2, rq_dim, generator=gen).to(device)
        rq_code = F.normalize(torch.randn(rq_codes, rq_dim, generator=gen), dim=-1).to(device)
        rq_head = nn.Linear(student_backbone.embed_dim * len(jepa_taps), rq_codes).to(device)
    # Round-3 idea A6 (CAPI-style clustering targets, beside not instead of the JEPA smooth-L1): a
    # LEARNED, weight-tied prototype matrix (SwAV-style: the same head projects both the teacher
    # target and the student prediction) turns each masked token's target into a Sinkhorn-balanced
    # soft cluster assignment, and the predictor is additionally trained with cross-entropy against
    # it. Unlike BEST-RQ's fixed random codebook (null on this board), the codebook is learned via
    # the student branch's gradient; the target branch stays no_grad so it can't collapse.
    # DEVIATION FROM CAPI: their Sinkhorn balances prototype usage PER MASK POSITION so position
    # can't drive assignment; this balances across the whole step's masked-token batch jointly (the
    # same mechanism as the existing DINO sinkhorn()) to keep the added cost negligible, at the risk
    # of weaker protection against positional shortcuts.
    capi_weight = float(dino_cfg.get("capi_weight", 0.0))
    capi_protos = int(dino_cfg.get("capi_protos", 8192))
    capi_head = nn.Linear(student_backbone.embed_dim * len(jepa_taps), capi_protos, bias=False).to(device) if capi_weight else None
    prog_cfg = cfg["prog"] if (cfg.get("prog") or {}).get("enabled") else None
    prog_weight = float(prog_cfg["weight"]) if prog_cfg else 0.0
    prog_head = ProgHead(student_backbone.embed_dim, len(prog_cfg["targets"])).to(device) if prog_cfg else None
    param_groups = build_param_groups(student_backbone, student_dino_head, student_predictor, dino_cfg["layerwise_decay"], dino_cfg["patch_embed_lr_mult"])
    if predictors:
        param_groups.append({"params": [p for m in predictors.values() for p in m.parameters()], "lr_mult": 1.0, "wd_mult": 1.0, "last_layer": False})
    for aux in (prog_head, rq_head, capi_head):
        if aux is not None:
            param_groups.append({"params": list(aux.parameters()), "lr_mult": 1.0, "wd_mult": 1.0, "last_layer": False})
    opt = torch.optim.AdamW(param_groups, lr=1.0, betas=(0.9, dino_cfg["adam_beta2"]))
    # FINO prototype banks: one unit vector per discrete-factor value, EMA-updated from teacher CLS in compute_losses.
    protos = {f: F.normalize(torch.randn(fino_meta["n"][f], student_backbone.embed_dim, device=device), dim=-1) for f, _ in fino_disc} if fino_cfg else {}
    # FINO grad-equalisation EMA bank (one running grad-norm per factor); init 1.0 -> s_t~1 early. Not checkpointed
    # (mu=0.99 -> ~100-step memory, re-warms quickly on resume). Used only when fino.grad_equalize is set.
    grad_eq_ema = {f: torch.ones((), device=device) for f, _ in (fino_disc + fino_cont)} if fino_cfg else {}
    step = 0
    batch_size = int(train_cfg["batch_size"])
    max_train_samples = int(train_cfg["max_train_samples"])
    robust_norm_tiles = 6144 if train_cfg.get("robust_norm") else 0
    train_sample_budget = max_train_samples - robust_norm_tiles
    examples_seen = 0
    visible_patch_presentations = 0
    train_flops = 0
    output_dir = Path(cfg["project"]["output_dir"])
    wandb_dir = Path(cfg["project"]["wandb_dir"])
    wandb_name = cfg["project"]["name"]
    if labless_autosubmit_file:
        wandb_name = json.loads(Path(labless_autosubmit_file).read_text()).get("run_name") or wandb_name
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    latest_checkpoint_path = output_dir / "latest.pt"
    # Fresh launches always start from scratch and wipe output_dir.
    resume_path = Path(train_cfg["resume"]) if train_cfg["resume"] else None
    if resume_path is None and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    wandb_meta = None
    if resume_path is not None:
        print(f"{console_prefix()} Resume  loading checkpoint: {resume_path}", flush=True)
        # Resume restores training progress, optimizer state, and wandb identity.
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        student_backbone.load_state_dict(checkpoint["model"])
        teacher_backbone.load_state_dict(checkpoint["model_ema"])
        student_dino_head.load_state_dict(checkpoint["dino_head"])
        teacher_dino_head.load_state_dict(checkpoint["dino_head_ema"])
        student_predictor.load_state_dict(checkpoint["predictor"])
        opt.load_state_dict(checkpoint["opt"])
        if fino_cfg:
            protos = {k: v.to(device) for k, v in checkpoint["protos"].items()}
            for f, mdl in predictors.items():
                mdl.load_state_dict(checkpoint["predictors"][f])
        step = int(checkpoint["step"])
        examples_seen = int(checkpoint["examples_seen"])
        visible_patch_presentations = int(checkpoint["visible_patch_presentations"])
        train_flops = int(checkpoint["train_flops"])
        wandb_meta = dict(checkpoint["wandb"])
    wandb_init = {
        "project": "nanopath",
        "name": wandb_name,
        "dir": str(wandb_dir),
        "config": cfg,
        "settings": wandb.Settings(
            console="wrap",
            x_file_stream_transmit_interval=5,
        ),
    }
    if wandb_meta is not None:
        wandb_init["id"] = wandb_meta["id"]
        wandb_init["resume"] = "must"
    wandb_run = wandb.init(**wandb_init)
    for key in ("probe/target_flops", "probe/wall_seconds"):
        wandb_run.define_metric(key, hidden=True, overwrite=True)
    print(
        f"{console_prefix()} Run  start: {wandb_name}  "
        f"config: {cfg['config_path']}  batch_size: {batch_size}  max_train_samples: {max_train_samples}  "
        f"max_train_flops: {train_cfg['max_train_flops']}  "
        f"probe_count: {cfg['probe']['count']}  warmup_fraction: {dino_cfg['warmup_fraction']}  "
        f"lr: {dino_cfg['lr']}  adam_beta2: {dino_cfg['adam_beta2']}  kde_loss_weight: {dino_cfg['kde_loss_weight']}  "
        f"kde_concentration: {dino_cfg['kde_concentration']}  drop_path: {dino_cfg['drop_path_rate']}  "
        f"layerwise_decay: {dino_cfg['layerwise_decay']}",
        flush=True,
    )
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()
    git_remote = subprocess.run(["git", "config", "--get", "remote.origin.url"], cwd=repo_dir, text=True, capture_output=True, check=False).stdout.strip()
    source_id = f"nanopath-source-{wandb_run.id}"
    artifact_ignore = [
        line.strip() for line in (repo_dir / ".gitignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ] + [".git/", "baselines/", "slurm/", "AGENTS.md", "CLAUDE.md"]
    ignored_roots = [output_dir.resolve(), wandb_dir.resolve()]

    def artifact_ignored(path):
        if any(path.resolve().is_relative_to(root) for root in ignored_roots):
            return True
        rel_path = path.relative_to(repo_dir)
        if any(part.startswith(".") for part in rel_path.parts):
            return True
        rel, name = rel_path.as_posix(), path.name
        for pat in artifact_ignore:
            pat = pat.rstrip("/") if pat.endswith("/") else pat
            if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel, pat) or rel == pat or rel.startswith(pat + "/"):
                return True
        return False

    source_files = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = sorted(d for d in dirs if not artifact_ignored(Path(root) / d))
        for name in sorted(files):
            path = Path(root) / name
            if artifact_ignored(path):
                continue
            rel = path.relative_to(repo_dir)
            source_files.append((path, rel))
    source_snapshot_dir = output_dir / "labless_source"
    if source_snapshot_dir.exists():
        shutil.rmtree(source_snapshot_dir)
    for path, rel in source_files:
        target = source_snapshot_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    wandb_meta = {"entity": wandb_run.entity, "project": "nanopath", "id": wandb_run.id, "name": wandb_name, "url": wandb_run.url,
                  "mode": getattr(wandb_run.settings, "mode", ""), "source_artifact": source_id,
                  "source_dir": str(source_snapshot_dir), "git": {"commit": git_commit, "remote": git_remote}}
    train_ds = TCGATileDataset(cfg, is_train=True)
    val_ds = TCGATileDataset(cfg, is_train=False)
    probe_state = prepare_probe_state(cfg, output_dir) if probe_enabled(cfg) else None

    # Train shuffles + drops partials; the loop never starts a batch that would exceed
    # max_train_samples, so every optimizer step keeps the configured batch size.
    loader_kwargs = dict(batch_size=batch_size, drop_last=True, num_workers=train_cfg["num_workers"], pin_memory=True,
                         prefetch_factor=train_cfg["prefetch_factor"] if train_cfg["num_workers"] > 0 else None,
                         persistent_workers=train_cfg["persistent_workers"] and train_cfg["num_workers"] > 0)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    activation_checkpointing = bool(train_cfg["activation_checkpointing"])
    global_grid = train_cfg["global_size"] // student_backbone.patch_size
    global_patches = global_grid ** 2
    block_mask_fn = make_inverse_block_mask if dino_cfg.get("inverse_block_mask", False) else make_block_mask
    local_patches = (train_cfg["local_size"] // student_backbone.patch_size) ** 2
    last_time = time.time()
    last_examples = examples_seen
    last_visible_patch_presentations = visible_patch_presentations
    last_train_flops = train_flops
    unique_tile_patch_count = (TILE_SIZE // student_backbone.patch_size) ** 2
    seen_ids = {"sample": set(), "slide": set(), "patient": set()}
    pending_ids = {key: set() for key in seen_ids}

    # cpu_state(m) materializes an on-CPU copy of a module's state_dict for torch.save.
    def cpu_state(m): return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

    # Full checkpoint (latest.pt) covers explicit train.resume whereas probe checkpoint is a slim
    # weights-only ckpt, given probe.py does not need optimizer or projection heads.
    def checkpoint_payload(next_step, full):
        payload = {"model": cpu_state(student_backbone), "model_ema": cpu_state(teacher_backbone), "step": next_step, "config": cfg}
        if not full:
            return payload
        return {**payload, "dino_head": cpu_state(student_dino_head), "dino_head_ema": cpu_state(teacher_dino_head),
                "predictor": cpu_state(student_predictor), "opt": opt.state_dict(),
                "examples_seen": examples_seen, "visible_patch_presentations": visible_patch_presentations,
                "train_flops": train_flops, "wandb": wandb_meta,
                **({"protos": {k: v.cpu() for k, v in protos.items()}, "predictors": {f: cpu_state(m) for f, m in predictors.items()}} if fino_cfg else {}),
                **({"prog_head": cpu_state(prog_head)} if prog_head is not None else {}),
                **({"rq_head": cpu_state(rq_head)} if rq_head is not None else {}),
                **({"capi_head": cpu_state(capi_head)} if capi_head is not None else {})}

    def save_latest_checkpoint(checkpoint_step):
        nonlocal last_saved_step
        print(f"{console_prefix()} Checkpoint  [{checkpoint_step}]  save: latest.pt", flush=True)
        tmp_path = latest_checkpoint_path.with_suffix(".pt.tmp")
        torch.save(checkpoint_payload(checkpoint_step, full=True), tmp_path)
        os.replace(tmp_path, latest_checkpoint_path)
        for stale_checkpoint_path in output_dir.glob("step_*.pt"):
            stale_checkpoint_path.unlink()
        last_saved_step = checkpoint_step

    # Count unique tiles/slides/patients for data-coverage diagnostics.
    def flush_unique_counts():
        for key in seen_ids:
            seen_ids[key].update(pending_ids[key])
            pending_ids[key].clear()
        unique_tiles_seen = len(seen_ids["sample"])
        return {
            "unique_slides_seen": len(seen_ids["slide"]),
            "unique_patients_seen": len(seen_ids["patient"]),
            "unique_tiles_seen": unique_tiles_seen,
            "unique_patches_seen": unique_tiles_seen * unique_tile_patch_count,
        }

    # Compute (dino_loss, jepa_loss, kde) for one batch of (gf, lf) crops with the given masks +
    # schedule values. Used by both the train step and evaluate() (no_grad).
    def compute_losses(gf, lf, b, masks, mask_idx, mask_w, t_temp, k_scale, ckpt=False, meta=None, cond=None, prog_target=None, masks2=None, site=None, stain_x=None):
        with torch.no_grad():
            t = teacher_backbone(gf, taps=jepa_taps)
            head_input = t["cls"]
            if site_center and site is not None:
                bucket = site.repeat(train_cfg["global_views"]) % SITE_CENTER_BUCKETS
                head_input = t["cls"] - site_center_mu[bucket]
                feat = t["cls"].float()
                upd = torch.zeros_like(site_center_mu).index_add_(0, bucket, feat)
                cnt = torch.zeros(SITE_CENTER_BUCKETS, device=feat.device).index_add_(0, bucket, torch.ones_like(feat[:, 0]))
                seen_now = cnt > 0
                batch_mean = torch.zeros_like(site_center_mu)
                batch_mean[seen_now] = upd[seen_now] / cnt[seen_now].unsqueeze(1)
                first_sight = seen_now & ~site_center_seen
                site_center_mu[first_sight] = batch_mean[first_sight]
                resight = seen_now & site_center_seen
                site_center_mu[resight] = 0.99 * site_center_mu[resight] + 0.01 * batch_mean[resight]
                site_center_seen[seen_now] = True
            t_cls = teacher_dino_head(head_input).chunk(train_cfg["global_views"])
            t_prob = sinkhorn(torch.cat((t_cls[1], t_cls[0])), t_temp).view(2, b, -1)
        sg = student_backbone(gf, masks=masks, checkpoint=ckpt, taps=gram_taps)
        sl = student_backbone(lf, checkpoint=ckpt)
        sg_cls, sl_cls = student_dino_head(sg["cls"]), student_dino_head(sl["cls"])
        L = train_cfg["local_views"]
        local_loss = sum(dino_ce(x, y) for x in sl_cls.chunk(L) for y in t_prob) / (2 * L + 2)
        global_loss = dino_ce(sg_cls, t_prob.flatten(0, 1)) * 2 / (2 * L + 2)
        # Each tapped depth is z-scored over the embedding dim on its own, then concatenated, which is
        # Bootleg's target construction; with one final-layer tap this is the plain I-JEPA target.
        target = torch.cat([F.layer_norm(v.flatten(0, 1), (student_backbone.embed_dim,)) for v in t["tapped"]], dim=-1)[mask_idx]
        pred = student_predictor(sg["patches"], cond).flatten(0, 1)[mask_idx]
        siam_kl, pred2 = sg["cls"].new_zeros(()), None
        if masks2 is not None:
            sg2 = student_backbone(gf, masks=masks2, checkpoint=ckpt, taps=gram_taps)
            pred2 = student_predictor(sg2["patches"], cond).flatten(0, 1)[mask_idx]
            # symmetric stop-gradient alignment between the two branches
            l1, l2 = sg_cls / 0.1, student_dino_head(sg2["cls"]) / 0.1
            siam_kl = siam_kl_weight * 0.5 * (
                F.kl_div(F.log_softmax(l1, -1), F.softmax(l2.detach(), -1), reduction="batchmean")
                + F.kl_div(F.log_softmax(l2, -1), F.softmax(l1.detach(), -1), reduction="batchmean"))
        err = jepa_lossfn(pred, target, reduction="none").mean(-1)
        if pred2 is not None:
            # the paper averages the two LOSSES, not the two predictions; averaging predictions would let
            # the branches cancel each other's errors, which is a different and weaker objective
            err = 0.5 * (err + jepa_lossfn(pred2, target, reduction="none").mean(-1))
        if jepa_focal_gamma:
            # Focal JEPA: detached unit-mean weights err**gamma concentrate the gradient on hard masked
            # tokens while leaving the objective scale unchanged.
            w = err.detach().clamp_min(1e-6).pow(jepa_focal_gamma)
            err = err * (w / w.mean().clamp_min(1e-6))
        jepa_loss = err.mul(mask_w).sum() / max(1, b * 2)
        kde = dino_cfg["kde_loss_weight"] * k_scale * sum(kde_loss(x, dino_cfg["kde_concentration"]) for x in sg["cls"].chunk(train_cfg["global_views"]))
        # FINO metadata guidance on the CLS token (train-only; meta=None in eval), orthogonal to the JEPA patch
        # objective. lambda_meta=0.03/branch; GradScale gates the encoder gradient by the DANN ramp gamma with the
        # per-factor sign (+ M+ encourage / - M- suppress). fp32 island (1/tau=0.023 too sharp for bf16); missing
        # factors masked. Discrete: L2-normed student CLS vs EMA prototype bank (clone-rebind keeps the backward-saved
        # bank valid). Continuous: an MLP regresses the z-scored value.
        meta_loss = sg["cls"].new_zeros(())
        if meta is not None:
            gamma, md, mc = meta  # md (B,n_disc) int64 (-1 missing); mc {factor: (B,dim) float, nan missing}
            phi_s = F.normalize(sg["cls"].float(), dim=-1)
            phi_t = F.normalize(t["cls"].float(), dim=-1)
            terms = []  # (factor, per-branch loss 0.03*L_t); combined below, optionally gradient-equalized
            with torch.autocast(device_type="cuda", enabled=False):
                for j, (f, sign) in enumerate(fino_disc):
                    lab = md[:, j].repeat(train_cfg["global_views"]); ok = lab >= 0  # repeat, NOT interleave
                    if ok.any():
                        logits = (GradScale.apply(phi_s[ok], sign * gamma) @ protos[f].t()) / 0.023
                        terms.append((f, 0.03 * F.cross_entropy(logits, lab[ok])))
                        with torch.no_grad():
                            pt, lt = phi_t[ok], lab[ok]
                            upd = torch.zeros_like(protos[f]).index_add_(0, lt, pt)
                            cnt = torch.zeros(protos[f].shape[0], 1, device=device).index_add_(0, lt, torch.ones_like(pt[:, :1]))
                            seen = cnt.squeeze(1) > 0; new = protos[f].clone()
                            new[seen] = F.normalize(0.99 * new[seen] + 0.01 * (upd[seen] / cnt[seen]), dim=-1); protos[f] = new
                # FINO Eq.3 regresses continuous factors from the RAW backbone CLS; phi_s is L2-normalized (needed only
                # for the cosine discrete branch and it strips the radial magnitude). raw_cls=True feeds the raw CLS.
                cls_cont = sg["cls"].float() if fino_cfg.get("raw_cls") else phi_s
                for f, sign in fino_cont:
                    val = mc[f].repeat(train_cfg["global_views"], 1); ok = ~torch.isnan(val).any(dim=1)
                    if ok.any():
                        cpred = predictors[f](GradScale.apply(cls_cont[ok], sign * gamma))
                        terms.append((f, 0.03 * F.mse_loss(cpred, val[ok])))
                # FINO Alg A.3 per-branch gradient equalisation: rescale each branch by n_bar/EMA(||dL_t/dCLS||) so the
                # discrete-CE and continuous-MSE gradients reach the encoder at matched magnitudes (detached -> reweight
                # only; geometric-mean target; no-op for <2 branches). grad_eq_ema = per-factor EMA bank (mu=0.99).
                if fino_cfg.get("grad_equalize") and len(terms) > 1:
                    g = {f: torch.autograd.grad(L, sg["cls"], retain_graph=True)[0].norm() for f, L in terms}
                    for f in g: grad_eq_ema[f] = 0.99 * grad_eq_ema[f] + 0.01 * g[f].detach().float()
                    nbar = torch.exp(torch.stack([grad_eq_ema[f].log() for f, _ in terms]).mean())
                    meta_loss = sum((nbar / grad_eq_ema[f]).detach() * L for f, L in terms)
                else:
                    for _, L in terms: meta_loss = meta_loss + L
        # Prognostic sidecar: one prediction per global crop, averaged to one per tile. Masking is
        # per ELEMENT, not per row -- coverage differs by factor (e.g. til 3564 patients vs fga 8835).
        gram_loss = sg["cls"].new_zeros(())
        if gram_teacher is not None:
            # one global view per image keeps this to a single extra forward rather than two
            ps = F.normalize(sg["tapped"][0][:b].float(), dim=-1)
            with torch.no_grad():
                pt = F.normalize(gram_teacher(gf[:b], taps=(gram_layer,))["tapped"][0].float(), dim=-1)
            gram_loss = gram_weight * F.mse_loss(ps @ ps.transpose(1, 2), pt @ pt.transpose(1, 2))
        rq_loss = sg["cls"].new_zeros(())
        if rq_head is not None:
            q = student_backbone.patch_size
            px = gf.unfold(2, q, q).unfold(3, q, q).permute(0, 2, 3, 1, 4, 5).flatten(3).flatten(0, 2)[mask_idx].float()
            with torch.no_grad():
                # Standardise each patch first: without it the projection is dominated by patch brightness and
                # the codebook collapses (measured on real tiles: 647/8192 codes, 2.66 nats, top code 37%;
                # standardised: 2853 codes, 6.89 nats, top code 2.4%).
                px = (px - px.mean(-1, keepdim=True)) / px.std(-1, keepdim=True).clamp_min(1e-6)
                labels = (F.normalize(px @ rq_proj, dim=-1) @ rq_code.t()).argmax(-1)
            rq_loss = rq_weight * F.cross_entropy(rq_head(pred.float()), labels)
        capi_loss = sg["cls"].new_zeros(())
        if capi_head is not None:
            # SwAV-style weight tying: the SAME head projects both sides, so gradient only ever
            # reaches the prototypes through the student branch below -- the no_grad target branch
            # can't collapse them toward a trivial constant assignment.
            with torch.no_grad():
                capi_q = sinkhorn(capi_head(target.float()), 0.1)
            capi_loss = capi_weight * -(capi_q * F.log_softmax(capi_head(pred.float()) / 0.1, dim=-1)).sum(-1).mean()
        prog_loss = sg["cls"].new_zeros(())
        if prog_head is not None and prog_target is not None:
            pred_prog = prog_head(sg["patches"]).view(train_cfg["global_views"], b, -1).mean(0)
            ok = ~torch.isnan(prog_target)
            if ok.any():
                prog_loss = prog_weight * F.smooth_l1_loss(pred_prog[ok], prog_target[ok])
        xsite_loss = sg["cls"].new_zeros(())
        if xsite_weight and site is not None:
            with torch.no_grad():
                anchor_t = F.normalize(t["cls"][:b].float(), dim=-1)
                valid = xsite_queue_site >= 0
                sim = anchor_t @ xsite_queue_feat.t()  # the one matmul per step
                other_site = site.unsqueeze(1) != xsite_queue_site.unsqueeze(0)
                sim = sim.masked_fill(~(valid.unsqueeze(0) & other_site), -2.0)
                has_nn = (sim > -1.5).any(dim=1)
                nn_target = xsite_queue_feat[sim.argmax(dim=1)]
            anchor_s = F.normalize(sg["cls"][:b].float(), dim=-1)
            per_sample = 1.0 - (anchor_s * nn_target).sum(-1)
            xsite_loss = xsite_weight * (per_sample * has_nn).sum() / has_nn.sum().clamp_min(1)
            with torch.no_grad():
                n = min(b, xsite_queue_size)
                idx = (xsite_ptr[0] + torch.arange(n, device=anchor_t.device)) % xsite_queue_size
                xsite_queue_feat[idx] = anchor_t[:n]
                xsite_queue_site[idx] = site[:n]
                xsite_ptr[0] = (xsite_ptr[0] + n) % xsite_queue_size
        stain_loss = sg["cls"].new_zeros(())
        if stain_weight and stain_x is not None:
            sg_stain = student_backbone(stain_x, checkpoint=ckpt)
            student_p = F.normalize(sg_stain["patches"].float(), dim=-1)
            teacher_p = F.normalize(t["patches"][:b].float(), dim=-1)
            stain_loss = stain_weight * (1.0 - (student_p * teacher_p).sum(-1)).mean()
        return local_loss + global_loss, jepa_loss, kde, meta_loss, prog_loss, rq_loss + gram_loss + siam_kl + xsite_loss + stain_loss + capi_loss

    # Held-out validation pass: same DINO + JEPA + KDE losses on `val_batches` of the val split.
    # Schedule terms (teacher_temp, kde_scale) drift over training, so read val curves as same-step
    # diagnostics. RNG is snapshotted/restored so val masks don't perturb the next training step.
    def evaluate(eval_step, eval_teacher_temp, eval_kde_scale):
        for m in (student_backbone, student_dino_head, student_predictor):
            m.eval()
        py_rng, cpu_rng, cuda_rng = random.getstate(), torch.random.get_rng_state(), torch.cuda.get_rng_state(device)
        random.seed(train_cfg["seed"] + eval_step)
        torch.manual_seed(train_cfg["seed"] + eval_step)
        sums = torch.zeros(4, device=device)
        n_batches = 0
        for vb_idx, vbatch in enumerate(val_loader):
            if vb_idx >= int(train_cfg["val_batches"]):
                break
            vg, vl = vbatch["global_views"].to(device, non_blocking=True), vbatch["local_views"].to(device, non_blocking=True)
            b = vg.shape[0]
            with torch.no_grad(), autocast:
                gf, lf = vg.transpose(0, 1).flatten(0, 1), vl.transpose(0, 1).flatten(0, 1)
                masks, mask_idx, mask_w = block_mask_fn(b * train_cfg["global_views"], global_grid, device, n_blocks=int(dino_cfg["jepa_blocks"]), block_scale=float(dino_cfg["jepa_block_scale"]))
                m1, m2 = split_context(masks) if siam else (masks, None)
                dino_l, jepa_l, kde_v, _, _, _ = compute_losses(gf, lf, b, m1, mask_idx, mask_w, eval_teacher_temp, eval_kde_scale, masks2=m2)
            sums += torch.tensor([float(dino_l), float(jepa_l), float(kde_v), float(dino_l + jepa_l + kde_v)], device=device)
            n_batches += 1
        random.setstate(py_rng)
        torch.random.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
        return dict(zip(("dino", "jepa", "kde", "total"), (sums / max(1, n_batches)).tolist()))

    # Ingest completed probe result JSONs into metrics.jsonl and wandb.
    def log_probe_results():
        if probe_state is not None:
            collect_probe_results(probe_state, wandb_run, metrics_path)

    # Queue a probe at `checkpoint_step` for the given sample target; no-op if already done.
    def run_probe_at(checkpoint_step, target_samples):
        if probe_state is None or (probe_state["paths"]["results_dir"] / f"step_{checkpoint_step:07d}.json").exists():
            log_probe_results()
            return
        queue_probe_job(probe_state, checkpoint_payload(checkpoint_step, full=False), checkpoint_step, train_flops, min(1.0, target_samples / max_train_samples))
        log_probe_results()

    # Queue the furthest crossed sample milestone so delayed probes do not run on stale checkpoints.
    def maybe_run_probe(checkpoint_step):
        nonlocal next_probe_idx
        if probe_state is None or next_probe_idx >= len(probe_targets) or examples_seen < probe_targets[next_probe_idx]:
            return
        while next_probe_idx + 1 < len(probe_targets) and examples_seen >= probe_targets[next_probe_idx + 1]:
            next_probe_idx += 1
        run_probe_at(checkpoint_step, probe_targets[next_probe_idx])
        next_probe_idx += 1

    log_probe_results()
    max_train_flops = int(train_cfg["max_train_flops"])
    warmup_train_samples = math.ceil(max_train_samples * dino_cfg["warmup_fraction"])
    # Probe targets are sample milestones: one tile counts once even with many global/local crops.
    probe_count = int(cfg["probe"]["count"]) if probe_enabled(cfg) else 0
    probe_targets = [math.ceil(max_train_samples * (i + 1) / probe_count) for i in range(probe_count)]
    if len(set(probe_targets)) != len(probe_targets):
        raise ValueError(f"probe.count={probe_count} is too large for max_train_samples={max_train_samples}")
    next_probe_idx = 0
    if probe_state is not None:
        completed = [round(float(json.loads(p.read_text()).get("target_fraction", -1)) * max_train_samples) for p in probe_state["paths"]["results_dir"].glob("step_*.json")]
        if completed:
            next_probe_idx = sum(target <= max(completed) for target in probe_targets)
    train_loop_started_at = time.monotonic()
    last_saved_step = step
    last_console_step = step
    last_console_monotonic = time.monotonic()
    data_wait_started_at = time.monotonic()
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if train_cfg["bf16"] else contextlib.nullcontext()
    # Per-step FLOPs are measured once via FlopCounterMode on the first wrapped step (forward +
    # backward + opt.step) and reused for every subsequent step since the shapes don't change.
    # Counts the EMA teacher forward + DINO/JEPA heads, not just the backbone, so the
    # 1e18 leaderboard cap reflects real GPU work.
    measured_flops_per_step = None

    while examples_seen + batch_size <= train_sample_budget and train_flops < max_train_flops:
        for batch in train_loader:
            if examples_seen + batch_size > train_sample_budget or train_flops >= max_train_flops:
                break
            batch_started_at = time.monotonic()
            data_seconds = batch_started_at - data_wait_started_at
            student_backbone.train()
            student_dino_head.train()
            student_predictor.train()
            completed_step = step + 1
            should_log = completed_step == 1 or completed_step % train_cfg["log_every"] == 0
            # Data identifiers stay on CPU and feed coverage metrics; image tensors move below.
            for key, batch_key in (("sample", "sample_idx"), ("slide", "slide_id"), ("patient", "patient_id")):
                pending_ids[key].update(int(x) for x in batch[batch_key].tolist())
            global_views, local_views = [batch[key].to(device, non_blocking=True) for key in ("global_views", "local_views")]
            visible_now = batch_size * (train_cfg["global_views"] * global_patches + train_cfg["local_views"] * local_patches)
            # LR warmup uses the 1M-tile sample cap; decay/WD/teacher/freeze/KDE default to the public FLOP budget.
            # But this run hits the sample cap at ~19% of the FLOP budget, so a FLOP-keyed cosine only traverses ~0.11
            # of its arc (LR never anneals, KDE peaks at 0.22, WD ~0.05). lr_key/reg_key="sample" re-key the decay/reg
            # schedules to SAMPLE progress so they complete over the actual 1M-tile run (same fix as the FINO gamma ramp).
            frac = min(1.0, train_flops / max_train_flops)
            sfrac = min(1.0, examples_seen / max_train_samples)
            lr_frac = sfrac if dino_cfg.get("lr_key") == "sample" else frac
            reg_frac = sfrac if dino_cfg.get("reg_key") == "sample" else frac
            warmup = min(1.0, examples_seen / max(1, warmup_train_samples))
            if warmup < 1.0:
                lr = dino_cfg["lr"] * warmup
            else:
                lr = cosine_schedule(dino_cfg["lr"], dino_cfg["lr_min"], (lr_frac - dino_cfg["warmup_fraction"]) / max(1e-9, 1 - dino_cfg["warmup_fraction"]))
            wd = cosine_schedule(0.04, 0.2, reg_frac)
            teacher_temp = 0.04 + min(1.0, reg_frac / 0.2727) * (0.07 - 0.04)
            last_layer_lr = 0.0 if frac < dino_cfg["freeze_last_layer_fraction"] else lr
            for group in opt.param_groups:
                base_lr = last_layer_lr if group["last_layer"] else lr
                group["lr"] = base_lr * group["lr_mult"]
                group["weight_decay"] = wd * group["wd_mult"]
            masks, mask_idx, mask_w = block_mask_fn(batch_size * train_cfg["global_views"], global_grid, device, n_blocks=int(dino_cfg["jepa_blocks"]), block_scale=float(dino_cfg["jepa_block_scale"]))
            if gram_weight and gram_teacher is None and sfrac >= gram_start:
                gram_teacher = deepcopy(teacher_backbone).eval().requires_grad_(False)
                # the anchor adds a forward pass, so the step cost measured at step 1 no longer holds;
                # clearing it re-runs FlopCounterMode next step and keeps train_flops honest
                measured_flops_per_step = None
                print(f"{console_prefix()} gram anchor: froze teacher at sample_frac {sfrac:.3f}", flush=True)
            siam_masks2 = None
            if siam:
                masks, siam_masks2 = split_context(masks)
            kde_scale = min(1.0, max(0.0, (reg_frac - 0.1) / 0.4))
            # Wrap forward + backward + opt.step in FlopCounterMode on the first step only;
            # subsequent steps reuse measured_flops_per_step (fixed shapes => fixed cost).
            flop_ctx = FlopCounterMode(display=False) if measured_flops_per_step is None else contextlib.nullcontext()
            with flop_ctx:
                with autocast:
                    # Crop-major flatten: collate shape is (B, V, 3, H, W) but DINO wants per-crop chunks
                    # so [crop0_img0, crop0_img1, ..., crop1_img0, ...] for clean teacher/student alignment.
                    gf = global_views.transpose(0, 1).flatten(0, 1)
                    lf = local_views.transpose(0, 1).flatten(0, 1)
                    # FINO DANN ramp keyed to nanopath's SAMPLE budget (NOT FLOPs — sample-capped at ~19% of the FLOP
                    # cap, so a flop-keyed ramp stalls gamma at ~0.75*gamma_max). Counted from the backbone-unfreeze
                    # point: gamma=0 through the frozen Phase 1 (banks warm), then ramps to full gamma_max by the cap.
                    ramp = max(0.0, (examples_seen / max_train_samples - freeze_backbone_frac) / max(1e-6, 1.0 - freeze_backbone_frac))
                    meta = ((fino_cfg["gamma_max"] * (2.0 / (1.0 + math.exp(-10.0 * ramp)) - 1.0),
                             batch["meta_disc"].to(device, non_blocking=True),
                             {f: batch["mc_" + f].to(device, non_blocking=True) for f, _ in fino_cont}) if fino_cfg else None)
                    cond = batch["meta_disc"][:, cond_col].repeat(train_cfg["global_views"]).to(device, non_blocking=True) if jepa_cond else None
                    dino_loss_value, jepa_loss, kde, meta_loss, prog_loss, rq_loss = compute_losses(
                        gf, lf, batch_size, masks, mask_idx, mask_w, teacher_temp, kde_scale,
                        ckpt=activation_checkpointing, meta=meta, cond=cond,
                        prog_target=batch["prog_target"].to(device, non_blocking=True) if prog_head is not None else None,
                        masks2=siam_masks2,
                        site=batch["site_id"].to(device, non_blocking=True) if (xsite_weight or site_center) else None,
                        stain_x=batch["stain_view"].to(device, non_blocking=True) if stain_weight else None,
                    )
                    total_loss = dino_loss_value + jepa_loss + kde + meta_loss + prog_loss + rq_loss
                opt.zero_grad(set_to_none=True)
                total_loss.backward()
                if examples_seen / max_train_samples < freeze_backbone_frac:  # Phase 1: backbone frozen (patch_embed + heads + metadata still train)
                    for n, p in student_backbone.named_parameters():
                        if not n.startswith("patch_embed"): p.grad = None
                grad_norm = nn.utils.clip_grad_norm_(
                    [*student_backbone.parameters(), *student_dino_head.parameters(), *student_predictor.parameters()],
                    dino_cfg["clip_grad"],
                )
                opt.step()
            if measured_flops_per_step is None:
                measured_flops_per_step = int(flop_ctx.get_total_flops())
                print(f"{console_prefix()} measured_flops_per_step: {measured_flops_per_step:,}", flush=True)
            step_train_flops = measured_flops_per_step
            with torch.no_grad():
                m = cosine_schedule(0.994, 1.0, reg_frac)
                update_ema(student_backbone, teacher_backbone, m)
                update_ema(student_dino_head, teacher_dino_head, m)
            step_seconds = time.monotonic() - batch_started_at
            examples_seen += batch_size
            visible_patch_presentations += visible_now
            train_flops += step_train_flops
            if should_log:
                reduced = {
                    "dino": float(dino_loss_value.detach()),
                    "jepa": float(jepa_loss.detach()),
                    "kde": float(kde.detach()),
                    "total": float(total_loss.detach()),
                }
                unique_counts = flush_unique_counts()
                now = time.time()
                elapsed = max(1e-6, now - last_time)
                items_per_sec = (examples_seen - last_examples) / elapsed
                visible_patches_per_sec = (visible_patch_presentations - last_visible_patch_presentations) / elapsed
                flops_per_sec = (train_flops - last_train_flops) / elapsed
                train_loop_wall_seconds = time.monotonic() - train_loop_started_at
                last_time = now
                last_examples = examples_seen
                last_visible_patch_presentations = visible_patch_presentations
                last_train_flops = train_flops
                gpu_mem_gb = torch.cuda.memory_allocated(device) / (1024**3)
                gpu_peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                console_now = time.monotonic()
                console_gap_ms = 1000.0 * (console_now - last_console_monotonic)
                steps_since_console = max(1, completed_step - last_console_step)
                flop_steps_remaining = math.ceil(max(0, max_train_flops - train_flops) / max(1, step_train_flops))
                sample_steps_remaining = max(0, train_sample_budget - examples_seen) // batch_size
                steps_remaining = min(flop_steps_remaining, sample_steps_remaining)
                total_steps_estimate = completed_step + steps_remaining
                eta_seconds = int(max(0.0, steps_remaining * console_gap_ms / 1000.0 / steps_since_console))
                eta_string = f"{eta_seconds // 3600}:{(eta_seconds % 3600) // 60:02d}:{eta_seconds % 60:02d}"
                current_lr = opt.param_groups[0]["lr"]
                train_log = {
                    "step": completed_step,
                    **reduced,
                    "items_per_sec": items_per_sec,
                    "visible_patches_per_sec": visible_patches_per_sec,
                    "flops_per_sec": flops_per_sec,
                    "wall_seconds": train_loop_wall_seconds,
                    "step_seconds": step_seconds,
                    "data_seconds": data_seconds,
                    "console_gap_ms": console_gap_ms,
                    "eta_seconds": eta_seconds,
                    "flop_fraction": min(1.0, float(train_flops) / float(max_train_flops)),
                    "sample_fraction": min(1.0, float(examples_seen) / float(max_train_samples)),
                    "lr": current_lr,
                    "wd": wd,
                    "teacher_temp": teacher_temp,
                    "teacher_momentum": m,
                    "kde_scale": kde_scale,
                    "batch_size": batch_size,
                    "examples_seen": examples_seen,
                    "visible_patch_presentations": visible_patch_presentations,
                    "train_flops": train_flops,
                    "gpu_mem_gb": gpu_mem_gb,
                    "gpu_peak_mem_gb": gpu_peak_mem_gb,
                    "grad_norm": float(grad_norm.detach()),
                }
                train_log.update(unique_counts)
                print(
                    f"{console_prefix()} Training  "
                    f"[{completed_step}/{total_steps_estimate}]  eta: {eta_string}  gap: {console_gap_ms:.2f} ms  "
                    f"lr: {current_lr:.6f}  total: {reduced['total']:.4f}  "
                    f"dino: {reduced['dino']:.4f}  jepa: {reduced['jepa']:.4f}  kde: {reduced['kde']:.4f}  "
                    f"grad_norm: {train_log['grad_norm']:.4f}  flops/s: {flops_per_sec:.3e}  "
                    f"time: {step_seconds:.6f}  data: {data_seconds:.6f}  "
                    f"max mem: {int(gpu_peak_mem_gb * 1024)}",
                    flush=True,
                )
                last_console_step = completed_step
                last_console_monotonic = console_now
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(train_log) + "\n")
                wandb_run.log(
                    {f"train/{key}": value for key, value in train_log.items() if key != "step"},
                    step=completed_step,
                )
                log_probe_results()
                torch.cuda.reset_peak_memory_stats(device)
            if save_checkpoints and completed_step % save_every == 0:
                # Atomic rename keeps the previous good latest.pt intact if a
                # kill lands mid-save.
                save_latest_checkpoint(completed_step)
            # Probe at intermediate sample milestones (probe.count > 1); the final probe
            # always runs after the loop exits, regardless of milestones.
            maybe_run_probe(completed_step)
            if completed_step % int(train_cfg["eval_every"]) == 0 or train_flops >= max_train_flops or examples_seen + batch_size > train_sample_budget:
                val = evaluate(completed_step, teacher_temp, kde_scale)
                val_log = {"step": completed_step, **{f"val_{k}": v for k, v in val.items()}}
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(val_log) + "\n")
                wandb_run.log({f"val/{k}": v for k, v in val.items()}, step=completed_step)
                print(f"{console_prefix()} Validation  [{completed_step}]  total: {val['total']:.4f}  dino: {val['dino']:.4f}  jepa: {val['jepa']:.4f}  kde: {val['kde']:.4f}", flush=True)
                # Reset rate clocks after validation so the next train log is train-rate only.
                last_console_step, last_console_monotonic = completed_step, time.monotonic()
                last_time, last_examples, last_visible_patch_presentations, last_train_flops = time.time(), examples_seen, visible_patch_presentations, train_flops
            step = completed_step
            data_wait_started_at = time.monotonic()
            if train_flops >= max_train_flops or examples_seen + batch_size > train_sample_budget:
                break
    train_loop_wall_seconds = time.monotonic() - train_loop_started_at
    stop_reason = "max_train_flops" if train_flops >= max_train_flops else "max_train_samples"
    final_unique_counts = flush_unique_counts()
    if step > 0:
        # Final probes have their own readers; close pretraining workers before they compete for CPU/IO.
        if train_cfg["num_workers"] > 0:
            if train_loader._iterator is not None:
                train_loader._iterator._shutdown_workers()
                train_loader._iterator = None
        # Probes get their own short-lived checkpoint via run_probe_at; only persist latest.pt
        # at end-of-run when periodic saving is on (save_every set) so smoke runs leave nothing.
        if robust_norm_tiles:
            # Fit scanner-response directions after optimization so training is unchanged.
            started = time.monotonic()
            data_dir = Path(cfg["data"]["dataset_dir"])
            jpegs = [jpeg for shard in range(128) for jpeg in pq.ParquetFile(data_dir / f"shard-{shard:05d}.parquet").read_row_group(0, columns=["jpeg"])["jpeg"].to_pylist()[:48]]
            assert len(jpegs) == robust_norm_tiles
            resize = transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.ToTensor()])
            mean = torch.tensor(cfg["data"]["mean"], device=device).view(1, 3, 1, 1)
            std = torch.tensor(cfg["data"]["std"], device=device).view(1, 3, 1, 1)
            generator = torch.Generator().manual_seed(555)
            gamma = torch.empty(robust_norm_tiles, 3, 1, 1).uniform_(0.8, 1.25, generator=generator)
            gain = torch.empty_like(gamma).uniform_(0.85, 1.18, generator=generator)
            huesat = torch.empty(robust_norm_tiles, 2).uniform_(0, 1, generator=generator)

            @torch.no_grad()
            def robust_features(images):
                with autocast:
                    x, taps = teacher_backbone._prepare_tokens((images.to(device) - mean) / std), []
                    for i, block in enumerate(teacher_backbone.blocks):
                        x = block(x)
                        if i in (2, 4, 6, 8, 11):
                            taps.append(teacher_backbone.norm(x)[:, 0])
                    x = teacher_backbone.norm(x)
                return torch.stack([x[:, 0], x[:, 1 + teacher_backbone.registers :].mean(1), *taps], 1).float().cpu()

            bases, deltas = [], []
            for start in range(0, robust_norm_tiles, batch_size):
                base = torch.stack([resize(Image.open(io.BytesIO(jpeg)).convert("RGB")) for jpeg in jpegs[start : start + batch_size]])
                base_features = robust_features(base)
                views = (
                    base.clamp_min(1e-6) ** gamma[start : start + batch_size],
                    (base * gain[start : start + batch_size]).clamp(0, 1),
                    torch.stack([TF.adjust_saturation(TF.adjust_hue(tile, float((hs[0] - 0.5) * 0.1)), float(0.7 + hs[1] * 0.7)) for tile, hs in zip(base, huesat[start : start + batch_size])]),
                )
                bases.append(base_features)
                deltas.extend(robust_features(view) - base_features for view in views)
            base_features, delta_features = torch.cat(bases), torch.cat(deltas)
            centered = delta_features - delta_features.mean(0)
            directions = torch.linalg.svd(centered.movedim(1, 0).to(device), full_matrices=False)[2]
            for model in (student_backbone, teacher_backbone):
                model.rn_mu.copy_(base_features.mean(0)[:2]); model.rn_v.copy_(directions[:2, :32])
                model.pf_mu.copy_(base_features.mean(0)[2:]); model.pf_v.copy_(directions[2:, :1])
                model.rn_fitted.fill_(True); model.pf_fitted.fill_(True)
            print(f"{console_prefix()} RobustNorm  [{step}]  fitted rank 32 + per-tap rank 1 from {len(jpegs)} tiles in {time.monotonic() - started:.0f}s", flush=True)
        if save_checkpoints and step != last_saved_step:
            save_latest_checkpoint(step)
        run_probe_at(step, examples_seen)
    log_probe_results()
    # Summary is the small, stable artifact downstream scripts and humans compare across runs.
    summary = {
        "project": cfg["project"]["name"],
        "family": cfg["project"]["family"],
        "recipe_id": cfg["project"]["recipe_id"],
        "config_path": cfg["config_path"],
        "wandb": wandb_meta,
        "slurm_job_id": slurm_job_id,
        "backbone_activated_params": backbone_activated_params,
        "batch_size": batch_size,
        "max_train_samples": max_train_samples,
        "max_train_flops": max_train_flops,
        "train_loop_wall_seconds": train_loop_wall_seconds,
        "stop_reason": stop_reason,
        "steps_completed": step,
        "tile_presentations": examples_seen,
        "visible_patch_presentations": visible_patch_presentations,
        **final_unique_counts,
        "train_flops": train_flops,
        "flop_fraction": min(1.0, float(train_flops) / float(max_train_flops)),
        "sample_fraction": min(1.0, float(examples_seen) / float(max_train_samples)),
        # Average throughput over the train loop; wall time is diagnostic, not an eligibility cap.
        "flops_per_sec": train_flops / max(1.0, train_loop_wall_seconds),
        "visible_patches_per_sec": visible_patch_presentations / max(1.0, train_loop_wall_seconds),
        "warmup_fraction": dino_cfg["warmup_fraction"],
        "warmup_train_samples": warmup_train_samples,
        "lr": dino_cfg["lr"],
        "adam_beta2": dino_cfg["adam_beta2"],
        "kde_loss_weight": dino_cfg["kde_loss_weight"],
        "kde_concentration": dino_cfg["kde_concentration"],
        "drop_path_rate": dino_cfg["drop_path_rate"],
        "layerwise_decay": dino_cfg["layerwise_decay"],
        "probe_target_samples": probe_targets,
        "probe_target_fractions": [None if max_train_samples == 0 else target / max_train_samples for target in probe_targets],
        **({} if probe_state is None else completed_probe_summary(output_dir)),
    }
    if probe_state is not None and "final_score" not in summary:
        raise ValueError("probe.enabled is true but final_score is missing; check probe.count, probe failures, and final checkpoint scheduling")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"{console_prefix()} Summary  "
        f"steps: {step}  train_wall: {train_loop_wall_seconds:.2f}s  "
        f"final_score: {summary.get('final_score')}",
        flush=True,
    )
    for key in summary.keys():
        wandb_run.summary[key] = summary[key]
    wandb_run.finish()
    finish_labless_autosubmit(labless_autosubmit_file, output_dir, repo_dir)


if __name__ == "__main__":
    main()
