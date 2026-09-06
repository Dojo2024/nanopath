# Run the full frozen-probe suite on deterministic random DINOv2-small/14-reg weights.
# Useful as a probe sanity floor: same architecture as DINOv2-small, no checkpoint.

from dinov2_small_baseline import run_dinov2_baseline


if __name__ == "__main__":
    run_dinov2_baseline(
        "dinov2_random_baseline.py",
        "baseline-dinov2-random",
        "dinov2-vits14-reg-random-init-seed0",
        "dinov2_vits14_reg",
        "/data/$USER/nanopath/baselines/dinov2-random",
        pretrained=False,
    )
