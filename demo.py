import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from dataset import model_forward_unscale
from model import init_model
from util.config import get_config_from_path
from util.train import to_cuda


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("predictions/demo"))
    parser.add_argument("--config", type=Path, default=Path("config/val.py"))
    args = parser.parse_args()

    config = get_config_from_path(args.config.resolve().as_posix())
    checkpoint = args.checkpoint_path.expanduser().resolve()
    config["checkpoint_path"] = checkpoint.as_posix()
    config["resumed_from"] = checkpoint.as_posix()
    config["local_rank"] = 0
    config["rank"] = 0

    model = init_model(config)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    sample = {
        "image": torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float()[None]
        / 255.0
    }
    sample = to_cuda(sample)
    with torch.inference_mode():
        result = model_forward_unscale(model, {"image": sample["image"]})
    depth = result["depth"][0].sort(dim=0).values.cpu().numpy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    for i, layer in enumerate(depth):
        # plt.imsave(args.output_dir / f"depth_layer_{i + 1}.png", layer, cmap="turbo")
        if i < 3:
            ax = axes[(i + 1) // 2, (i + 1) % 2]
            ax.imshow(layer, cmap="turbo_r")
            ax.set_title(f"Depth Layer {i + 1}")
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.output_dir / "comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(
        f"Saved {len(depth)} depth layers and comparison image to {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
