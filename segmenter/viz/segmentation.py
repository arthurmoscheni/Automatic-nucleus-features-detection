from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

def display_results(results: dict) -> None:
    """Quick look: preprocessed channels + mask overlays."""
    ch = results["preprocessed_channels"]
    masks = results["masks"]

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    for i, name in enumerate(["blue", "green", "red"]):
        if name in ch:
            axs[0, i].imshow(ch[name], cmap="gray")
            axs[0, i].set_title(f"{name} (preprocessed)"); axs[0, i].axis("off")
        if name in masks:
            axs[1, i].imshow(ch.get(name, np.zeros_like(next(iter(ch.values())))), cmap="gray")
            axs[1, i].imshow((masks[name] > 0).astype(float), alpha=0.5)
            axs[1, i].set_title(f"{name} mask"); axs[1, i].axis("off")
    plt.tight_layout(); plt.show()
