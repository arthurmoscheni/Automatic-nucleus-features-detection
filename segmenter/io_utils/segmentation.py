from __future__ import annotations
import os
import numpy as np
from skimage import io as skio, util
import tifffile as tiff

def save_segmentation_files(results: dict, base_name: str, output_dir: str) -> None:
    """Save masks as a multi-page TIF and a composite PNG visualization."""
    os.makedirs(output_dir, exist_ok=True)
    masks = results["masks"]

    # save multi-page / multi-channel mask stack
    stack = []
    order = ["blue", "green", "red"]
    for name in order:
        if name in masks:
            stack.append(masks[name].astype(np.uint16))
    if stack:
        tiff.imwrite(os.path.join(output_dir, f"{base_name}_masks.tif"), np.stack(stack, axis=0))

    # simple composite visualization: max of labeled masks, mapped to uint8
    if stack:
        comp = np.maximum.reduce([m > 0 for m in stack]).astype(np.uint8) * 255
        skio.imsave(os.path.join(output_dir, f"{base_name}_masks_vis.png"), comp)

def save_preprocessed_images(results: dict, base_name: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for channel, img in results["preprocessed_channels"].items():
        img16 = util.img_as_uint(img)
        path = os.path.join(output_dir, f"{channel}_preprocessed.tif")
        skio.imsave(path, img16)
        print(f"{channel.capitalize()} preprocessed image saved to: {path}")

def save_original_image(results: dict, base_name: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for channel, img in results["original_image"].items():
        img16 = util.img_as_uint(img)
        path = os.path.join(output_dir, f"{channel}_original.tif")
        skio.imsave(path, img16)
        print(f"{channel.capitalize()} original image saved to: {path}")
