import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage import io, img_as_ubyte
from matplotlib import cm
import tkinter as tk
from tkinter import filedialog, simpledialog
import os
import numpy as np


def choose_input_file():
    """File selection dialog for input image."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    path = filedialog.askopenfilename(
        title="Select input image",
        filetypes=[("Image files", "*.tif *.tiff *.png *.jpg *.jpeg")]
    )
    root.destroy()
    
    if not path:
        raise ValueError("No input file selected.")
    return path.replace('\\', '/')


def choose_output_paths(input_path):
    """Choose output directory and filename."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    output_dir = filedialog.askdirectory(title="Select Output Folder")
    if not output_dir:
        root.destroy()
        raise ValueError("No output folder selected.")

    default_filename = os.path.basename(input_path)
    filename = simpledialog.askstring(
        "Save As", 
        "Enter filename (with or without .tif):", 
        initialvalue=default_filename
    )
    root.destroy()
    
    if not filename:
        raise ValueError("No file name entered.")

    if not filename.lower().endswith('.tif'):
        filename += '.tif'
        
    output_path = os.path.join(output_dir, filename).replace('\\', '/')
    vis_path = output_path.replace('.tif', '_masks_vis.png')
    return output_path, vis_path


def _save_masks(results, output_path, vis_path):
    """Save mask results to files."""
    masks = results['masks']
    base_path = output_path.replace('.tif', '')
    
    # Save all channel masks
    for channel in ['blue', 'green', 'red']:
        if channel in masks:
            channel_path = f"{base_path}_{channel}.tif"
            io.imsave(channel_path, masks[channel].astype(np.uint16))
            print(f"{channel.capitalize()} masks saved to: {channel_path}")
    
    # Create visualization for green channel
    if 'green' in masks:
        green_masks = masks['green']
        colored_mask = cm.jet(green_masks / green_masks.max())[:, :, :3]
        io.imsave(vis_path, img_as_ubyte(colored_mask))
        print(f"Visualization saved to: {vis_path}")


def _save_preprocessed(results, output_path):
    """Save preprocessed images."""
    preprocessed_channels = results['preprocessed_channels']
    base_path = output_path.replace('.tif', '')
    
    for channel, img in preprocessed_channels.items():
        channel_path = f"{base_path}_{channel}_preprocessed.tif"
        io.imsave(channel_path, img.astype(np.uint16))
        print(f"Preprocessed {channel} image saved to: {channel_path}")


def save_results(results, output_path, vis_path, path='masks'):
    """Save segmentation results to files."""
    if path == 'masks':
        _save_masks(results, output_path, vis_path)
    elif path == 'preprocessed':
        _save_preprocessed(results, output_path)


def _plot_channels(channels, title):
    """Helper function to plot image channels."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    channel_names = ['blue', 'green', 'red']
    
    for i, name in enumerate(channel_names):
        if name in channels:
            axes[i].imshow(channels[name], cmap='gray')
            axes[i].set_title(f'{title} {name.capitalize()} Channel')
            axes[i].axis('off')
    
    plt.suptitle(f'{title} Channels')
    plt.tight_layout()
    plt.show()


def _plot_masks(masks):
    """Helper function to plot segmentation masks."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    channel_names = ['blue', 'green', 'red']
    
    # Row 1: Original segmentation results
    for i, name in enumerate(channel_names):
        if name in masks:
            colored_labels = label2rgb(masks[name], bg_label=0)
            axes[0, i].imshow(colored_labels)
            num_masks = len(np.unique(masks[name])) - 1
            axes[0, i].set_title(f'{name.capitalize()} Masks ({num_masks} objects)')
            axes[0, i].axis('off')
    
    # Row 2: Filtered results
    filter_configs = [
        ('blue_filtered', 'Blue (Filtered)'),
        ('green', 'Green (Reference)'),
        ('red_filtered', 'Red (Filtered)')
    ]
    
    for i, (filter_name, display_name) in enumerate(filter_configs):
        if filter_name in masks:
            colored_labels = label2rgb(masks[filter_name], bg_label=0)
            axes[1, i].imshow(colored_labels)
            num_masks = len(np.unique(masks[filter_name])) - 1
            axes[1, i].set_title(f'{display_name} ({num_masks} objects)')
            axes[1, i].axis('off')
        else:
            axes[1, i].axis('off')
    
    plt.suptitle('Segmentation Results')
    plt.tight_layout()
    plt.show()


def display_results(results):
    """Display comprehensive results."""
    # Display original and preprocessed channels
    _plot_channels(results['original_image'], 'Original')
    _plot_channels(results['preprocessed_channels'], 'Preprocessed')
    
    # Display segmentation results
    _plot_masks(results['masks'])