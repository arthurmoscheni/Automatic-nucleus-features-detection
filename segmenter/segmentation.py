import os
import numpy as np
from skimage import io, util
from cellpose import models
from scipy.ndimage import gaussian_filter
from utils import save_results, display_results


class ImageSegmentationPipeline:
    """A clean pipeline for multi-channel image segmentation using Cellpose."""
    
    def __init__(self, model_type='nuclei'):
        """
        Initialize the segmentation pipeline.
        
        Args:
            model_type (str): Type of Cellpose model ('nuclei', 'cyto', etc.)
        """
        self.model_type = model_type
        self.model = models.CellposeModel(model_type=model_type)
        print("Using CPU")
    
    def preprocess_image(self, img, apply_gaussian=True, sigma=1.0, 
                        enhance_contrast=True, alpha=1, beta=0.1):
        """
        Preprocess a single channel image for segmentation.
        
        Args:
            img (np.ndarray): Input image
            apply_gaussian (bool): Whether to apply Gaussian filtering
            sigma (float): Standard deviation for Gaussian filter
            enhance_contrast (bool): Whether to enhance contrast
            alpha (float): Contrast enhancement multiplier
            beta (float): Brightness adjustment
            
        Returns:
            np.ndarray: Preprocessed image normalized to [0, 1]
        """
        # Convert to float32 and normalize to [0, 1]
        img_norm = img.astype(np.float32)
        if img_norm.max() > 0:
            img_norm /= img_norm.max()
        
        # Apply contrast enhancement
        if enhance_contrast:
            img_norm = np.clip(alpha * img_norm + beta, 0, 1)
        
        # Apply Gaussian filter
        if apply_gaussian and img_norm.max() > 1e-3:
            img_norm = gaussian_filter(img_norm, sigma=sigma)
        
        return img_norm
    
    def load_and_preprocess_multichannel(self, image_path, preprocess_params=None):
        """
        Load a multi-channel image and preprocess each channel.
        
        Args:
            image_path (str): Path to the input image
            preprocess_params (dict): Parameters for preprocessing
            
        Returns:
            tuple: (original_image, preprocessed_channels)
        """
        if preprocess_params is None:
            preprocess_params = {
                'apply_gaussian': True,
                'sigma': 1.0,
                'enhance_contrast': True,
                'alpha': 1.0,
                'beta': 0.1
            }
        
        # Load original image
        original_img = io.imread(image_path)
        print(f"Loaded image shape: {original_img.shape}, dtype: {original_img.dtype}")
        
        if original_img.ndim != 3 or original_img.shape[-1] < 3:
            raise ValueError("Image must have at least 3 channels (RGB)")
        
        # Extract and preprocess each channel
        channels = {}
        channel_names = ['blue', 'green', 'red']
        
        for i, name in enumerate(channel_names):
            channel_img = original_img[:, :, i]
            channels[name] = self.preprocess_image(channel_img, **preprocess_params)
            print(f"Preprocessed {name} channel: shape {channels[name].shape}, "
                  f"range [{channels[name].min():.3f}, {channels[name].max():.3f}]")
        
        return original_img, channels
    
    def segment_channel(self, image, diameter=None, flow_threshold=0.5, 
                       cellprob_threshold=0, progress=False):
        """
        Segment a single channel using Cellpose.
        
        Args:
            image (np.ndarray): Preprocessed image to segment
            diameter (float): Expected cell diameter (None for auto)
            flow_threshold (float): Flow error threshold
            cellprob_threshold (float): Cell probability threshold
            progress (bool): Show progress bar
            
        Returns:
            tuple: (masks, flows, styles) from Cellpose
        """
        masks, flows, styles = self.model.eval(
            image,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            progress=progress
        )
        
        num_objects = len(np.unique(masks)) - 1  # Subtract 1 for background
        print(f"Segmented {num_objects} objects")
        
        return masks, flows, styles
    
    def calculate_overlap(self, mask1, mask2, label1, label2):
        """Calculate overlap ratio between two mask regions."""
        region1 = (mask1 == label1)
        region2 = (mask2 == label2)
        intersection = np.sum(region1 & region2)
        union = np.sum(region1 | region2)
        return intersection / union if union > 0 else 0
    
    def filter_masks_by_overlap(self, target_masks, reference_masks, overlap_threshold=0.3):
        """Filter masks based on overlap with reference masks."""
        target_labels = np.unique(target_masks)[1:]  # Exclude background
        reference_labels = np.unique(reference_masks)[1:]
        
        filtered_masks = np.zeros_like(target_masks)
        label_mapping = {}
        new_label = 1
        
        for target_label in target_labels:
            best_overlap = 0
            for ref_label in reference_labels:
                overlap = self.calculate_overlap(
                    target_masks, reference_masks, target_label, ref_label
                )
                if overlap > best_overlap:
                    best_overlap = overlap
            
            if best_overlap > overlap_threshold:
                filtered_masks[target_masks == target_label] = new_label
                label_mapping[target_label] = new_label
                new_label += 1
        
        return filtered_masks, label_mapping
    
    def process_multichannel_image(self, image_path, segment_params=None, 
                                  filter_by_green=True, overlap_threshold=0.3):
        """Complete pipeline: load, preprocess, and segment a multi-channel image."""
        if segment_params is None:
            segment_params = {
                'diameter': None,
                'flow_threshold': 0.5,
                'cellprob_threshold': 0,
                'progress': True
            }
        
        # Load and preprocess
        original_img, channels = self.load_and_preprocess_multichannel(image_path)

        # Segment each channel
        masks = {}
        flows = {}
        styles = {}
        original_imgs = {
            'blue': original_img[:, :, 0],
            'green': original_img[:, :, 1],
            'red': original_img[:, :, 2]
        }
        
        for channel_name, channel_img in channels.items():
            print(f"\nSegmenting {channel_name} channel...")
            mask, flow, style = self.segment_channel(channel_img, **segment_params)
            masks[channel_name] = mask
            flows[channel_name] = flow
            styles[channel_name] = style
        
        # Filter red and blue masks by green channel overlap if requested
        if filter_by_green and 'green' in masks:
            print(f"\nFiltering masks by green channel overlap (threshold={overlap_threshold})...")
            
            if 'red' in masks:
                masks['red'], _ = self.filter_masks_by_overlap(
                    masks['red'], masks['green'], overlap_threshold
                )
                print(f"Red masks: {len(np.unique(masks['red']))-1} kept after filtering")
                
            if 'blue' in masks:
                masks['blue'], _ = self.filter_masks_by_overlap(
                    masks['blue'], masks['red'], overlap_threshold
                )
                print(f"Blue masks: {len(np.unique(masks['blue']))-1} kept after filtering")
        
        return {
            'original_image': original_imgs,
            'preprocessed_channels': channels,
            'masks': masks,
            'flows': flows,
            'styles': styles
        }


def run_batch_segmentation(image_paths, output_dir, visualization=False):
    """Run segmentation on multiple images and return structured results."""
    pipeline = ImageSegmentationPipeline(model_type='nuclei')
    batch_results = {}
    
    os.makedirs(output_dir, exist_ok=True)
    
    for image_path in image_paths:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\nProcessing {base_name}…")
        
        try:
            # Process image
            results = pipeline.process_multichannel_image(image_path)
            
            # Create folder structure for this image
            image_output_dir = os.path.join(output_dir, base_name)
            segmentation_dir = os.path.join(image_output_dir, "segmentation")
            original_dir = os.path.join(image_output_dir, "original")
            preprocessed_dir = os.path.join(image_output_dir, "preprocessed")

            # Create directories
            for directory in [segmentation_dir, original_dir, preprocessed_dir]:
                os.makedirs(directory, exist_ok=True)

            # Save results to disk
            save_segmentation_files(results, base_name, segmentation_dir)
            save_preprocessed_images(results, base_name, preprocessed_dir)
            save_original_image(results, base_name, original_dir)
            
            # Flatten results for batch analysis
            flattened_data = flatten_segmentation_results(results)
            batch_results[base_name] = flattened_data
            print(f"  → Processed {base_name} successfully")
            
        except Exception as e:
            print(f"  → Failed on {base_name}: {e}")
            
        if visualization:
            display_results(results)
    
    return batch_results


def save_segmentation_files(results, base_name, output_dir):
    """Save segmentation masks and visualizations to disk."""
    output_mask_path = os.path.join(output_dir, f"{base_name}_masks.tif")
    output_vis_path = os.path.join(output_dir, f"{base_name}_masks_vis.png")
    save_results(results, output_mask_path, output_vis_path, path='masks')


def save_preprocessed_images(results, base_name, output_dir):
    """Save preprocessed channel images to disk."""
    preprocessed_channels = results['preprocessed_channels']
    
    for channel, img in preprocessed_channels.items():
        img16 = util.img_as_uint(img)  # Convert float [0,1] to uint16
        channel_path = os.path.join(output_dir, f"{channel}_preprocessed.tif")
        io.imsave(channel_path, img16)
        print(f"{channel.capitalize()} preprocessed image saved to: {channel_path}")


def save_original_image(results, base_name, output_dir):
    """Save the original channel images to disk."""
    original_images = results['original_image']
    
    for channel, img in original_images.items():
        img16 = util.img_as_uint(img)  # Convert to uint16
        channel_path = os.path.join(output_dir, f"{channel}_original.tif")
        io.imsave(channel_path, img16)
        print(f"{channel.capitalize()} original image saved to: {channel_path}")


def flatten_segmentation_results(results):
    """Convert segmentation results into a flat dictionary structure."""
    flattened = {'original_image': results['original_image']}
    
    # Add original images
    for channel, original_image in results['original_image'].items():
        flattened[f"{channel}_original_image"] = original_image
    
    # Add preprocessed channel images
    for channel, preprocessed_image in results['preprocessed_channels'].items():
        flattened[f"{channel}_preprocessed_image"] = preprocessed_image

    # Add masks
    for mask_name, mask in results['masks'].items():
        flattened[f"{mask_name}_mask"] = mask
    
    # Add flows
    for channel, flow in results['flows'].items():
        flattened[f"{channel}_flow"] = flow
    
    # Add styles
    for channel, style in results['styles'].items():
        flattened[f"{channel}_style"] = style
    
    return flattened
