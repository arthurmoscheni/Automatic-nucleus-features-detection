import os
import tkinter as tk
from tkinter import filedialog
import warnings

warnings.filterwarnings('ignore')

# Import segmentation module
from segmentation import run_batch_segmentation
from image_analysis import prepare_image_data_for_analysis, run_morphological_analysis, print_analysis_summary
from dna_analysis import prepare_dna_data_for_analysis, run_dna_analysis

visualization = False  # Set to False to disable visualizations

def select_input_folder():
    """Select input folder containing subfolders with images."""
    root = tk.Tk()
    root.withdraw()
    return filedialog.askdirectory(title="Select folder containing subfolders with images")


def select_output_directory():
    """Select output directory for results."""
    root = tk.Tk()
    root.withdraw()
    return filedialog.askdirectory(title="Select output directory")


def get_image_files(folder_path):
    """Get all image files from the selected folder."""
    image_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
            if f.lower().endswith(image_extensions)]


def get_subfolders_with_images(parent_folder):
    """Get all subfolders that contain image files."""
    subfolders = {}
    for item in os.listdir(parent_folder):
        item_path = os.path.join(parent_folder, item)
        if os.path.isdir(item_path):
            image_files = get_image_files(item_path)
            if image_files:
                subfolders[item] = image_files
    return subfolders


def process_folder(folder_name, image_list, output_dir, visualization=visualization):
    """Process a single folder with images."""
    print(f"\n{'='*50}")
    print(f"Processing folder: {folder_name}")
    print(f"{'='*50}")
    
    # Create output subfolder
    folder_output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(folder_output_dir, exist_ok=True)
    
    # Run batch segmentation
    batch_results = run_batch_segmentation(image_list, folder_output_dir, visualization=False)
    
    # Morphological analysis
    image_data_list_morpho = prepare_image_data_for_analysis(batch_results)
    print(f"✓ Prepared {len(image_data_list_morpho)} images for morphological analysis")
    
    morpho_results = run_morphological_analysis(
        image_data_list_morpho, folder_output_dir, 
        visualization=visualization, pixel_size_um=0.124
    )
    print_analysis_summary(morpho_results)
    
    # DNA analysis
    print("Preparing DNA data...")
    dna_data_list = prepare_dna_data_for_analysis(batch_results)
    print(f"✓ Prepared {len(dna_data_list)} DNA dataset")
    
    print("Running DNA analysis...")
    run_dna_analysis(dna_data_list, folder_output_dir, visualization=visualization)
    
    print(f"✓ Completed processing folder: {folder_name}")


def main():
    """Main function to orchestrate the image processing workflow."""
    # Select input folder containing subfolders
    parent_folder = select_input_folder()
    if not parent_folder:
        return
    
    # Get subfolders with images
    subfolders_dict = get_subfolders_with_images(parent_folder)
    if not subfolders_dict:
        print("No subfolders with image files found in selected folder")
        return
    
    print(f"Found {len(subfolders_dict)} subfolders with images:")
    for folder_name in subfolders_dict.keys():
        print(f"  - {folder_name}")
    
    # Select output directory
    output_dir = select_output_directory()
    if not output_dir:
        return
    
    # Process each subfolder
    for folder_name, image_list in subfolders_dict.items():
        process_folder(folder_name, image_list, output_dir)
    
    print(f"\n{'='*50}")
    print("All folders processed successfully!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()