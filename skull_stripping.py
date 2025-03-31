import os
import subprocess
import pandas as pd
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

def check_hd_bet():
    """Check if hd-bet command is available"""
    result = subprocess.run(["which", "hd-bet"], capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: hd-bet not found. Make sure it is installed and in your PATH.")
        return False
    return True

def process_nifti(input_file, main_folder, output_folder, device="cpu", disable_tta=True):
    """
    Process a single NIfTI file through HD-BET, generate binary mask, and apply dilation.
    
    Args:
    - input_file (str): Path to the input NIfTI file.
    - output_folder (str): Output folder for the results.
    - device (str): "cpu" or "cuda".
    - disable_tta (bool): Disable test-time augmentation.
    
    Returns:
    - dict: Status of each step (success, skipped, failed).
    """
    filename = os.path.basename(input_file)
    subfolder = os.path.relpath(os.path.dirname(input_file), start=main_folder)

    # Output paths
    hd_bet_file = os.path.join(output_folder, subfolder, f"hd_bet_{filename[len('reoriented_'):]}")
    mask_file = os.path.join(output_folder, subfolder, f"hd_bet_mask_{filename[len('reoriented_'):]}")
    dilated_file = os.path.join(output_folder, subfolder, f"hd_bet_dilated_{filename[len('reoriented_'):]}")
    print(hd_bet_file)
    
    os.makedirs(os.path.dirname(hd_bet_file), exist_ok=True)

    # Track step statuses
    status = {
        "hd_bet": "skipped" if os.path.exists(hd_bet_file) else "pending",
        "mask": "skipped" if os.path.exists(mask_file) else "pending",
        "dilated": "skipped" if os.path.exists(dilated_file) else "pending"
    }

    try:
        # Run HD-BET if output doesn't already exist
        if not os.path.exists(hd_bet_file):
            command = ["hd-bet", "-i", input_file, "-o", hd_bet_file, "-device", device]
            if disable_tta:
                command.append("--disable_tta")

            subprocess.run(command, check=True, capture_output=True, text=True)
            
            if os.path.exists(hd_bet_file):
                status["hd_bet"] = "success"
            else:
                status["hd_bet"] = "failed"

        # Create binary mask if it doesn't exist
        if not os.path.exists(mask_file):
            img = nib.load(hd_bet_file)
            data = img.get_fdata()
            
            # Generate binary mask
            binary_volume = (data > 0).astype(np.uint8)
            nib.save(nib.Nifti1Image(binary_volume, img.affine, img.header), mask_file)
            
            if os.path.exists(mask_file):
                status["mask"] = "success"
            else:
                status["mask"] = "failed"

        # Apply dilation if it doesn't exist
        if not os.path.exists(dilated_file):
            mask_img = nib.load(mask_file)
            mask_data = mask_img.get_fdata()

            struct_elem = np.ones((7, 7, 7))
            dilated_data = binary_dilation(mask_data, structure=struct_elem)

            # Apply dilated mask to the original volume
            original_img = nib.load(input_file)
            original_data = original_img.get_fdata()

            dilated_volume = original_data * dilated_data
            nib.save(nib.Nifti1Image(dilated_volume, original_img.affine, original_img.header), dilated_file)

            if os.path.exists(dilated_file):
                status["dilated"] = "success"
            else:
                status["dilated"] = "failed"

    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        status["hd_bet"] = status.get("hd_bet", "failed")
        status["mask"] = status.get("mask", "failed")
        status["dilated"] = status.get("dilated", "failed")

    return {
        "input": input_file,
        "hd_bet": status["hd_bet"],
        "mask": status["mask"],
        "dilated": status["dilated"]
    }

def run_hd_bet(main_folder, output_folder, device="cpu", disable_tta=True):
    """
    Main function to run HD-BET on all reoriented NIfTI files in subfolders.

    Args:
    - main_folder (str): Path to the main folder containing NIfTI files.
    - output_folder (str): Path where the hd-bet output files will be saved.
    - device (str): "cpu" or "cuda".
    - disable_tta (bool): Disable test-time augmentation.
    """
    if not check_hd_bet():
        return

    os.makedirs(output_folder, exist_ok=True)

    # Log file data
    log_data = []

    print(f"Starting HD-BET processing in: {main_folder}")

    for root, _, files in os.walk(main_folder):
        for file in files:
            if file.startswith("reoriented_") and file.endswith(".nii.gz"):
                input_file = os.path.join(root, file)
                print(f"Processing: {input_file}")
                result = process_nifti(input_file, main_folder, output_folder, device, disable_tta)
                log_data.append(result)

    # Save log
    df = pd.DataFrame(log_data)
    log_file = os.path.join(output_folder, "hd_bet_log.csv")
    
    if os.path.exists(log_file):
        # Append to existing log
        existing_df = pd.read_csv(log_file)
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_csv(log_file, index=False)
    print(f"HD-BET processing completed. Log saved at: {log_file}")

# ---------------------------
# Execution example (for running as a standalone script)
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run HD-BET on reoriented NIfTI files in subfolders.")
    parser.add_argument("main_folder", help="Path to the main folder containing NIfTI files.")
    parser.add_argument("output_folder", help="Path to the output folder for processed files.")
    parser.add_argument("--device", default="cpu", help="Device to use for processing (cpu or cuda).")
    parser.add_argument("--disable_tta", type=bool, default=True, help="Disable test-time augmentation (default=True).")
    
    args = parser.parse_args()

    # Execute the function with the provided arguments
    run_hd_bet(args.main_folder, args.output_folder, args.device, args.disable_tta)

"""
python skull_stripping.py "./data/example_input_images" "./data/example_input_hdbet_processed"
"""
