import os
import glob
import pickle
import argparse
import numpy as np
from PIL import Image


def resize_image(img, size=(224, 224)):
    """
    Resizes an image to match the TacStyle model input size.

    Args:
        img (np.ndarray): Input image in HxWx3 or 3xHxW format.
        size (tuple): Target dimensions (width, height).

    Returns:
        np.ndarray: Resized image as a 3xH'xW' float32 array in the range [0, 1].
    """
    if img.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {img.shape}")

    # Convert CHW -> HWC if needed (PyTorch uses CHW, PIL expects HWC)
    if img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))

    if img.shape[-1] != 3:
        raise ValueError(f"Expected 3 channels, got shape {img.shape}")

    # Convert image to uint8 for PIL
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    # Resize using bilinear interpolation
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(size, Image.BILINEAR)

    # Convert back to float32 in [0, 1] and CHW format
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))

    return arr


def process_cloth_wipe_sim_data(input_root, output_root):
    """
    Converts raw simulated cloth wipe demonstrations into a standardized training format.

    Input formats:
      1. Trajectory pickle: A list of dictionaries containing robot states per timestep.
      2. Image pickle: A list of camera frames corresponding to the trajectory.

    Output formats:
      joints:   [T, joint_dim] (float32 array)
      gripper:  [T, 1] (float32 array)
      tactile:  [T, tactile_dim] (float32 array)
      images:   [T, 3, 224, 224] (float32 array)
    """
    os.makedirs(output_root, exist_ok=True)

    # Find all trajectory files (excluding the _images.pkl files)
    all_pkl_files = sorted(glob.glob(os.path.join(input_root, "*.pkl")))
    traj_files = [f for f in all_pkl_files if not f.endswith("_images.pkl")]

    if not traj_files:
        print(f"No valid trajectory .pkl files found in {input_root}")
        return

    for traj_path in traj_files:
        # Determine corresponding image file
        filename_base = os.path.splitext(os.path.basename(traj_path))[0]
        img_path = os.path.join(input_root, f"{filename_base}_images.pkl")

        if not os.path.exists(img_path):
            print(f"Skipping {filename_base}: missing image file.")
            continue

        output_path = os.path.join(output_root, os.path.basename(traj_path))

        # Load data
        with open(traj_path, "rb") as f:
            traj_list = pickle.load(f)

        with open(img_path, "rb") as f:
            img_list = pickle.load(f)

        joints = []
        gripper = []
        tactile = []

        # Extract timestep arrays
        for step_data in traj_list:
            joints.append(step_data["joint_angles"])
            gripper.append([step_data["gripper_q"]])
            tactile.append(step_data["tactile_data"])

        demo_dict = {
            "joints": np.array(joints, dtype=np.float32),
            "gripper": np.array(gripper, dtype=np.float32),
            "tactile": np.array(tactile, dtype=np.float32),
        }

        # Resize all camera frames
        resized_images = [resize_image(img, size=(224, 224)) for img in img_list]
        demo_dict["images"] = np.stack(resized_images, axis=0).astype(np.float32)

        # Save processed demo
        with open(output_path, "wb") as f:
            pickle.dump(demo_dict, f)

        print(f"Saved resized demo: {output_path} | images shape: {demo_dict['images'].shape}")

    print(f"Wrote {len(traj_files)} files to {output_root}")


def process_cloth_fold_sim_data(input_root, output_root):
    """
    Processes simulated cloth fold demonstrations into a standard training format.
    
    This function standardizes trajectories by ensuring all sequence data are
    properly formatted as numpy arrays.
    """
    pkl_files = sorted(glob.glob(os.path.join(input_root, "*.pkl")))
    if not pkl_files:
        print(f"No .pkl files found in {input_root}")
        return

    os.makedirs(output_root, exist_ok=True)

    for fpath in pkl_files:
        with open(fpath, "rb") as f:
            demo = pickle.load(f)
            
        demo["joints"] = np.array(demo["joints"], dtype=np.float32)
        demo["gripper"] = np.array(demo["gripper"], dtype=np.float32)
        
        if "tcp" in demo:
            demo["tcp"] = np.array(demo["tcp"], dtype=np.float32)
        if "tactile" in demo:
            demo["tactile"] = np.array(demo["tactile"], dtype=np.float32)
        if "images" in demo:
            demo["images"] = np.array(demo["images"], dtype=np.float32)

        out_path = os.path.join(output_root, os.path.basename(fpath))
        with open(out_path, "wb") as f:
            pickle.dump(demo, f)

        print(f"Preprocessed {os.path.basename(fpath)} | Length: {len(demo['joints'])}")

    print(f"Wrote {len(pkl_files)} files to {output_root}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess tactile-learning datasets.")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["cloth_wipe_sim", "cloth_fold_sim"],
        help="Dataset task to preprocess."
    )
    args = parser.parse_args()

    input_dir = f"main/data_{args.task}"
    output_dir = f"main/data_{args.task}_processed"

    if args.task == "cloth_wipe_sim":
        process_cloth_wipe_sim_data(input_root=input_dir, output_root=output_dir)
    elif args.task == "cloth_fold_sim":
        process_cloth_fold_sim_data(input_root=input_dir, output_root=output_dir)

    print("Done preprocessing.")