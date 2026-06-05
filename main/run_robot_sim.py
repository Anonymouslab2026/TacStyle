import os
import sys
import math
import time
import pickle

import numpy as np
# pyrefly: ignore [missing-import]
import pybullet as p
# pyrefly: ignore [missing-import]
import pybullet_data
import torch

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

# This file lives inside main/
main_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(main_dir)

# sim_environments path
SIM_ENV_DIR = os.path.join(main_dir, "..", "sim_environments")
sys.path.insert(0, os.path.abspath(SIM_ENV_DIR))

# Fairino / tactile SDK path
SDK_DIR = os.path.join(main_dir, "..", "sim_environments", "python-sdk")
sys.path.insert(0, os.path.abspath(SDK_DIR))

# ---------------------------------------------------------------------
# Imports from project
# ---------------------------------------------------------------------

from models import TacStylePolicy, PolicyConfig
from preprocess_data import resize_image
from language_api import get_z_value
# pyrefly: ignore [missing-import]
from tactile_gym.tactile_sensor import TactileSensor
# pyrefly: ignore [missing-import]
from cloth_wipe_env import setup_cloth_wipe_sim, move_to_pose, anchor_cloth_to_gripper
from cloth_fold_env import setup_cloth_fold_env, post_rollout_cloth_fold
from train_model import OracleAgent
# pyrefly: ignore [missing-import]
from utils import patch_urdf, actuate_gripper


# ---------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------

def normalize_data(x, mu, std):
    """Normalize raw input using mean/std from training."""
    return (x - mu) / std


def denormalize_data(x, mu, std):
    """Convert normalized data back to raw scale."""
    return (x * std) + mu


# ---------------------------------------------------------------------

# Environment Setup Functions
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Single cloth simulation rollout
# ---------------------------------------------------------------------

def run_single_sim(
    model,
    device,
    joints_mu,
    joints_std,
    tactile_mu,
    tactile_std,
    z_current=None,
    language_text=None,
    num_steps=150,
    record=False,
    task_name="cloth_wipe_sim",
):
    """
    Run one cloth_wipe_sim simulation using a fixed z value.

    Flow:
      fixed z -> TacStylePolicy -> predicted joint action -> PyBullet rollout
    """

    if "cloth_fold" in task_name:
        robot_id, arm_joints, gripper_joints, grasp_index, tactile_sensor, view_mat, proj_mat = setup_cloth_fold_env(p, record)
    else:
        robot_id, arm_joints, gripper_joints, grasp_index, tactile_sensor, view_mat, proj_mat = setup_cloth_wipe_sim(p, record)

    # -----------------------------------------------------------------
    # Phase 2: neural policy control
    # -----------------------------------------------------------------

    print("Starting Neural Network Control Loop...")

    width, height = 320, 180

    history_states = []
    history_tactile = []
    history_images = []
    traj_data = []

    commanded_state_norm = None

    for step in range(num_steps):
        # -------------------------------------------------------------
        # 1. Query current simulation state
        # -------------------------------------------------------------

        joints = [p.getJointState(robot_id, j)[0] for j in arm_joints[:6]]

        gripper_q = p.getJointState(
            robot_id,
            gripper_joints["gripper_finger1_inner_knuckle_joint"],
        )[0]

        if "cloth_fold" in task_name:
            # Use zeros for missing tactile data in cloth_fold simulation
            tactile_data = np.zeros(15, dtype=np.float32)
        else:
            # Tactile data is summarized as image mean values for cloth_wipe_sim.
            tactile_imgs = tactile_sensor.get_imgs()
            tactile_data = np.array(
                [np.mean(img) for img in tactile_imgs],
                dtype=np.float32,
            )

        # RGB camera image
        _, _, rgb_img, _, _ = p.getCameraImage(
            width,
            height,
            viewMatrix=view_mat,
            projectionMatrix=proj_mat,
            renderer=p.ER_TINY_RENDERER,
        )

        frame_rgb = np.reshape(rgb_img, (height, width, 4)).astype(np.uint8)[:, :, :3]

        # Record rollout data if evaluating
        if record:
            xyz = p.getLinkState(robot_id, grasp_index)[0]

            traj_data.append(
                {
                    "time_step": step,
                    "joint_angles": joints,
                    "gripper_q": gripper_q,
                    "tactile_data": tactile_data.tolist(),
                    "xyz": list(xyz),
                }
            )

        # -------------------------------------------------------------
        # 2. Normalize data exactly as training used
        # -------------------------------------------------------------

        tactile_norm = normalize_data(
            tactile_data,
            tactile_mu,
            tactile_std,
        ).flatten()

        if "cloth_fold" in task_name:
            joints_input = [math.degrees(j) for j in joints]
            # Real data gripper is ~[100, 0] meaning open
            gripper_input = [100.0, 0.0]
        else:
            joints_input = joints
            gripper_input = [gripper_q]

        state_raw = np.hstack([joints_input, gripper_input]).astype(np.float32)

        state_norm = normalize_data(
            state_raw,
            joints_mu,
            joints_std,
        ).flatten()

        if commanded_state_norm is None:
            commanded_state_norm = state_norm.copy()

        resized_img = resize_image(frame_rgb, size=(224, 224))

        # Add initial padding history.
        if step == 0:
            pad_len = 10
            history_states = [np.zeros_like(state_norm) for _ in range(pad_len)]
            history_tactile = [np.zeros_like(tactile_norm) for _ in range(pad_len)]
            history_images = [np.zeros_like(resized_img) for _ in range(pad_len)]

        history_states.append(commanded_state_norm.copy())
        history_tactile.append(tactile_norm)
        history_images.append(resized_img)

        states_t = torch.from_numpy(
            np.stack(history_states)
        ).float().unsqueeze(0).to(device)

        tactile_t = torch.from_numpy(
            np.stack(history_tactile)
        ).float().unsqueeze(0).to(device)

        images_t = torch.from_numpy(
            np.stack(history_images)
        ).float().unsqueeze(0).to(device)

        # Fixed style value for this rollout.
        z_style_t = torch.tensor(
            [[z_current]],
            dtype=torch.float32,
            device=device,
        ) if z_current is not None else None

        # -------------------------------------------------------------
        # 3. Predict normalized action
        # -------------------------------------------------------------

        with torch.no_grad():
            pred_actions, _ = model(
                images_t,
                tactile_t,
                states_t,
                z_style=z_style_t,
                language_text=language_text,
                actions=None,
            )

        actions = pred_actions[0, -1].detach().cpu().numpy().flatten()

        # -------------------------------------------------------------
        # 4. Apply action
        # -------------------------------------------------------------

        # Cloth actions are delta in normalized joint-state space.
        commanded_state_norm = commanded_state_norm + actions

        # Convert back to raw joint/gripper target.
        new_state = denormalize_data(
            commanded_state_norm,
            joints_mu,
            joints_std,
        ).flatten()

        if "cloth_fold" in task_name:
            joint_target = np.asarray([math.radians(j) for j in new_state[:6]], dtype=float)
            gripper_target = 0.0
        else:
            joint_target = np.asarray(new_state[:6], dtype=float)
            gripper_target = float(new_state[6])

        p.setJointMotorControlArray(
            robot_id,
            arm_joints[:6],
            p.POSITION_CONTROL,
            targetPositions=joint_target.tolist(),
        )

        actuate_gripper(robot_id, gripper_joints, q=gripper_target, force=100)

        p.stepSimulation()

        if not record:
            time.sleep(1.0 / 120.0)

    if "cloth_fold" in task_name:
        post_rollout_cloth_fold(p, robot_id, arm_joints, gripper_joints, grasp_index)

    if not record:
        print("Simulation finished.")
        input("Press Enter to close simulation...")

    return traj_data


# ---------------------------------------------------------------------
# Load model/checkpoint
# ---------------------------------------------------------------------

def load_policy_and_stats(task_name, conditioning_mode, device, num_labels=None):
    """
    Load TacStylePolicy checkpoint and normalization statistics.
    """
    import glob
    if conditioning_mode == "language":
        if num_labels is not None:
            pattern = os.path.join(main_dir, "learned", f"tacstyle_{task_name}_language_{num_labels}labels_model.pt")
        else:
            pattern = os.path.join(main_dir, "learned", f"tacstyle_{task_name}_language_*labels_model.pt")
            
        matches = glob.glob(pattern)
        if len(matches) == 0:
            raise FileNotFoundError(f"No language model found matching {pattern}. Use --num_labels to specify.")
            
        if num_labels is None and len(matches) > 1:
            print(f"Warning: Multiple language models found for {task_name}. Loading the latest one: {os.path.basename(matches[-1])}")
            print("You can specify a specific label count using --num_labels")
            
        model_path = matches[-1]
    else:
        model_path = os.path.join(main_dir, "learned", f"tacstyle_{task_name}_pdr_model.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(main_dir, "learned", f"tacstyle_{task_name}_model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    cfg = PolicyConfig(**ckpt["cfg"])

    model = TacStylePolicy(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    required_keys = [
        "joints_mu",
        "joints_std",
        "tactile_mu",
        "tactile_std",
        "z_min",
        "z_max",
    ]

    for key in required_keys:
        if key not in ckpt:
            raise KeyError(
                f"Missing key '{key}' in checkpoint. "
                f"Retrain using updated train_model.py."
            )

    stats = {
        "joints_mu": ckpt["joints_mu"],
        "joints_std": ckpt["joints_std"],
        "tactile_mu": ckpt["tactile_mu"],
        "tactile_std": ckpt["tactile_std"],
        "z_min": float(ckpt["z_min"]) if ckpt.get("z_min") is not None else None,
        "z_max": float(ckpt["z_max"]) if ckpt.get("z_max") is not None else None,
    }

    return model, stats


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def run_robot_sim():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        default="cloth_wipe_sim",
        choices=["cloth_wipe_sim", "cloth_fold_sim"],
        help="Task name to simulate (cloth_wipe_sim or cloth_fold_sim).",
    )

    parser.add_argument(
        "--conditioning_mode",
        type=str,
        default="z",
        choices=["z", "language"],
        help="Conditioning mode of the model (z or language).",
    )

    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of neural policy steps per rollout.",
    )

    parser.add_argument(
        "--num_labels",
        type=int,
        default=3,
        help="Number of discrete labels used for the baseline language model (e.g., 3). Only used if conditioning_mode is 'language'.",
    )

    args = parser.parse_args()

    task_name = args.task

    # Use CUDA if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load trained policy and stats.
    model, stats = load_policy_and_stats(task_name, args.conditioning_mode, device, num_labels=args.num_labels)

    joints_mu = stats["joints_mu"]
    joints_std = stats["joints_std"]
    tactile_mu = stats["tactile_mu"]
    tactile_std = stats["tactile_std"]
    z_min = stats["z_min"]
    z_max = stats["z_max"]

    if z_min is None or z_max is None:
        print("Loaded language model.")
    else:
        print(f"Loaded ours with z range: [{z_min:.4f}, {z_max:.4f}]")

    # -----------------------------------------------------------------
    # Interactive language-conditioned mode
    # -----------------------------------------------------------------

    if args.conditioning_mode == "language":
        user_statement = input("Enter your request (e.g., 'fold tightly' or 'wipe gently'): ")
        z_current = None
        language_text = [user_statement]
        print(f"Policy conditioned directly with text: '{user_statement}'")
        input("Press Enter to continue...")
    else:
        mapping_path = os.path.join(main_dir, "learned", f"tacstyle_{task_name}_pdr_z_mapping.pkl")
        if not os.path.exists(mapping_path):
            mapping_path = os.path.join(main_dir, "learned", f"tacstyle_{task_name}_z_mapping.pkl")
            
        with open(mapping_path, "rb") as f:
            mapping = pickle.load(f)
        
        styles = mapping["styles"]
        z_vals = mapping["z_values"]
        
        style_for_z_min = styles[np.argmin(z_vals)]
        style_for_z_max = styles[np.argmax(z_vals)]
        
        oracle = OracleAgent(use_cache=False)
        if task_name == "cloth_wipe_sim":
            z_min_label = oracle.generate_prompt(task_name, f"height={float(style_for_z_min):.3f}")
            z_max_label = oracle.generate_prompt(task_name, f"height={float(style_for_z_max):.3f}")
        else:
            z_min_label = oracle.generate_prompt(task_name, str(style_for_z_min))
            z_max_label = oracle.generate_prompt(task_name, str(style_for_z_max))
            
        print(f"Automatically retrieved z_min_label: '{z_min_label}'")
        print(f"Automatically retrieved z_max_label: '{z_max_label}'")

        user_statement = input("Enter your request (e.g., 'fold tightly' or 'wipe gently'): ")

        print(f"Querying Language API with z_min={z_min:.4f}, z_max={z_max:.4f}...")
        
        z_current = get_z_value(
            z_current=(z_min + z_max) / 2.0,
            user_statement=user_statement,
            z_min=z_min,
            z_max=z_max,
            z_min_label=z_min_label,
            z_max_label=z_max_label,
        )
        language_text = None
        print(f"Policy conditioned with z_value: {z_current:.4f}")
        input("Press Enter to continue...")

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    run_single_sim(
        model=model,
        device=device,
        joints_mu=joints_mu,
        joints_std=joints_std,
        tactile_mu=tactile_mu,
        tactile_std=tactile_std,
        z_current=z_current,
        language_text=language_text,
        num_steps=args.num_steps,
        record=False,
        task_name=task_name,
    )

    p.disconnect()

if __name__ == "__main__":
    run_robot_sim()
