import os
import pickle
import argparse
import glob
import sys

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

from models import *
from openai import OpenAI


class OracleAgent:
    def __init__(self, use_cache=False):
        self.client = OpenAI()
        self.use_cache = use_cache
        self.cache = {}
        
        self.task_descriptions = {
            "cloth_wipe_sim": "A robotic arm with a tactile sensor wiping across a cloth surface. The 'height' parameter controls the contact force; lower heights mean stronger contact and higher heights mean gentler contact.",
            "cloth_fold_sim": "A simulated robotic arm folding a piece of cloth on a table. The 'style' parameter controls the tightness of the fold, where 0.0 means the loosest fold and 1.0 means the tightest fold."
        }
        
        example_inputs = {
            "cloth_wipe_sim": ["height=0.05", "height=0.10", "height=0.15"],
            "cloth_fold_sim": ["style=0.0", "style=0.5", "style=1.0"]
        }
        
        self.examples = {}
        for task, inputs in example_inputs.items():
            self.examples[task] = "\n\n".join([
                f"Ground Truth: {val}\nOutput: {self._get_hardcoded_prompt(task, val)}"
                for val in inputs
            ])

    def _get_hardcoded_prompt(self, task, style_value):
        if task == "cloth_wipe_sim":
            try:
                height = float(style_value.split("=")[1])
                if height <= 0.05:
                    return "wipe with stronger contact and lower height"
                if height >= 0.125:
                    return "wipe gently with lighter contact and higher height"
                return "wipe with moderate contact"
            except:
                return "wipe with moderate contact"

        elif task == "cloth_fold_sim":
            try:
                style_num = float(style_value.split("=")[1])
                if style_num <= 0.2:
                    return "fold the cloth loosely"
                if style_num >= 0.8:
                    return "fold the cloth tightly"
                return "fold the cloth with medium tightness"
            except:
                return "fold the cloth with medium tightness"
            
        return "perform the task normally"

    def generate_prompt(self, task, style_value):
        if self.use_cache:
            if (task, style_value) in self.cache:
                return self.cache[(task, style_value)]
            
            # If not in cache and cache mode is on, fallback to hardcoded
            hardcoded = self._get_hardcoded_prompt(task, style_value)
            self.cache[(task, style_value)] = hardcoded
            return hardcoded
            
        system_prompt = (
            "You are an expert robotics data annotator. Your job is to generate a short, natural language command "
            "that a human would give to a robot to perform a task with a specific style.\n\n"
            f"Task Context: {self.task_descriptions.get(task, 'A generic robotic task.')}\n\n"
            "Generate a slightly diverse, natural-sounding instruction that captures the specified style. "
            "Do NOT include conversational filler. Just output the command (e.g., 'wipe the cloth gently').\n\n"
            "Examples for this task:\n"
            f"{self.examples.get(task, '')}"
        )
        
        user_prompt = f"Ground Truth: {style_value}\nOutput:"
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=30,
        )
        
        generated_text = response.choices[0].message.content.strip()
        
        return generated_text


class TacStyleDataset(Dataset):
    """
    Dataset wrapper to support both:
      - z mode: tensors only
      - language mode: tensors + text prompt
    """

    def __init__(self, images, tactile, states, actions, language_texts=None):
        self.images = torch.from_numpy(images).float()
        self.tactile = torch.from_numpy(tactile).float()
        self.states = torch.from_numpy(states).float()
        self.actions = torch.from_numpy(actions).float()
        self.language_texts = language_texts

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        if self.language_texts is None:
            return (
                self.images[idx],
                self.tactile[idx],
                self.states[idx],
                self.actions[idx],
            )

        return (
            self.images[idx],
            self.tactile[idx],
            self.states[idx],
            self.actions[idx],
            self.language_texts[idx],
        )


class StylePDRLoss(nn.Module):
    """
    Pairwise distance regression loss:

        |z_i - z_j| ≈ alpha * d_traj(x_i, x_j)

    where alpha is a single learned global scale.
    """

    def __init__(self, init_alpha=1.0):
        super().__init__()
        self.log_alpha = nn.Parameter(
            torch.log(torch.tensor(float(init_alpha)))
        )

    def forward(self, pred_z, traj):
        z = pred_z.view(-1)
        B = z.shape[0]

        if B < 2:
            return pred_z.sum() * 0.0

        # Pairwise trajectory distances
        diff = traj.unsqueeze(0) - traj.unsqueeze(1)      # [B, B, T', D]
        d_traj = diff.norm(dim=-1).mean(dim=-1).detach()  # [B, B]

        # Pairwise latent distances
        d_z = (z[:, None] - z[None, :]).abs()             # [B, B]

        # Use only upper triangle (unique unordered pairs)
        mask = torch.triu(
            torch.ones(B, B, dtype=torch.bool, device=z.device),
            diagonal=1,
        )

        d_z = d_z[mask]
        d_traj = d_traj[mask]

        alpha = torch.exp(self.log_alpha)

        loss = ((d_z - alpha * d_traj) ** 2).mean()

        return loss


def balance_data(data, method="last"):
    if method == "clip":
        min_demo_len = min(len(demo) for demo in data)
        return np.array([demo[:min_demo_len] for demo in data], dtype=np.float32)

    max_demo_len = max(len(demo) for demo in data)
    balanced_data = []
    
    for demo in data:
        pad_len = max_demo_len - len(demo)
        if method == "last":
            pad_arr = np.repeat(demo[-1:], pad_len + 5, axis=0)  # pad extra 5 steps to all demos to indicate stopping
        elif method == "zero":
            pad_arr = np.zeros((pad_len + 5,) + demo.shape[1:], dtype=demo.dtype)
        demo = np.concatenate([demo, pad_arr], axis=0)
        balanced_data.append(demo)

    return np.array(balanced_data, dtype=np.float32)


def normalize_list(data, method="per_dim", min_std=1e-8):
    all_data = np.concatenate(data, axis=0)
    
    if method == "global":
        mu = all_data.mean()
        std = all_data.std()
    else:
        mu = all_data.mean(0, keepdims=True)
        if method == "max_std":
            std = np.max(all_data.std(0))
        else: # per_dim
            std = all_data.std(0, keepdims=True)
            
    std = np.maximum(std, min_std)

    return [(x - mu) / std for x in data], mu, std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["cloth_wipe_sim", "cloth_fold_sim"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--conditioning_mode", type=str, default="z", choices=["z", "language"])
    parser.add_argument("--use_cache", action="store_true", help="Use cached oracle prompts instead of generating new ones")
    parser.add_argument("--num_labels", default="all", help="Number of language labels to use")
    parser.add_argument("--target_loss", type=float, default=0.0, help="Stop training if MSE loss drops below this threshold.")
    parser.add_argument("--lambda_pdr", type=float, default=0.0, help="Weight for the proportionality loss.")
    args = parser.parse_args()

    oracle = OracleAgent(use_cache=args.use_cache)

    data_dir = f"main/data_{args.task}_processed"

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Conditioning mode:", args.conditioning_mode)
    
    # find all pickle files
    pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    if not pkl_files:
        print(f"No .pkl files found in {data_dir}")
        sys.exit(1)

    # get names of all demo files (excluding image pkl files)
    demo_files = []
    for file_path in pkl_files:
        if "_images" in file_path:
            continue
        demo_files.append(file_path)

    # load training data
    joints_data = []
    tcp_data = []
    images_data = []
    tactile_data = []

    # metadata
    actual_styles = []
    language_texts = []

    # -------------------------------------------------------------------
    # Load all demonstrations
    # -------------------------------------------------------------------
    for file_path in demo_files:

        demo = pickle.load(open(file_path, "rb"))

        states_joints = np.hstack([demo["joints"], demo["gripper"]]).astype(np.float32)
        if "tcp" in demo:
            states_tcp = np.hstack([demo["tcp"], demo["gripper"]]).astype(np.float32)
        else:
            states_tcp = states_joints.copy()
        
        T = states_joints.shape[0]

        if "tactile" in demo:
            tactile = np.asarray(demo["tactile"], dtype=np.float32)
        else:
            tactile = np.zeros((T, 3), dtype=np.float32)

        if "images" in demo:
            images = np.asarray(demo["images"], dtype=np.float32)
        else:
            images = np.zeros((T, 3, 224, 224), dtype=np.float32)

        if args.task == "cloth_wipe_sim":
            filename = os.path.basename(file_path)
            h_str = filename.split("_h")[1].split("_d")[0]
            actual_styles.append(float(h_str))
            language_texts.append(oracle.generate_prompt(args.task, f"height={float(h_str)}"))

        elif args.task == "cloth_fold_sim":
            filename = os.path.basename(file_path)
            style_str = filename.split("_")[2].split(".")[0]
            if style_str == "loose":
                style_val = 0.0
            elif style_str == "tight":
                style_val = 1.0
            else:
                style_val = 0.5
            
            actual_styles.append(style_val)
            language_texts.append(oracle.generate_prompt(args.task, f"style={style_val}"))

        # store loaded data
        joints_data.append(states_joints)
        tcp_data.append(states_tcp)
        images_data.append(images)
        tactile_data.append(tactile)

        print(f"loaded {file_path} with length {len(states_joints)}")

    # -------------------------------------------------------------------
    # Selectively blank out language tokens
    # -------------------------------------------------------------------
    unique_styles = sorted(set(actual_styles))
    num_labels = len(unique_styles) if args.num_labels == "all" else int(args.num_labels)

    if args.conditioning_mode == "language" and num_labels < len(unique_styles):
        # Pick n evenly spaced styles from the sorted unique list
        selected_styles = [unique_styles[i * (len(unique_styles) - 1) // max(1, num_labels - 1)] for i in range(num_labels)]
        
        # Replace text with " " for unselected styles to prevent tokenizer crash
        language_texts = [text if style in selected_styles else " " for text, style in zip(language_texts, actual_styles)]

    print("\n--- Generated Language Texts ---")
    for fp, text in zip(demo_files, language_texts):
        print(f"{os.path.basename(fp)}: {text}")
    print("--------------------------------\n")

    # -------------------------------------------------------------------
    # Normalize and balance data
    # -------------------------------------------------------------------
    # print stds before normalization
    np.set_printoptions(suppress=True)
    print("Before normalization: ")
    print("Joints std: ", np.std(np.concatenate(joints_data, axis=0), axis=0))

    # normalize data
    joints_data, joints_mu, joints_std = normalize_list(joints_data, method="max_std", min_std=0.01)
    tcp_data, tcp_mu, tcp_std = normalize_list(tcp_data, method="max_std", min_std=0.01)
    tactile_data, tactile_mu, tactile_std = normalize_list(tactile_data, method="per_dim", min_std=0.1)



    print("After normalization: ")
    norm_states_std = np.std(np.concatenate(joints_data, axis=0), axis=0)
    norm_tactile_std = np.std(np.concatenate(tactile_data, axis=0), axis=0)
    print("Joints std: ", norm_states_std)

    # convert to tensors
    norm_states_std = torch.from_numpy(norm_states_std).float().to(device)
    norm_tactile_std = torch.from_numpy(norm_tactile_std).float().to(device)

    # balance_data
    joints_data = balance_data(joints_data)
    tcp_data = balance_data(tcp_data)
    tactile_data = balance_data(tactile_data)
    if args.task == "cloth_fold_sim":
        images_tensor = np.zeros((joints_data.shape[0], joints_data.shape[1], 3, 224, 224))
    images_data = balance_data(images_data)

    # -------------------------------------------------------------------
    # Build states/actions for training
    # -------------------------------------------------------------------
    states_data = joints_data.copy()  # Using joints as states. Replace with tcp if needed.
    actions_data = states_data[:, 1:, :] - states_data[:, :-1, :]
    states_data = states_data[:, :-1, :]
    tactile_data = tactile_data[:, :-1, :]
    images_data = images_data[:, :-1, :, :, :]
    
    # training data shapes
    _, _, state_dim = states_data.shape
    _, _, action_dim = actions_data.shape
    _, _, tactile_dim = tactile_data.shape

    # -------------------------------------------------------------------
    # Add padding at beginning of every trajectory
    # -------------------------------------------------------------------
    PAD_LEN = 10
    B = states_data.shape[0]

    states_pad = np.zeros((B, PAD_LEN, state_dim), dtype=np.float32)
    tactile_pad = np.zeros((B, PAD_LEN, tactile_dim), dtype=np.float32)
    images_pad = np.zeros(
        (B, PAD_LEN, images_data.shape[2], images_data.shape[3], images_data.shape[4]),
        dtype=np.float32,
    )
    actions_pad = np.zeros((B, PAD_LEN, action_dim), dtype=np.float32)

    # Concatenate padding before real trajectory
    states_data = np.concatenate([states_pad, states_data], axis=1)
    tactile_data = np.concatenate([tactile_pad, tactile_data], axis=1)
    images_data = np.concatenate([images_pad, images_data], axis=1)
    actions_data = np.concatenate([actions_pad, actions_data], axis=1)

    # -------------------------------------------------------------------
    # Create PyTorch dataset
    # -------------------------------------------------------------------
    if args.conditioning_mode == "z":
        language_texts = None

    train_dataset = TacStyleDataset(
        images_data,
        tactile_data,
        states_data,
        actions_data,
        language_texts=language_texts,
    )

    # -------------------------------------------------------------------
    # Configure TacStylePolicy
    # -------------------------------------------------------------------
    cfg = PolicyConfig(
        d_model=256,
        nhead=4,
        num_layers=4,
        state_dim=state_dim,
        action_dim=action_dim,
        tactile_dim=tactile_dim,
        traj_stride=4,
        use_adaln=False,
        use_vision=False if args.task == "cloth_fold_sim" else True,
        vision_pretrained=False if args.task == "cloth_fold_sim" else True,
        conditioning_mode=args.conditioning_mode,
        qwen_model_name="Qwen/Qwen2-0.5B-Instruct"
    )

    # -------------------------------------------------------------------
    # Initialize model
    # -------------------------------------------------------------------
    torch.manual_seed(0)

    model = TacStylePolicy(cfg).to(device)
    model.train()

    # data loader
    BATCH_SIZE = args.batch_size
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # optimizer
    EPOCHS = args.epochs
    LR = args.lr
    mse = nn.MSELoss()
    pdr = StylePDRLoss().to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters()) + list(pdr.parameters())), lr=LR)
    losses = []

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------
    
    # pre-calculate constants for noise augmentation
    NOISE_SCALE = 0.05
    TACTILE_NOISE_SCALE = 0.05
    apply_corrective_noise = True
    states_std_tensor = norm_states_std.view(1, 1, -1) + 1e-6
        
    tactile_std_tensor = norm_tactile_std.view(1, 1, -1) + 1e-6

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_prop_loss = 0.0

        for batch in train_loader:
            images = batch[0].to(device)
            tactile = batch[1].to(device)
            states = batch[2].to(device)
            actions = batch[3].to(device)
            batch_language_texts = list(batch[4]) if len(batch) > 4 else None

            # noise augmentation
            noise = torch.randn_like(states) * NOISE_SCALE * states_std_tensor
            noise[:, :PAD_LEN, :] = 0.0
            
            target_actions = actions - noise if apply_corrective_noise else actions.clone()
            noisy_states = states + noise

            tactile_noise = torch.randn_like(tactile) * TACTILE_NOISE_SCALE * tactile_std_tensor
            tactile_noise[:, :PAD_LEN, :] = 0.0
            noisy_tactile = tactile + tactile_noise

            # # tactile sensor dropout (10% chance to drop out a sensor reading at any timestep)
            # tactile_dropout_mask = (torch.rand_like(tactile) > 0.1).float()
            # noisy_tactile = noisy_tactile * tactile_dropout_mask

            # prediction
            pred_actions, pred_z = model(
                images,
                noisy_tactile,
                noisy_states,
                z_style=None,
                language_text=batch_language_texts,
                actions=target_actions,
            )

            loss = mse(pred_actions[:, PAD_LEN:, :], target_actions[:, PAD_LEN:, :])
            epoch_mse_loss += loss.item()
            
            if args.conditioning_mode == "z":
                prop_loss = args.lambda_pdr * pdr(pred_z, states[:, PAD_LEN:, :])
                loss += prop_loss
                epoch_prop_loss += prop_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss / len(train_loader))

        if epoch == 0 or (epoch + 1) % 50 == 0:
            msg = f"Epoch [{epoch + 1}/{args.epochs}], total={losses[-1]:.6f}, mse={epoch_mse_loss / len(train_loader):.6f}, pdr={epoch_prop_loss / len(train_loader):.6f}"
            print(msg)
            
        if losses[-1] < args.target_loss:
            print(f"Reached target loss {args.target_loss} at epoch {epoch + 1}. Stopping early.")
            break

    # -------------------------------------------------------------------
    # Save training loss plot
    # -------------------------------------------------------------------
    os.makedirs("main/results", exist_ok=True)

    plt.figure()
    plt.plot(losses, label="total")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"{args.task} Training Loss ({args.conditioning_mode})")
    plt.legend()
    plt.tight_layout()

    loss_path = f"main/results/{args.task}_{args.conditioning_mode}_loss.png"
    plt.savefig(loss_path)
    plt.close()

    # Check proportionality
    if args.conditioning_mode == "z":
        alpha_value = torch.exp(pdr.log_alpha).item()
        print(f"Learned proportionality scale (alpha): {alpha_value:.4f}")

    # SAVING ---------------------------------------------------------------

    z_min = None
    z_max = None

    if args.conditioning_mode == "z":
        # find style limits and compare with actual heights
        with torch.inference_mode():
            z_values = []

            test_loader = DataLoader(
                dataset=train_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )

            for i, (images, tactile, states, actions) in enumerate(test_loader):
                images = images.to(device)
                tactile = tactile.to(device)
                states = states.to(device)
                actions = actions.to(device)

                # infer z from trajectory using trained model
                _, z_val = model(
                    images,
                    tactile,
                    states,
                    z_style=None,
                    actions=actions,
                )
                z_values.append(z_val.cpu().numpy().flatten()[0])

            # learned z range from training data
            z_min = float(np.min(z_values))
            z_max = float(np.max(z_values))
            print(f"Learned z range: [{z_min:.4f}, {z_max:.4f}]")

            # save z mapping for analysis
            z_mapping = {
                "styles": actual_styles,
                "z_values": z_values,
            }
            if args.lambda_pdr > 0:
                mapping_path = f"main/learned/tacstyle_{args.task}_pdr_z_mapping.pkl"
            else:
                mapping_path = f"main/learned/tacstyle_{args.task}_z_mapping.pkl"
            os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
            with open(mapping_path, "wb") as f:
                pickle.dump(z_mapping, f)

            print(f"Saved z_value mapping to {mapping_path}")

    # -------------------------------------------------------------------
    # Save trained model checkpoint
    # -------------------------------------------------------------------
    if args.conditioning_mode == "language":
        save_path = f"main/learned/tacstyle_{args.task}_language_{num_labels}labels_model.pt"
    else:
        if args.lambda_pdr > 0:
            save_path = f"main/learned/tacstyle_{args.task}_pdr_model.pt"
        else:
            save_path = f"main/learned/tacstyle_{args.task}_model.pt"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": cfg.__dict__,
            "conditioning_mode": args.conditioning_mode,

            # normalization stats
            "joints_mu": joints_mu,
            "joints_std": joints_std,
            "tcp_mu": tcp_mu,
            "tcp_std": tcp_std,
            "tactile_mu": tactile_mu,
            "tactile_std": tactile_std,

            # learned z range
            "z_min": z_min,
            "z_max": z_max,
            "pad_len": PAD_LEN,
            "pdr_alpha": torch.exp(pdr.log_alpha).item() if args.conditioning_mode == "z" else None,

            # language baseline metadata
            "language_texts": language_texts,
            "qwen_model_name": cfg.qwen_model_name,

        },
        save_path,
    )

    print(f"Saved model to {save_path}")
    print("Done.")