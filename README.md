# TacStyle: Personalizing Tactile Robot Policies using Structured Behavior Representations

---

## 🛠 Installation

Install the required dependencies using pip:
```bash
pip install torch numpy pybullet openai transformers
```

*Note: The code uses OpenAI's API, remember to export your API key:*
```bash
export OPENAI_API_KEY="your_api_key_here"
```

---

## 🚀 Full Pipeline Workflow

This repository supports two main tasks: `cloth_wipe_sim` and `cloth_fold_sim`. 
Below are the sequential steps to generate data, process it, train your policy, and run interactive simulations.

### 1. Generate Demonstration Data
Run the PyBullet simulation environment to generate synthetic expert demonstrations.
```bash
# For wiping:
python sim_environments/cloth_wipe_env.py

# For folding:
python sim_environments/cloth_fold_env.py --num_demos 5
```
*(Data is saved to `main/data_cloth_wipe_sim` or `main/data_cloth_fold_sim`)*

### 2. Preprocess Data
Format the generated demonstrations and calculate normalization statistics.
```bash
python main/preprocess_data.py --task cloth_wipe_sim
# or --task cloth_fold_sim
```

### 3. Train the Model
Train the TacStyle transformer policy. You can choose to train a latent style `z` conditionined model (Ours) or a `language` conditioned model (Baseline).

```bash
# Train Ours:
python main/train_model.py --task cloth_wipe_sim --epochs 4000  --conditioning_mode z --lamba_pdr 0.1

# Train Discrete Language Baseline:
python main/train_model.py --task cloth_wipe_sim --epochs 4000 --conditioning_mode language
```
*(Checkpoints are saved to `main/learned/`)*

### 4. Interactive Robot Simulation
Test your learned policies interactively. Provide natural language instructions and watch the robot execute the behaviors in real-time.

```bash
# Run Continuous TacStyle Model:
python main/run_robot_sim.py --task cloth_wipe_sim --conditioning_mode z

# Run Discrete Language Baseline (Specify number of labels used during training):
python main/run_robot_sim.py --task cloth_wipe_sim --conditioning_mode language --num_labels 3
```

---

## 🧠 File Structure Overview

* `sim_environments/`: Contains the PyBullet setups for generating the cloth wiping and folding demonstrations.
* `main/preprocess_data.py`: Prepares the data, resizes images, and computes normalization stats.
* `main/train_model.py`: Trains the TacStyle (Ours) and Baseline policies.
* `main/models.py`: Defines TacStyle (Ours) and Baseline architectures.
* `main/language_api.py`: Connects Natural Language requests to continuous latent `z` values.
* `main/run_robot_sim.py`: The unified evaluation script to visually test policies in PyBullet.