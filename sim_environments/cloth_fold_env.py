import sys
import os
import math
import pickle
import glob
import numpy as np
import time
import argparse

import pybullet as p
import pybullet_data

SDK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'python-sdk')
sys.path.insert(0, os.path.abspath(SDK_DIR))

from utils import patch_urdf, actuate_gripper
from tactile_gym.tactile_sensor import TactileSensor


def add_trajectory_noise(pos, scale=0.002):
    return pos + np.random.normal(0, scale, size=3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_demos', type=int, default=5, help='Number of demos to generate per style')
    args = parser.parse_args()

    save_dir = os.path.join("main", "data_cloth_fold_sim")
    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Hardcoded data for average grasp points and initial states
    # ---------------------------------------------------------
    initial_joints_deg = [0.000, -60.000, -100.000, -110.000, 90.000, 0.000]
    
    # Reference robot TCP start position
    real_tcp_start = np.array([0.261, -0.102, 0.555, -180.000, 0.000, -90.000])
    
    # Target TCP positions for each folding style
    avg_tcps = {
        "loose": np.array([0.249, 0.254, 0.269, 179.946, 0.017, -7.667]),
        "mid": np.array([0.231, 0.193, 0.266, 179.949, 0.017, -7.668]),
        "tight": np.array([0.231, 0.111, 0.268, 179.953, 0.013, -7.672])
    }

    # ---------------------------------------------------------
    # 2. Setup PyBullet Simulation
    # ---------------------------------------------------------
    urdf_path = os.path.join(SDK_DIR, 'urdf', 'fairino5_v6_with_ag95_and_digit.urdf')
    patched_urdf = patch_urdf(urdf_path)

    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    TABLE_LENGTH = 1.8288
    TABLE_WIDTH = 0.9144
    TABLE_HEIGHT = 0.8636

    for style in ["loose", "mid", "tight"]:
        if style not in avg_tcps:
            continue
            
        target_real_tcp = avg_tcps[style]

        for demo_idx in range(1, args.num_demos + 1):
            print(f"\n=== Generating Demo {demo_idx}/{args.num_demos} for Style '{style}' ===")
            
            num_steps = 75
            real_demo_data = {
                "timestamp": np.linspace(0, 5, num_steps),
                "tactile": np.zeros((num_steps, 15), dtype=np.float32),
                "gripper": np.full((num_steps, 2), 100.0, dtype=np.float32)
            }

            p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
            p.setGravity(0, 0, -9.81)
            p.resetDebugVisualizerCamera(1.5, 45.0, -30.0, [0.5, 0.0, 0.5])
            p.loadURDF("plane.urdf")

            p.setPhysicsEngineParameter(fixedTimeStep=1.0/1000.0, numSolverIterations=200, numSubSteps=4)

            # Table
            table_half = [TABLE_LENGTH / 2, TABLE_WIDTH / 2, TABLE_HEIGHT / 2]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=table_half, rgbaColor=[0.85, 0.75, 0.55, 1.0])
            p.createMultiBody(0, col, vis, basePosition=[0.0, 0.0, TABLE_HEIGHT / 2])

            # Robot
            robot_base_pos = [TABLE_LENGTH / 2 - 0.1, 0.0, TABLE_HEIGHT + 0.001]
            robot_base_orn = p.getQuaternionFromEuler([0.0, 0.0, math.pi])
            robot_id = p.loadURDF(patched_urdf, basePosition=robot_base_pos, baseOrientation=robot_base_orn, useFixedBase=True)

            arm_joints = []
            gripper_joints = {}
            link_name_to_index = {}
            grasp_index = None

            for j in range(p.getNumJoints(robot_id)):
                info = p.getJointInfo(robot_id, j)
                joint_name = info[1].decode("utf-8")
                link_name = info[12].decode("utf-8")
                if info[2] == p.JOINT_REVOLUTE and 'gripper' not in joint_name:
                    arm_joints.append(j)
                elif info[2] == p.JOINT_REVOLUTE and 'gripper' in joint_name:
                    gripper_joints[joint_name] = j
                elif 'grasp_frame' in joint_name:
                    grasp_index = j
                link_name_to_index[link_name] = j
                
            if grasp_index is None:
                grasp_index = arm_joints[-1]

            tactile_link_ids = {
                "body": link_name_to_index["fairino_digit_body_link"],
                "tip": link_name_to_index["fairino_digit_tip_link"],
            }

            tactile_sensor = TactileSensor(
                p, robot_id=robot_id, tactile_link_ids=tactile_link_ids,
                image_size=[128, 128], turn_off_border=False,
                t_s_name="digit", t_s_type="standard", t_s_core="no_core",
                t_s_dynamics={}, show_tactile=False, t_s_num=1, assets_dir=SDK_DIR
            )

            # Initialize joints to start pose
            for idx, jid in enumerate(arm_joints[:6]):
                target_rad = math.radians(initial_joints_deg[idx])
                p.resetJointState(robot_id, jid, target_rad)
                p.setJointMotorControl2(robot_id, jid, p.POSITION_CONTROL, targetPosition=target_rad)

            # Start with gripper open
            initial_knuckle = p.getJointState(robot_id, gripper_joints["gripper_finger1_inner_knuckle_joint"])[0]
            fixed_gripper_q = 0.0 # Actuate it open
            for i in range(50):
                actuate_gripper(robot_id, gripper_joints, q=fixed_gripper_q, force=100)
                p.stepSimulation()


            cam_info = p.getDebugVisualizerCamera()
            cam_view_matrix, cam_proj_matrix = cam_info[2], cam_info[3]
            cam_args = (320, 180, cam_view_matrix, cam_proj_matrix)

            movable_joints = [j for j in range(p.getNumJoints(robot_id)) if p.getJointInfo(robot_id, j)[2] != p.JOINT_FIXED]
            
            # Extract limits for null-space IK
            ll = []
            ul = []
            jr = []
            rp = []
            for j in movable_joints:
                info = p.getJointInfo(robot_id, j)
                lower = info[8]
                upper = info[9]
                ll.append(lower)
                ul.append(upper)
                jr.append(upper - lower)
                rp.append(p.getJointState(robot_id, j)[0])

            traj_data = []
            
            def record_step(step_idx):
                p.stepSimulation()
                joint_angles = [p.getJointState(robot_id, j)[0] for j in arm_joints[:6]]
                joint_angles_deg = [math.degrees(j) for j in joint_angles]
                
                # Get wrist link world pose
                wrist_pos, wrist_orn = p.getLinkState(robot_id, arm_joints[5])[:2]
                
                # Convert to robot base frame
                inv_base_pos, inv_base_orn = p.invertTransform(robot_base_pos, robot_base_orn)
                rel_pos, rel_orn = p.multiplyTransforms(inv_base_pos, inv_base_orn, wrist_pos, wrist_orn)
                
                # Format to mm and deg
                rel_pos_mm = [p_val * 1000.0 for p_val in rel_pos]
                rel_orn_deg = [math.degrees(e) for e in p.getEulerFromQuaternion(rel_orn)]
                tcp_final = rel_pos_mm + rel_orn_deg
                
                traj_data.append({
                    "timestamp": real_demo_data["timestamp"][step_idx],
                    "joints": joint_angles_deg,
                    "tcp": tcp_final,
                    "tactile": real_demo_data["tactile"][step_idx],
                    "gripper": real_demo_data["gripper"][step_idx]
                })

            start_state = p.getLinkState(robot_id, grasp_index)
            start_pos = np.array(start_state[0])
            start_orn = start_state[1] # quaternion

            # PHASE 1: Rotate gripper around Z-axis by 90 degrees
            start_euler = p.getEulerFromQuaternion(start_orn)
            target_rz = start_euler[2] + math.radians(90.0)
            target_euler_rot = (start_euler[0], start_euler[1], target_rz)
            target_orn_rot = p.getQuaternionFromEuler(target_euler_rot)
            
            steps_rot = 25
            for i in range(steps_rot):
                alpha = (i + 1) / float(steps_rot)
                interp_orn = p.getQuaternionSlerp(start_orn, target_orn_rot, alpha)
                theta = p.calculateInverseKinematics(
                    robot_id, grasp_index, start_pos.tolist(), interp_orn,
                    lowerLimits=ll, upperLimits=ul, jointRanges=jr, restPoses=rp,
                    solver=p.IK_DLS, maxNumIterations=100
                )
                p.setJointMotorControlArray(robot_id, arm_joints[:6], p.POSITION_CONTROL, targetPositions=theta[:6])
                actuate_gripper(robot_id, gripper_joints, q=fixed_gripper_q, force=100)
                record_step(i)
                time.sleep(0.03)

            # PHASE 2: Move to Target TCP
            real_delta = target_real_tcp[:3] - real_tcp_start[:3]
            
            # Since robot base is at [0,0,pi], local delta X corresponds to -X in world, local Y to -Y in world
            world_delta = np.array([-real_delta[0], -real_delta[1], real_delta[2]])
            
            # Add noise to target position
            noise_scale = 0.005 # 5mm noise
            world_delta_noisy = world_delta + np.random.normal(0, noise_scale, size=3)
            
            target_pos = start_pos + world_delta_noisy
            start_orn_phase2 = p.getLinkState(robot_id, grasp_index)[1]
            
            steps_move = 50
            for i in range(steps_move):
                alpha = (i + 1) / float(steps_move)
                interp_pos = start_pos + alpha * (target_pos - start_pos)
                
                # Add trajectory noise
                noise_magnitude = math.sin(alpha * math.pi) * 0.003
                interp_pos += np.random.normal(0, noise_magnitude, size=3)
                
                theta = p.calculateInverseKinematics(
                    robot_id, grasp_index, interp_pos.tolist(), start_orn_phase2,
                    lowerLimits=ll, upperLimits=ul, jointRanges=jr, restPoses=rp,
                    solver=p.IK_DLS, maxNumIterations=100
                )
                p.setJointMotorControlArray(robot_id, arm_joints[:6], p.POSITION_CONTROL, targetPositions=theta[:6])
                actuate_gripper(robot_id, gripper_joints, q=fixed_gripper_q, force=100)
                record_step(steps_rot + i)
                time.sleep(0.03)

            time.sleep(0.3)

            traj_dict = {
                "timestamp": [step["timestamp"] for step in traj_data],
                "joints": [step["joints"] for step in traj_data],
                "tcp": [step["tcp"] for step in traj_data],
                "tactile": [step["tactile"] for step in traj_data],
                "gripper": [step["gripper"] for step in traj_data]
            }
            
            final_sim_tcp = traj_data[-1]["tcp"]

            save_name = f"trial_{demo_idx}_{style}.pkl"
            save_path = os.path.join(save_dir, save_name)
            with open(save_path, "wb") as f: pickle.dump(traj_dict, f)
            print(f"Saved {save_path}")

    p.disconnect()
    print("Generation complete!")

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------
# Environment Setup for run_robot_sim.py
# ---------------------------------------------------------------------

def setup_cloth_fold_env(p, record):
    p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
    p.setGravity(0, 0, -9.81)
    p.resetDebugVisualizerCamera(1.5, 45.0, -30.0, [0.5, 0.0, 0.5])
    p.loadURDF("plane.urdf")

    p.setPhysicsEngineParameter(fixedTimeStep=1.0/1000.0, numSolverIterations=200, numSubSteps=4)

    # Table setup
    TABLE_HEIGHT = 0.8636
    TABLE_LENGTH = 1.8288
    TABLE_WIDTH = 0.9144
    table_half = [TABLE_LENGTH / 2, TABLE_WIDTH / 2, TABLE_HEIGHT / 2]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=table_half, rgbaColor=[0.85, 0.75, 0.55, 1.0])
    p.createMultiBody(0, col, vis, basePosition=[0.0, 0.0, TABLE_HEIGHT / 2])

    # Robot setup
    robot_base_pos = [TABLE_LENGTH / 2 - 0.1, 0.0, TABLE_HEIGHT + 0.001]
    robot_base_orn = p.getQuaternionFromEuler([0.0, 0.0, math.pi])
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sdk_dir = os.path.join(current_dir, 'python-sdk')
    urdf_path = os.path.join(sdk_dir, 'urdf', 'fairino5_v6_with_ag95_and_digit.urdf')
    patched_urdf = patch_urdf(urdf_path)

    robot_id = p.loadURDF(patched_urdf, basePosition=robot_base_pos, baseOrientation=robot_base_orn, useFixedBase=True)

    arm_joints = []
    gripper_joints = {}
    link_name_to_index = {}
    grasp_index = None

    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        name = info[1].decode("utf-8")
        link = info[12].decode("utf-8")
        if info[2] == p.JOINT_REVOLUTE and "gripper" not in name:
            arm_joints.append(j)
        elif info[2] == p.JOINT_REVOLUTE and "gripper" in name:
            gripper_joints[name] = j
        elif "grasp_frame" in name:
            grasp_index = j
        link_name_to_index[link] = j

    if grasp_index is None:
        grasp_index = arm_joints[-1]

    tactile_sensor = None

    # Move robot to home fold start position
    HOME_DEG = [0.0, -60.0, -100.0, -110.0, 90.0, 0.0]
    import math
    for idx, jid in enumerate(arm_joints[:6]):
        target_rad = math.radians(HOME_DEG[idx])
        p.resetJointState(robot_id, jid, target_rad)
        p.setJointMotorControl2(robot_id, jid, p.POSITION_CONTROL, targetPosition=target_rad)

    # Open gripper fully
    for j in gripper_joints.values():
        p.resetJointState(robot_id, j, 0.0)
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=0.0)

    for _ in range(50):
        actuate_gripper(robot_id, gripper_joints, q=0.0, force=100)
        p.stepSimulation()

    cam_info = p.getDebugVisualizerCamera()
    view_mat, proj_mat = cam_info[2], cam_info[3]

    return robot_id, arm_joints, gripper_joints, grasp_index, tactile_sensor, view_mat, proj_mat


def post_rollout_cloth_fold(p, robot_id, arm_joints, gripper_joints, grasp_index):
    import time
    import numpy as np
    import math

    print("Closing gripper...")
    for _ in range(100):
        actuate_gripper(robot_id, gripper_joints, q=0.0, force=100)
        p.stepSimulation()
        if p.getConnectionInfo()['connectionMethod'] == p.GUI:
            time.sleep(1.0 / 120.0)

    print("Moving to final target TCP coordinates: x: 726.36, y: 171.10, z: 236.24")
    target_real_tcp_mm = [726.36, 171.10, 236.24]
    
    TABLE_HEIGHT = 0.8636
    TABLE_LENGTH = 1.8288
    robot_base_pos = [TABLE_LENGTH / 2 - 0.1, 0.0, TABLE_HEIGHT + 0.001]

    world_target_pos = [
        robot_base_pos[0] - (target_real_tcp_mm[0] / 1000.0),
        robot_base_pos[1] - (target_real_tcp_mm[1] / 1000.0),
        robot_base_pos[2] + (target_real_tcp_mm[2] / 1000.0),
    ]

    current_state = p.getLinkState(robot_id, grasp_index)
    current_orn = current_state[1]
    
    movable_joints = [j for j in range(p.getNumJoints(robot_id)) if p.getJointInfo(robot_id, j)[2] != p.JOINT_FIXED]
    ll, ul, jr, rp = [], [], [], []
    for j in movable_joints:
        info = p.getJointInfo(robot_id, j)
        lower, upper = info[8], info[9]
        if lower >= upper:
            lower, upper = -math.pi * 2, math.pi * 2
        ll.append(lower)
        ul.append(upper)
        jr.append(upper - lower)
        rp.append(p.getJointState(robot_id, j)[0])

    print("Executing Cartesian move (IK)...")
    steps_move = 100
    start_pos = np.array(current_state[0])
    target_pos = np.array(world_target_pos)

    for i in range(steps_move):
        alpha = (i + 1) / float(steps_move)
        interp_pos = start_pos + alpha * (target_pos - start_pos)
        
        theta = p.calculateInverseKinematics(
            robot_id, grasp_index, interp_pos.tolist(), current_orn,
            lowerLimits=ll, upperLimits=ul, jointRanges=jr, restPoses=rp,
            solver=p.IK_DLS, maxNumIterations=100
        )
        
        p.setJointMotorControlArray(robot_id, arm_joints[:6], p.POSITION_CONTROL, targetPositions=theta[:6])
        actuate_gripper(robot_id, gripper_joints, q=0.0, force=100)
        p.stepSimulation()
        if p.getConnectionInfo()['connectionMethod'] == p.GUI:
            time.sleep(0.02)
