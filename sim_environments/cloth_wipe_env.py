import sys
import os
import math
import time
import pickle

import argparse
import re, tempfile
import numpy as np

import pybullet as p
import pybullet_data

from tactile_gym.tactile_sensor import TactileSensor

SDK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'python-sdk')
sys.path.insert(0, os.path.abspath(SDK_DIR))

from utils import patch_urdf, actuate_gripper


def move_to_pose(p, robot_id, arm_joints, grasp_index, target_pos, target_orn, step_size=0.005, ik_steps=None,
                 record=False, tactile_sensor=None, gripper_joints=None, save_mean_tactile=True, cam_args=None):
    
    start_state = p.getLinkState(robot_id, grasp_index)
    start_pos = np.array(start_state[0])
    start_orn = start_state[1]
    target_pos = np.array(target_pos)
    
    # calculate number of steps based on step_size
    dist = np.linalg.norm(target_pos - start_pos)
    num_steps = max(10, int(dist / step_size))
        
    if ik_steps is None:
        ik_steps = num_steps
    
    # pre-calculate IK waypoints
    theta_goals = []
    for i in range(ik_steps):
        alpha = (i + 1) / float(ik_steps)
        interp_pos = start_pos + alpha * (target_pos - start_pos)
        interp_orn = p.getQuaternionSlerp(start_orn, target_orn, alpha)
        
        theta = p.calculateInverseKinematics(robot_id, 
                                             grasp_index, 
                                             interp_pos.tolist(),
                                             interp_orn,
                                             solver=p.IK_DLS, 
                                             maxNumIterations=1000, 
                                             residualThreshold=1e-5)
        theta_goals.append(theta)
        
    traj_data = []

    # read closed gripper position once before moving
    fixed_gripper_q = None
    if gripper_joints is not None:
        initial_knuckle = p.getJointState(robot_id, gripper_joints["gripper_finger1_inner_knuckle_joint"])[0]
        fixed_gripper_q = initial_knuckle / 1.49462955

    # execute physics steps, mapping them evenly to the waypoints
    for step in range(num_steps):
        waypoint_idx = int((step / num_steps) * ik_steps)
        waypoint_idx = min(waypoint_idx, ik_steps - 1)
        p.setJointMotorControlArray(robot_id, arm_joints, p.POSITION_CONTROL,
                                    targetPositions=theta_goals[waypoint_idx][:len(arm_joints)])

        if gripper_joints is not None:
            # Command the gripper to the fixed grasp position
            actuate_gripper(robot_id, gripper_joints, q=fixed_gripper_q, force=100)

        p.stepSimulation()
        time.sleep(1.0/120.0)
        
        if record:
            
            joint_angles = [p.getJointState(robot_id, j)[0] for j in arm_joints]
            gripper_q = p.getJointState(robot_id, gripper_joints["gripper_finger1_inner_knuckle_joint"])[0]
            tactile_imgs = tactile_sensor.get_imgs()
            
            if save_mean_tactile:
                tactile_data = [np.mean(img) for img in tactile_imgs]
            else:
                tactile_data = tactile_imgs
                
            step_data = {
                "time_step": step,
                "joint_angles": joint_angles,
                "gripper_q": gripper_q,
                "tactile_data": tactile_data,
                "tcp": list(p.getLinkState(robot_id, grasp_index)[0])
            }
            
            if cam_args is not None:
                width, height, view_mat, proj_mat = cam_args
                _, _, rgbImg, _, _ = p.getCameraImage(width, height, viewMatrix=view_mat, projectionMatrix=proj_mat)
                step_data["cam_image"] = np.reshape(rgbImg, (height, width, 4)).astype(np.uint8)[:, :, :3]
                
            traj_data.append(step_data)
            
    return traj_data


def anchor_cloth_to_gripper(p, cloth_id, robot_id, link_index, threshold=0.05):
    try:
        data = p.getMeshData(cloth_id, -1, flags=1)
        if len(data) >= 2:
            vertices = data[1]
            link_state = p.getLinkState(robot_id, link_index)
            link_pos = link_state[0]
            anchors_created = 0
            for i, v in enumerate(vertices):
                dist = sum((a - b)**2 for a, b in zip(v, link_pos))**0.5
                if dist < threshold:
                    p.createSoftBodyAnchor(cloth_id, i, robot_id, link_index)
                    anchors_created += 1
            print(f"Created {anchors_created} soft body anchors to prevent slipping.")
    except Exception as e:
        print("Could not create soft body anchors:", e)


def run_sim(num_demos=6):

    HOME_DEG = [0.0, -70.0, 90.0, -110.0, -90.0, 30.0]

    urdf_path = os.path.join(SDK_DIR, 'urdf', 'fairino5_v6_with_ag95_and_digit.urdf')
    if not os.path.exists(urdf_path):
        print(f"ERROR: URDF not found at {urdf_path}")
        sys.exit(1)

    patched_urdf = patch_urdf(urdf_path)

    print("Starting PyBullet simulation...")
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Ensure data directory exists
    os.makedirs(os.path.join('main', 'data_cloth_wipe_sim'), exist_ok=True)
    
    # Wiping height preferences
    base_heights = [0.05, 0.075, 0.10, 0.125, 0.150]
    
    for b_idx, base_height in enumerate(base_heights):
        for demo in range(1, num_demos + 1):
            print(f"\n=== Starting demo {demo}/{num_demos} for base_height {base_height} ===")
            
            p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
            p.setGravity(0, 0, -9.81)
            p.resetDebugVisualizerCamera(1., -15., -15., [0.55, 0.27, 0.9])
            p.loadURDF("plane.urdf")

            p.setPhysicsEngineParameter(
                fixedTimeStep=1.0/1000.0,
                numSolverIterations=200,
                numSubSteps=4
            )

            # Add table to the environment
            TABLE_LENGTH = 1.8288  # meters
            TABLE_WIDTH = 0.9144  # meters
            TABLE_HEIGHT = 0.8636  # meters
            table_half = [TABLE_LENGTH / 2, TABLE_WIDTH / 2, TABLE_HEIGHT / 2]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=table_half,
                                      rgbaColor=[0.85, 0.75, 0.55, 1.0])
            p.createMultiBody(0, col, vis, basePosition=[0.0, 0.0, TABLE_HEIGHT / 2])

            # Add robot to the environment
            robot_base_pos = [TABLE_LENGTH / 2 - 0.1, 0.0, TABLE_HEIGHT + 0.001]
            robot_base_orn = p.getQuaternionFromEuler([0.0, 0.0, math.pi])
            robot_id = p.loadURDF(patched_urdf, basePosition=robot_base_pos,
                                  baseOrientation=robot_base_orn, useFixedBase=True)

            # Find arm and gripper joints
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
                p,
                robot_id=robot_id,
                tactile_link_ids=tactile_link_ids,
                image_size=[128, 128],
                turn_off_border=False,
                t_s_name="digit",
                t_s_type="standard",
                t_s_core="no_core",
                t_s_dynamics={},
                show_tactile=False,
                t_s_num=1,
                assets_dir=SDK_DIR,
            )
            # tactile_sensor.save_reference_images()

            # Set cloth orientation and initial position
            CLOTH_X = 0.10
            CLOTH_Y = 0.0
            cloth_euler = (math.pi/2, 0.0, math.pi/2)   # sideways orientation
            cloth_quat = p.getQuaternionFromEuler(cloth_euler)

            # Set gripper orientation (facing down)
            ee_euler = (math.pi, 0.0, 0.0)
            ee_quat = p.getQuaternionFromEuler(ee_euler)
            
            # Reset robot to home position
            for idx, jid in enumerate(arm_joints[:6]):
                target_rad = math.radians(HOME_DEG[idx])
                p.resetJointState(robot_id, jid, target_rad)
                p.setJointMotorControl2(robot_id, jid, p.POSITION_CONTROL, targetPosition=target_rad)
                
            # Open gripper
            for j in gripper_joints.values():
                p.resetJointState(robot_id, j, 0.0)
                p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=0.0)
                
            # Constant cloth height, varying wiping height
            height_noise = np.random.uniform(-0.01, 0.01)
            CLOTH_Z = TABLE_HEIGHT + 0.15
            WIPE_HEIGHT = TABLE_HEIGHT + base_height + height_noise

            # Add objects (cloth) draped over the core at the starting height
            clothId = p.loadSoftBody("cloth_z_up.obj", 
                                     basePosition = [CLOTH_X, CLOTH_Y, CLOTH_Z], 
                                     baseOrientation = cloth_quat,
                                     scale = 0.07, mass = 0.1, 
                                     useNeoHookean = 0, useBendingSprings=1, useMassSpring=1, 
                                     springElasticStiffness=40, springDampingStiffness=.1, springDampingAllDirections = 1, 
                                     useSelfCollision = 0, frictionCoeff = 1.0, useFaceContact=1,
                                     collisionMargin=0.001)
            p.changeVisualShape(clothId, -1, 
                                rgbaColor=[1.0, 0., 0., 1.0], textureUniqueId=-1, 
                                flags=p.VISUAL_SHAPE_DOUBLE_SIDED)    

            # Let the cloth settle
            print("Letting cloth settle...")
            for _ in range(50):
                p.stepSimulation()

            # Phase 1: Approach & Grasp
            print("Phase 1: Approach & Grasp ...")
            move_to_pose(p, robot_id, arm_joints, grasp_index, [CLOTH_X, CLOTH_Y, CLOTH_Z], ee_quat, ik_steps=1)
            move_to_pose(p, robot_id, arm_joints, grasp_index, [CLOTH_X, CLOTH_Y, CLOTH_Z - 0.07], ee_quat, ik_steps=1)
            
            # Close gripper
            print("Grasping cloth ...")
            for i in range(100):
                q = 0.5 * (i / 100.0)
                actuate_gripper(robot_id, gripper_joints, q=q, force=100)
                p.stepSimulation()
                time.sleep(1.0/120.0) # sleep less to speed up data coll
                
            # Explicitly anchor the soft body to the gripper to prevent any physics slippage
            anchor_cloth_to_gripper(p, clothId, robot_id, grasp_index, threshold=0.05)
                
            time.sleep(0.5)

            # Setup camera for recording using the current view from the PyBullet GUI
            cam_info = p.getDebugVisualizerCamera()
            cam_view_matrix = cam_info[2]
            cam_proj_matrix = cam_info[3]
            # Capture at a smaller resolution that matches the visualizer's aspect ratio
            cam_args = (320, 180, cam_view_matrix, cam_proj_matrix)

            # Phase 3: Initialize style
            print("Phase 3: Wipe Preparation (RECORDING) ...")
            traj_data_p3 = move_to_pose(p, robot_id, arm_joints, grasp_index, [CLOTH_X, CLOTH_Y, WIPE_HEIGHT], ee_quat,
                                        record=True, tactile_sensor=tactile_sensor, gripper_joints=gripper_joints, cam_args=cam_args)

            # Phase 4: Wiping motion
            print("Phase 4: Wiping Motion (RECORDING) ...")
            traj_data_p4 = move_to_pose(p, robot_id, arm_joints, grasp_index, [CLOTH_X + 0.25, CLOTH_Y, WIPE_HEIGHT], ee_quat,
                                        record=True, tactile_sensor=tactile_sensor, gripper_joints=gripper_joints, cam_args=cam_args)

            traj_data_combined = traj_data_p3 + traj_data_p4

            # Separate images from numerical data to keep pickle files manageable
            traj_only = []
            cam_images = []
            for step_data in traj_data_combined:
                cam_image = step_data.pop("cam_image", None)
                if cam_image is not None:
                    cam_images.append(cam_image)
                traj_only.append(step_data)

            # Save the trajectory data
            save_name = f"cloth_wipe_env_h{base_height:.3f}_d{demo}.pkl"
            save_path = os.path.join("main", "data_cloth_wipe_sim", save_name)
            with open(save_path, "wb") as f:
                pickle.dump(traj_only, f)
            print(f"Saved trajectory data to {save_path}!")
            
            # Save the camera images separately
            img_save_name = f"cloth_wipe_env_h{base_height:.3f}_d{demo}_images.pkl"
            img_save_path = os.path.join("main", "data_cloth_wipe_sim", img_save_name)
            with open(img_save_path, "wb") as f:
                pickle.dump(cam_images, f)
            print(f"Saved camera images to {img_save_path}!")

    print("Data collection completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="tactile cloth wipe example")
    parser.add_argument("--real", action="store_true", help="Connect to real robot")
    parser.add_argument("--ip", default="192.168.58.2", help="Robot IP (real only)")
    parser.add_argument("--num_demos", type=int, default=6, help="Number of demos per height")
    args = parser.parse_args()

    if args.real:
        print("Real not yet implemented.")
    else:
        run_sim(args.num_demos)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------
# Environment Setup for run_robot_sim.py
# ---------------------------------------------------------------------

def setup_cloth_wipe_sim(p, record):
    p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
    p.setGravity(0, 0, -9.81)
    p.resetDebugVisualizerCamera(1.0, -15.0, -15.0, [0.55, 0.27, 0.9])
    p.loadURDF("plane.urdf")

    p.setPhysicsEngineParameter(
        fixedTimeStep=1.0 / 1000.0,
        numSolverIterations=200,
        numSubSteps=4,
    )

    TABLE_HEIGHT = 0.8636
    TABLE_HALF = [1.8288 / 2, 0.9144 / 2, TABLE_HEIGHT / 2]

    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=TABLE_HALF)
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=TABLE_HALF,
        rgbaColor=[0.85, 0.75, 0.55, 1.0],
    )
    p.createMultiBody(0, col, vis, basePosition=[0.0, 0.0, TABLE_HEIGHT / 2])

    urdf_path = os.path.join(SDK_DIR, "urdf", "fairino5_v6_with_ag95_and_digit.urdf")
    patched_urdf = patch_urdf(urdf_path)

    robot_base_pos = [1.8288 / 2 - 0.1, 0.0, TABLE_HEIGHT + 0.001]
    robot_base_orn = p.getQuaternionFromEuler([0.0, 0.0, math.pi])

    robot_id = p.loadURDF(
        patched_urdf,
        basePosition=robot_base_pos,
        baseOrientation=robot_base_orn,
        useFixedBase=True,
    )

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

    tactile_sensor = TactileSensor(
        p,
        robot_id=robot_id,
        tactile_link_ids={
            "body": link_name_to_index["fairino_digit_body_link"],
            "tip": link_name_to_index["fairino_digit_tip_link"],
        },
        image_size=[128, 128],
        show_tactile=not record,
        t_s_num=1,
        assets_dir=SDK_DIR,
        t_s_name="digit",
    )

    CLOTH_X, CLOTH_Y, CLOTH_Z = 0.10, 0.0, TABLE_HEIGHT + 0.15
    cloth_quat = p.getQuaternionFromEuler((math.pi / 2, 0.0, math.pi / 2))

    cloth_id = p.loadSoftBody(
        "cloth_z_up.obj",
        basePosition=[CLOTH_X, CLOTH_Y, CLOTH_Z],
        baseOrientation=cloth_quat,
        scale=0.07,
        mass=0.1,
        useNeoHookean=0,
        useBendingSprings=1,
        useMassSpring=1,
        springElasticStiffness=40,
        springDampingStiffness=0.1,
        springDampingAllDirections=1,
        useSelfCollision=0,
        frictionCoeff=1.0,
        useFaceContact=1,
        collisionMargin=0.001,
    )

    p.changeVisualShape(
        cloth_id,
        -1,
        rgbaColor=[1.0, 0.0, 0.0, 1.0],
        textureUniqueId=-1,
        flags=p.VISUAL_SHAPE_DOUBLE_SIDED,
    )

    HOME_DEG = [0.0, -70.0, 90.0, -110.0, -90.0, 30.0]
    for idx, jid in enumerate(arm_joints[:6]):
        target_rad = math.radians(HOME_DEG[idx])
        p.resetJointState(robot_id, jid, target_rad)
        p.setJointMotorControl2(robot_id, jid, p.POSITION_CONTROL, targetPosition=target_rad)

    for j in gripper_joints.values():
        p.resetJointState(robot_id, j, 0.0)
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=0.0)

    for _ in range(50):
        p.stepSimulation()
        if not record:
            time.sleep(1.0 / 120.0)

    ee_quat = p.getQuaternionFromEuler((math.pi, 0.0, 0.0))

    move_to_pose(
        p, robot_id, arm_joints, grasp_index, [CLOTH_X, CLOTH_Y, CLOTH_Z], ee_quat, ik_steps=1
    )
    move_to_pose(
        p, robot_id, arm_joints, grasp_index, [CLOTH_X, CLOTH_Y, CLOTH_Z - 0.07], ee_quat, ik_steps=1
    )

    for i in range(100):
        actuate_gripper(robot_id, gripper_joints, q=0.5 * (i / 100.0), force=100)
        p.stepSimulation()
        if not record:
            time.sleep(1.0 / 120.0)

    anchor_cloth_to_gripper(p, cloth_id, robot_id, grasp_index, threshold=0.05)

    cam_info = p.getDebugVisualizerCamera()
    return robot_id, arm_joints, gripper_joints, grasp_index, tactile_sensor, cam_info[2], cam_info[3]


