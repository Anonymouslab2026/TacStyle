import os
import re
import tempfile
import pybullet as p

def patch_urdf(urdf_path):
    """Replace ROS package:// paths with absolute filesystem paths."""
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    arm_mesh_dir = os.path.join(urdf_dir, "meshes")
    ag95_mesh_dir = os.path.join(urdf_dir, "meshes/ag95")
    digit_dir = os.path.join(urdf_dir, "digit")
    with open(urdf_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(r"package://fairino_description/meshes/fairino5_v6",
                     arm_mesh_dir.replace("\\", "/"), content)
    content = re.sub(r"package://fr5_description/meshes",
                     arm_mesh_dir.replace("\\", "/"), content)
    content = re.sub(r"package://ag95_meshes",
                     ag95_mesh_dir.replace("\\", "/"), content)
    content = re.sub(r"package://robot_assets/digit",
                     digit_dir.replace("\\", "/"), content)
    fd, path = tempfile.mkstemp(suffix=".urdf", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path

def actuate_gripper(robot_id, gripper_joints, q, force=60):

    q = max(0.0, min(0.65, q))

    for name in ["gripper_finger1_inner_knuckle_joint", "gripper_finger2_inner_knuckle_joint"]:
        p.setJointMotorControl2(
            robot_id,
            gripper_joints[name],
            p.POSITION_CONTROL,
            targetPosition=1.49462955 * q,
            force=force,
        )

    for name in ["gripper_finger1_finger_tip_joint", "gripper_finger2_finger_tip_joint"]:
        p.setJointMotorControl2(
            robot_id,
            gripper_joints[name],
            p.POSITION_CONTROL,
            targetPosition=1.49462955 * q,
            force=force,
        )

    qf1 = p.getJointState(robot_id, gripper_joints["gripper_finger1_inner_knuckle_joint"])[0]
    qf2 = p.getJointState(robot_id, gripper_joints["gripper_finger2_inner_knuckle_joint"])[0]

    for name, qm in [("gripper_finger1_joint", qf1), ("gripper_finger2_joint", qf2)]:
        p.setJointMotorControl2(
            robot_id,
            gripper_joints[name],
            p.POSITION_CONTROL,
            targetPosition=qm / 1.49462955,
            force=force,
        )

    for name, qm in [("gripper_finger1_finger_joint", qf1), ("gripper_finger2_finger_joint", qf2)]:
        p.setJointMotorControl2(
            robot_id,
            gripper_joints[name],
            p.POSITION_CONTROL,
            targetPosition=0.4563942 * qm / 1.49462955,
            force=force,
        )
