"""Franka Panda 자산 경로·관절·frame·그리퍼 상수"""

from pathlib import Path

_XMLS = Path(__file__).parent / "xmls"

# MuJoCo: 팔 7 motor + 손가락 2 tendon 그리퍼
PANDA_SCENE_XML = _XMLS / "scene_torque.xml"
# panda.urdf는 inertial 없음 → 손 없는 MJCF로 Pinocchio 동역학, 손 질량 약 1.9 kg 누락
PANDA_PINOCCHIO_XML = _XMLS / "xml" / "panda_nohand.xml"

PANDA_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")

# grasp: link7 → attachment(+0.107 m) → grasp(+0.103 m), 손가락 사이 TCP
PANDA_PINOCCHIO_FRAME_NAMES = {"ee": "grasp"}
PANDA_MUJOCO_FRAME_NAMES = {"ee": "grasp_site"}

# actuator8 = tendon split, ctrl 255 = tendon 길이 0.04(최대 열림)
PANDA_GRIPPER_ACTUATOR_ID = 7
PANDA_GRIPPER_OPEN_CTRL = 255
