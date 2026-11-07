"""UR5e 자산 경로·관절·frame 상수"""

from pathlib import Path

_XMLS = Path(__file__).parent / "xmls"

# position actuator 장면, 속도 IK·QP 데모
UR5E_SCENE_XML = _XMLS / "scene.xml"
# motor actuator 장면, torque 데모
UR5E_TORQUE_SCENE_XML = _XMLS / "scene_torque.xml"
UR5E_URDF = _XMLS / "urdf" / "ur5e.urdf"

UR5E_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
UR5E_FRAME_NAMES = {"ee": "tool0"}
