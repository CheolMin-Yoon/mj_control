"""Neuromeka Indy7 v3 자산 경로·관절·frame 상수"""

from pathlib import Path

_XMLS = Path(__file__).parent / "xmls"

# 6 motor(gear 1, ctrl N·m), 관절 damping 0.1·frictionloss 10, timestep 2 ms
INDY7_V3_SCENE_XML = _XMLS / "scene.xml"
INDY7_V3_URDF = _XMLS / "indy7_v3.urdf"

INDY7_V3_JOINT_NAMES = ("joint0", "joint1", "joint2", "joint3", "joint4", "joint5")
# Pinocchio frame·MuJoCo site 모두 tcp
INDY7_V3_FRAME_NAMES = {"ee": "tcp"}
