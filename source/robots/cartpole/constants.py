"""Cartpole 자산 경로·관절 상수"""

from pathlib import Path

_XMLS = Path(__file__).parent / "xmls"

# motor gear 50, timestep 0.01 s
CARTPOLE_SCENE_XML = _XMLS / "scene.xml"
CARTPOLE_ACTUATED_JOINT_NAMES = ("slider",)
# pole geom fromto='0 0 0 0 0 0.6'
CARTPOLE_POLE_LENGTH = 0.6
