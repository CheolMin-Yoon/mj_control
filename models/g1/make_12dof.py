import re
import os

urdf_path = '/home/frlab/mj_opt/models/g1/urdf/g1_29dof.urdf'
xml_path = '/home/frlab/mj_opt/models/g1/xml/g1_29dof.xml'
out_urdf_path = '/home/frlab/mj_opt/models/g1/urdf/g1_12dof_fixed.urdf'
out_xml_path = '/home/frlab/mj_opt/models/g1/xml/g1_12dof_fixed.xml'

joints_to_remove = [
    'waist_yaw', 'waist_roll', 'waist_pitch',
    'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow', 'left_wrist_roll', 'left_wrist_pitch', 'left_wrist_yaw',
    'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow', 'right_wrist_roll', 'right_wrist_pitch', 'right_wrist_yaw'
]

# 1. Process URDF
with open(urdf_path, 'r') as f:
    urdf_content = f.read()

for j in joints_to_remove:
    j_name = j + '_joint'
    # Find <joint name="JNAME" type="revolute">
    # regex since spacing might differ
    pattern = r'<joint\s+name="' + j_name + r'"\s+type="revolute">'
    urdf_content = re.sub(pattern, f'<joint name="{j_name}" type="fixed">', urdf_content)

with open(out_urdf_path, 'w') as f:
    f.write(urdf_content)

# 2. Process XML
with open(xml_path, 'r') as f:
    xml_content = f.read()

# We need to remove:
# <joint name="JNAME" ... />
# <jointpos name="..." joint="JNAME" ... />
# <jointvel name="..." joint="JNAME" ... />
# <jointactuatorfrc name="..." joint="JNAME" ... />
# <motor name="..." joint="JNAME" ... />
# Note that XML tags could span multiple lines, but in this XML they are mostly single line or a few lines.
# For <joint ...>, it might be multi-line: e.g.
# <joint name="waist_yaw_joint" pos="0 0 0" axis="0 0 1" range="-2.618 2.618"
#    actuatorfrcrange="-88 88" class="torso_motor"/>

for j in joints_to_remove:
    j_name = j + '_joint'
    
    # Remove <joint name="j_name" ... />
    # Also <motor ... joint="j_name" ... />
    # <jointpos ... joint="j_name" ... />
    # <jointvel ... joint="j_name" ... />
    # <jointactuatorfrc ... joint="j_name" ... />
    
    # It's safer to read line by line and skip lines that belong to these tags.
    # But wait, some tags like <joint> span multiple lines.

lines = xml_content.split('\n')
out_lines = []
skip_mode = False
for line in lines:
    if skip_mode:
        if '/>' in line or '</joint>' in line or '</motor>' in line or '</jointpos>' in line or '</jointvel>' in line or '</jointactuatorfrc>' in line:
            skip_mode = False
        continue
    
    skip_this = False
    for j in joints_to_remove:
        j_name = j + '_joint'
        # Check if line contains start of a tag we want to remove for this joint
        if (f'name="{j_name}"' in line and '<joint ' in line) or \
           (f'joint="{j_name}"' in line and any(x in line for x in ['<motor ', '<jointpos ', '<jointvel ', '<jointactuatorfrc '])):
            skip_this = True
            if '/>' not in line and not line.endswith('</joint>') and not line.endswith('</motor>'):
                skip_mode = True
            break
            
    if not skip_this and not skip_mode:
        out_lines.append(line)

with open(out_xml_path, 'w') as f:
    f.write('\n'.join(out_lines))

print("Created 12DoF URDF and XML files successfully.")
