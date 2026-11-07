# Indy7 모델 자산

뉴로메카 공식 `indy7` 일반형과 `indy7_v3`를 함께 둔다. 일반형의 공식 식별자는 `indy7`이며, 별도의 공식 `indy7_v1` 이름을
붙이지 않는다. 제어기 구현과 독립적인 고정 기반 6축 자산이다.

| 모델 | MuJoCo 장면 | Torque MJCF | Pinocchio URDF |
| --- | --- | --- | --- |
| 일반형 | [scene.xml](../../source/robots/indy7/xmls/scene.xml) | [indy7.xml](../../source/robots/indy7/xmls/indy7.xml) | [indy7.urdf](../../source/robots/indy7/xmls/indy7.urdf) |
| v3 | [scene.xml](../../source/robots/indy7_v3/xmls/scene.xml) | [indy7_v3.xml](../../source/robots/indy7_v3/xmls/indy7_v3.xml) | [indy7_v3.urdf](../../source/robots/indy7_v3/xmls/indy7_v3.urdf) |

## 사용

저장소 루트에서 실행한다.

```bash
~/.venvs/mj_control/bin/python -m mujoco.viewer --mjcf=source/robots/indy7/xmls/scene.xml
~/.venvs/mj_control/bin/python -m mujoco.viewer --mjcf=source/robots/indy7_v3/xmls/scene.xml
```

- 관절 순서: `joint0`, `joint1`, `joint2`, `joint3`, `joint4`, `joint5`
- Pinocchio frame: 기반 `link0`, 말단 `tcp`. MuJoCo 말단 site도 `tcp`
- URDF와 mesh는 각 모델의 `xmls/` 안에서 상대경로로 연결
- MJCF는 6개 `motor`, `gear=1`, `ctrl` 단위 N·m. 공식 URDF의 effort를 제어 입력과 관절 actuator force 한계로 사용
- 공식 URDF의 관절 damping `0.1`, friction `10.0`도 보존. Pinocchio의 `nle`와 MuJoCo의 수동 마찰·감쇠 힘을 구별
- 로컬 설정: timestep `0.002 s`, `implicitfast`, 바닥·조명 장면. `home` keyframe의 관절각은 관찰용 설정이며 공식 home 자세가 아님
- 일반형과 v3의 TCP·기구학·관성은 다르므로 두 모델의 MJCF·URDF·mesh를 섞지 않음

## 출처와 변환

원본은 [neuromeka-robotics/indy-ros2](https://github.com/neuromeka-robotics/indy-ros2),
revision `ca0353f077620c877c1edff2e26c8cd976773f27`이다.

- `indy_description/urdf_files/indy7.urdf`, `indy7_v3.urdf`
- `indy_description/meshes/indy7/`, `indy7_v3/`의 팔 visual·collision STL. IndyEye 변형은 포함하지 않음
- 라이선스 선언: 공식 `indy_description/package.xml`의 `BSD-3-Clause`. 원본 metadata는 각 `xmls/upstream_package.xml`에 보존

URDF의 개발자 홈 절대 mesh URI를 로컬 상대경로로 바꾸고, MuJoCo import용 compiler 설정만 추가했다. 관절 변환·축·한계와
질량·COM·관성은 원본을 사용한다. MJCF는 이 URDF를 MuJoCo로 변환하고 torque actuator, TCP site, 관찰용 keyframe을 추가한
로컬 파생 자산이며 뉴로메카가 배포한 공식 MJCF는 아니다.

Seo 등의 GUFIC 공개 XML을 가져온 자산이 아니다. Kim 등의 TRO 논문 *Impedance Control Design Framework Using
Commutative Map Between SE(3) and se(3)*는 실기 Indy7의 세대를 명시하지 않으므로, 두 모델 어느 쪽도 해당 논문의 저자 제공
모델이라고 표시하지 않는다. 제어기 문헌은 [논문 정본](/home/frlab/rby1-mjlab/docs/papers.md)이 소유한다.
