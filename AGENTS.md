# Agent Entry

MuJoCo 시뮬레이션과 Pinocchio 모델 함수로 고정 기반 로봇 제어기를 구현하는 저장소다. 작업은 아래 정본 문서에서 시작한다.

## 1. 라우팅

| 질문 | 정본 |
| --- | --- |
| 설치, 배치, 실행 명령 | [README](README.md) |
| 배치·명명·주석·lint 계약 | [작성원칙](docs/작성원칙.md) |
| UR5e·Panda 데모, 실행 흐름, 제어식, 알려진 제약 | [manipulator](docs/manipulator/README.md) |
| 카트폴 LQR·학습 목록 | [cartpole](docs/cartpole/README.md) |
| RB-Y1 양팔 SE(3) 임피던스 | [rby1](docs/rby1/README.md) |
| G1 humanoid·Go2 quadruped | 로컬 브랜치 `humanoid-quadruped`(정리 전 `master` 스냅샷, 노트북 포함) |

## 2. 환경

- venv: `~/.venvs/mj_control`(Python 3.14, `pin`에 cp315 wheel 없음). 의존성은 `pyproject.toml`, 고정은 `uv.lock`
- 동기화: `VIRTUAL_ENV=~/.venvs/mj_control uv sync --active --project <저장소>`. `--active` 없이 `uv sync`를 하면 저장소 안에
  `.venv`가 생기므로 쓰지 않는다
- 실행: `~/.venvs/mj_control/bin/python scripts/<script>.py`. 진입점은 저장소 루트를 `sys.path`에 넣고 `source.*`를 import한다
- 형식·lint: `~/.venvs/mj_control/bin/ruff format`, `ruff check` (설정은 `pyproject.toml`)

## 3. 규칙

1. 배치와 명명은 [작성원칙](docs/작성원칙.md)을 따른다. 노트북은 두지 않는다.
2. 사실 하나는 소유 문서 하나에 두고 다른 문서는 링크한다. 낡은 문서는 이력 메모 없이 지운다.
3. 커밋·푸시는 명시적으로 요청받을 때만 한다.
4. 가정은 드러내고, 요청 범위 밖의 코드는 고치지 않으며, 바꾼 줄마다 요청에 대응한다. 관련 없는 dead code는 보고만 한다.
5. 변경 검증은 `scripts/` 진입점을 viewer 없이 끝까지 돌려 추종 오차 출력을 변경 전후로 비교한다.

## 4. 도메인 지식 — MuJoCo

LQR 등 선형 모델의 B 행렬은 XML `<actuator>` 타입에 따라 다르게 다룬다.

| 타입 | B 행렬 처리 | 이유 |
| --- | --- | --- |
| `motor` | `B = B_physical * gear` | `force = ctrl × gear` |
| `position` | gear 곱하지 않음 | ctrl이 목표 관절각이고 내부 PD가 토크 계산 |
| `velocity` | gear 곱하지 않음 | ctrl이 목표 관절속도이고 내부 PD가 토크 계산 |

카트폴은 `motor`(gear 50), UR5e `scene.xml`은 `position`, Panda·UR5e `scene_torque.xml`은 `motor`다.

- MuJoCo 4.0: `mjData.qM` 없음, 질량행렬은 `mj_fullM(model, data, dst)`(`mj_fwdPosition` 뒤 유효)
- 관절 순서: actuator index와 qpos·dof 주소는 다르다 → `MujocoKernel`이 관절 이름으로 `actuator_ids`·`qpos_ids`·`dof_ids`를 따로
  매핑한다
