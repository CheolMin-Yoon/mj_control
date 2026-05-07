# ADMM 솔버의 GPU 구현: CuPy vs NVIDIA Warp

기존 학습 계획 (`optimization_study_plan.md`)에서 Phase 1–4 까지를 numpy/scipy로 구현한 뒤, GPU 가속을 추가하는 단계의 보조 문서입니다. 두 가지 유력한 도구인 **CuPy** 와 **NVIDIA Warp** 의 특성·선택 기준·구체 구현 방법을 정리합니다.

---

## 목차

- [개요: 왜 GPU 인가](#개요-왜-gpu-인가)
- [도구 지형도](#도구-지형도)
- [CuPy vs Warp: 정면 비교](#cupy-vs-warp-정면-비교)
- [선택 가이드](#선택-가이드)
- [CuPy 로 ADMM 구현하기](#cupy-로-admm-구현하기)
- [Warp 로 ADMM 구현하기](#warp-로-admm-구현하기)
- [공통 함정과 디버깅](#공통-함정과-디버깅)
- [학습 트랙에 끼워 넣는 방법](#학습-트랙에-끼워-넣는-방법)

---

## 개요: 왜 GPU 인가

CPU 단일 ADMM 솔버는 한 번 풀이 (= MPC 한 주기) 에 ms 단위가 듭니다. 100 Hz 제어에는 충분하지만, 다음 두 시나리오에서는 GPU 가 필수가 됩니다:

1. **Batched solve**: RL training 중 수천 개 environment 가 동시에 MPC 를 풀 때. CPU 로는 environment 수만큼 선형으로 느려짐. GPU 는 거의 동일 시간.
2. **고주파 + 큰 horizon**: NMPC horizon $T$ 가 30+ 이상이거나 제어 주파수가 500 Hz 이상이면 CPU 로는 한 주기 안에 못 끝낼 수 있음.

본 논문 (Residual MPC) 의 핵심 기여 중 하나가 정확히 이 batched MPC solve 의 GPU 가속이고, 그 위에서야 RL + MPC concurrent training 이 가능해졌습니다.

---

## 도구 지형도

GPU 에서 Python 으로 수치 코드를 작성하는 도구는 여러 추상화 수준에 걸쳐 있습니다.

| 도구 | 추상화 | 특징 |
|------|--------|------|
| CuPy | 높음 | numpy API 호환. cuBLAS/cuSOLVER 자동 호출 |
| CuPy RawKernel | 중간 | CUDA C 문자열로 커널 작성 |
| Numba CUDA | 중간 | Python 함수에 `@cuda.jit` 데코레이터 |
| **NVIDIA Warp** | 중간 | Python 함수에 `@wp.kernel`, 시뮬레이션 친화 |
| JAX | 높음 | 자동미분·`vmap`. dense 위주, GPU 추상화 |
| PyCUDA | 낮음 | CUDA Driver API 바인딩 |
| cuda-python (NVIDIA) | 가장 낮음 | Runtime/Driver API 1:1 매핑 |

본 문서에서는 **CuPy** 와 **NVIDIA Warp** 두 가지에 집중합니다. 학습 트랙 (Phase 1–4) 의 GPU 확장에 가장 실용적인 두 옵션이기 때문입니다.

> **이름 주의**: NVIDIA Warp 는 CUDA 의 "warp" (32 thread 단위) 와 이름만 같고 다른 개념입니다. 본 문서에서 그냥 "Warp" 라고 쓰면 NVIDIA Warp 라이브러리, "CUDA warp" 라고 쓰면 하드웨어 단위입니다.

---

## CuPy vs Warp: 정면 비교

### 입문 부담

| 항목 | CuPy | Warp |
|------|------|------|
| 설치 | `pip install cupy-cuda12x` | `pip install warp-lang` |
| numpy 코드 포팅 | `import cupy as cp` 한 줄 | 커널 함수로 재작성 필요 |
| 첫 동작까지 | 10분 | 반나절 |
| 새 문법 학습 | 거의 없음 | `@wp.kernel`, `wp.array`, `wp.tid()` 등 |

CuPy 가 압도적으로 빠르게 시작할 수 있습니다. 대신 Warp 는 새 문법을 익히는 만큼 **GPU 사고방식 자체를 학습**하게 됩니다.

### 표현력과 제어

| 기능 | CuPy | Warp |
|------|------|------|
| Element-wise 연산 | ✅ ufunc 자동 | ✅ 커널 함수 |
| Reduction | ✅ `cp.sum` 등 | ✅ `wp.atomic_add` |
| Linear algebra (BLAS) | ✅ 내장 (cuBLAS) | ❌ 직접 호출 또는 외부 |
| Sparse 연산 | 부분적 (`cupyx.scipy.sparse`) | ❌ 직접 구현 |
| Shared memory 직접 사용 | RawKernel 안에서만 | ✅ `wp.shared_array` |
| Warp shuffle | RawKernel 안에서만 | ✅ 내장 함수 |
| 자동미분 | ❌ | ✅ `wp.Tape` |
| CUDA Graph | 수동 (`cp.cuda.graphs`) | ✅ first-class |

ADMM 의 핵심 부품인 **backsolve (triangular solve)** 가 CuPy 는 cuSOLVER 호출 한 줄, Warp 는 직접 구현하거나 외부 호출이 필요합니다. 이게 가장 큰 실용적 차이.

### 성능 잠재력

대략적인 ballpark (RTX 3090 기준, ADMM 1 iteration 시간):

| 구현 | 단일 QP | 1000 batched QP |
|------|---------|----------------|
| numpy (CPU) | 50 µs | 50 ms |
| CuPy (high-level only) | 80 µs | 0.5 ms |
| CuPy + RawKernel hot path | 30 µs | 0.2 ms |
| Warp + CUDA Graph | 10 µs | 0.1 ms |
| 보내주신 `socu` 같은 specialized | 5 µs | 0.05 ms |

숫자는 워크로드에 따라 크게 달라지지만 경향성은 유지됩니다. **단일 QP 에서는 launch overhead 때문에 CPU 가 빠를 수도 있고**, batched 에서 GPU 의 진가가 나옵니다.

### 생태계와 미래 가치

| 항목 | CuPy | Warp |
|------|------|------|
| 공식 후원 | OSS 커뮤니티 (Preferred Networks) | NVIDIA |
| 주 사용 분야 | 데이터 과학, 머신러닝 | 로봇, 그래픽스, 물리 시뮬레이션 |
| 통합 생태계 | scipy 호환 (cupyx) | IsaacLab, IsaacSim, Newton |
| 학습 자료 | 풍부 | 적지만 늘고 있음 |
| 본 논문과의 거리 | 중간 | **가까움** |

본 논문 같은 로봇 MPC + RL 방향으로 가고 싶다면 Warp 의 생태계가 더 자연스럽습니다.

---

## 선택 가이드

### CuPy 를 추천하는 경우

- numpy 코드를 빠르게 GPU 로 옮겨 **일단 가속 효과만 확인**하고 싶다
- 학습 시간이 제한적이고, 새 문법 학습보다 결과가 우선
- 데이터 과학·일반 ML 영역에 더 가까운 작업
- ADMM 의 backsolve 같은 BLAS 의존도가 높은 부분이 핵심

### Warp 를 추천하는 경우

- 본 논문 같은 **GPU MPC + RL** 방향으로 진지하게 가고 싶다
- IsaacLab, IsaacSim 등 NVIDIA 로봇 생태계와 통합 예정
- **CUDA Graph, 자동미분** 같은 고급 기능이 필요
- GPU 사고방식 (커널, 메모리 모델) 자체를 배우는 게 목표
- Hot path 를 직접 커널로 작성할 의향

### 권장 절충안: 둘 다 쓴다

실제로 가장 합리적인 답은 **두 도구를 같이 쓰는 것**입니다. 둘 다 `__cuda_array_interface__` 를 지원해서 zero-copy 변환이 가능하기 때문이에요:

```python
import cupy as cp
import warp as wp

# Warp -> CuPy
z_cp = cp.from_dlpack(wp.to_dlpack(z_warp))

# CuPy -> Warp
A_warp = wp.from_dlpack(A_cp.toDlpack())
```

ADMM 의 부분별로 적합한 도구를 쓰는 패턴:

| 부품 | 추천 도구 | 이유 |
|------|----------|------|
| KKT backsolve | CuPy (cuSOLVER) | BLAS 가 핵심, 직접 짜면 손해 |
| Box projection | Warp | element-wise + clamp, Warp 가 깔끔 |
| Dual update | Warp 또는 CuPy | 어느 쪽이든 비슷 |
| 잔차 계산 | CuPy | reduction 한 줄 |
| 전체 루프 | Warp + CUDA Graph | overhead 제거 효과 큼 |

---

## CuPy 로 ADMM 구현하기

### 설치

```bash
pip install cupy-cuda12x   # CUDA 12 기준. 시스템에 맞게
```

### Step 1: numpy 코드 포팅

기존 Phase 1 의 ADMM 이 numpy 로 작성되어 있다면, `import` 한 줄만 바꿔도 동작합니다:

```python
# 기존
import numpy as np

# CuPy 버전
import cupy as cp
np = cp   # 또는 모듈 alias 패턴

class ADMMQPSolver:
    def __init__(self, P, q, A, l, u, ...):
        self.P = cp.asarray(P)
        self.q = cp.asarray(q)
        # ... 나머지 동일
```

대부분의 numpy 함수가 cupy 에 그대로 있어서 (`cp.linalg.solve`, `cp.clip`, `cp.linalg.norm` 등) 큰 변경 없이 동작합니다.

### Step 2: KKT 분해와 backsolve

ADMM 에서 가장 비싼 연산입니다. CuPy 는 cuSOLVER 를 통해 LDL 또는 LU 분해를 제공합니다:

```python
from cupyx.scipy.linalg import lu_factor, lu_solve

# 한 번만 분해
lu, piv = lu_factor(KKT)

# 매 iteration backsolve
def admm_iter(x, z, y, lu, piv, ...):
    rhs = cp.concatenate([sigma * x - q, z - y / rho])
    sol = lu_solve((lu, piv), rhs)
    x_tilde, nu = sol[:n], sol[n:]
    # ... z, y update
```

### Step 3: Element-wise 부분

Box projection, dual update 모두 한 줄:

```python
def z_update(Ax, z, y, l, u, rho, alpha):
    return cp.clip(alpha * Ax + (1 - alpha) * z + y / rho, l, u)

def y_update(y, Ax, z_old, z_new, rho, alpha):
    return y + rho * (alpha * Ax + (1 - alpha) * z_old - z_new)
```

CuPy 가 자동으로 GPU 커널을 fuse 해줍니다.

### Step 4: Batched solve

수천 개 QP 를 동시에. 핵심은 leading dim 을 batch 로 두는 것:

```python
# P_batch shape: (N, n, n)
# rhs_batch shape: (N, n+m)

# CuPy 의 batched linear solve
sols = cp.linalg.solve(KKT_batch, rhs_batch)
# 또는 cupyx.scipy.linalg.lu_solve 의 batched 버전
```

`cupy.linalg.solve` 는 leading dim 이 batch 라고 자동으로 인식해서 병렬 처리합니다. 가장 큰 실용적 이득이 여기서 나옵니다.

### Step 5: Hot path 만 RawKernel 로 (선택)

기본 CuPy 만으로도 충분히 빠르지만, hot path 를 직접 커널로 짜면 추가 가속 가능:

```python
box_clip_kernel = cp.RawKernel(r'''
extern "C" __global__
void box_clip(float* z, const float* l, const float* u, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float v = z[tid];
        v = fmaxf(v, l[tid]);
        v = fminf(v, u[tid]);
        z[tid] = v;
    }
}
''', 'box_clip')

# 사용
threads = 256
blocks = (n + threads - 1) // threads
box_clip_kernel((blocks,), (threads,), (z, l, u, n))
```

학습 가치가 가장 큰 부분. 메모리 coalescing, thread indexing 같은 GPU 기초가 손에 잡힘.

### 검증

기존 numpy ADMM 의 답과 일치 확인:

```python
x_np = solve_numpy(P, q, A, l, u)
x_cp = cp.asnumpy(solve_cupy(P, q, A, l, u))
assert np.allclose(x_np, x_cp, atol=1e-5)
```

---

## Warp 로 ADMM 구현하기

### 설치

```bash
pip install warp-lang
```

### Step 1: 기본 구조

Warp 코드는 세 가지 요소로 구성됩니다:

1. **`wp.array`**: GPU 메모리에 있는 텐서
2. **`@wp.kernel`**: GPU 에서 실행되는 함수
3. **`wp.launch`**: 커널을 실행하는 호출

```python
import warp as wp

wp.init()

# array
n = 1024
z = wp.zeros(n, dtype=float, device="cuda")
l = wp.full(n, -1.0, dtype=float, device="cuda")
u = wp.full(n, 1.0, dtype=float, device="cuda")

# kernel
@wp.kernel
def box_clip(z: wp.array(dtype=float),
             l: wp.array(dtype=float),
             u: wp.array(dtype=float)):
    i = wp.tid()
    z[i] = wp.clamp(z[i], l[i], u[i])

# launch
wp.launch(box_clip, dim=n, inputs=[z, l, u])
```

### Step 2: ADMM 의 element-wise 부분

z update 와 y update 를 커널로:

```python
@wp.kernel
def admm_z_update(
    Ax: wp.array(dtype=float),
    z: wp.array(dtype=float),     # in-place 업데이트
    y: wp.array(dtype=float),
    l: wp.array(dtype=float),
    u: wp.array(dtype=float),
    rho: float,
    alpha: float,
):
    i = wp.tid()
    v = alpha * Ax[i] + (1.0 - alpha) * z[i] + y[i] / rho
    z[i] = wp.clamp(v, l[i], u[i])

@wp.kernel
def admm_y_update(
    y: wp.array(dtype=float),
    Ax: wp.array(dtype=float),
    z_old: wp.array(dtype=float),
    z_new: wp.array(dtype=float),
    rho: float,
    alpha: float,
):
    i = wp.tid()
    y[i] = y[i] + rho * (alpha * Ax[i] + (1.0 - alpha) * z_old[i] - z_new[i])
```

### Step 3: Backsolve 는 어떻게?

Warp 는 BLAS 를 자체 제공하지 않습니다. 세 가지 선택지:

**(a) CuPy 와 혼용 (권장)**

```python
import cupy as cp

# Warp 에서 받은 데이터를 CuPy 로 변환 (zero-copy)
KKT_cp = cp.from_dlpack(wp.to_dlpack(KKT_warp))
rhs_cp = cp.from_dlpack(wp.to_dlpack(rhs_warp))

# CuPy/cuSOLVER 로 풀이
sol_cp = cp.linalg.solve(KKT_cp, rhs_cp)

# 결과를 다시 Warp 로
sol_warp = wp.from_dlpack(sol_cp.toDlpack())
```

**(b) 직접 구현** (`socu` 코드 패턴)

block-tridiagonal 구조를 활용해서 Cholesky + triangular solve 를 Warp 커널로 직접 작성. 본격적인 학습 가치는 있지만 시간 투자가 큼.

**(c) 외부 라이브러리 호출**

`cuSOLVER` 를 ctypes 로 직접 호출. 권장하지 않음.

### Step 4: 잔차 계산 (reduction)

```python
@wp.kernel
def compute_residual(
    Ax: wp.array(dtype=float),
    z: wp.array(dtype=float),
    out: wp.array(dtype=float),    # length 1
):
    i = wp.tid()
    diff = Ax[i] - z[i]
    wp.atomic_add(out, 0, diff * diff)
```

`wp.atomic_add` 가 thread-safe 누적. CuPy 의 `cp.linalg.norm` 한 줄에 비하면 손이 더 가지만, 학습 효과가 큼.

### Step 5: CUDA Graph 로 iteration 묶기

ADMM 의 진가가 드러나는 부분. 매 iteration 의 커널 시퀀스를 한 번 record 하면, 이후 launch overhead 가 거의 사라집니다:

```python
# Graph capture
with wp.ScopedCapture() as capture:
    # 한 iteration 분량의 모든 launch
    wp.launch(compute_Ax, dim=m, inputs=[A, x, Ax])
    wp.launch(admm_z_update, dim=m, inputs=[Ax, z, y, l, u, rho, alpha])
    wp.launch(admm_y_update, dim=m, inputs=[y, Ax, z_old, z, rho, alpha])
    # ... etc

graph = capture.graph

# 이후 반복 호출
for _ in range(max_iter):
    wp.capture_launch(graph)
```

전형적 효과: iteration 당 50–100 µs → 5–10 µs.

### Step 6: Batched 확장

leading dim 에 batch 를 추가하고 커널 인덱싱만 조정:

```python
@wp.kernel
def admm_z_update_batched(
    Ax: wp.array2d(dtype=float),    # (B, m)
    z: wp.array2d(dtype=float),
    y: wp.array2d(dtype=float),
    l: wp.array2d(dtype=float),
    u: wp.array2d(dtype=float),
    rho: float,
    alpha: float,
):
    b, i = wp.tid()
    v = alpha * Ax[b, i] + (1.0 - alpha) * z[b, i] + y[b, i] / rho
    z[b, i] = wp.clamp(v, l[b, i], u[b, i])

# launch with 2D dim
wp.launch(admm_z_update_batched, dim=(B, m), inputs=[...])
```

수천 environment 동시 풀이가 자연스럽게 됩니다.

### 검증

numpy 답과 비교:

```python
x_np = solve_numpy(P, q, A, l, u)

z_warp = wp.zeros(...)
solve_warp(...)
x_warp = z_warp.numpy()    # GPU -> CPU

assert np.allclose(x_np, x_warp, atol=1e-5)
```

---

## 공통 함정과 디버깅

### CuPy

| 증상 | 원인 | 대응 |
|------|------|------|
| 답이 numpy 와 다름 | float32 정밀도 차이 | dtype 명시적으로 지정, atol 넉넉히 |
| 첫 호출 매우 느림 | JIT 컴파일 지연 | warmup 1–2회 후 측정 |
| 메모리 부족 | 캐시 누적 | `cp.get_default_memory_pool().free_all_blocks()` |
| 동기화 누락 | 비동기 launch | 측정 시 `cp.cuda.Stream.null.synchronize()` |

### Warp

| 증상 | 원인 | 대응 |
|------|------|------|
| 커널 안에서 dict/list 못씀 | Warp 는 제한된 subset | 기본 자료형, `wp.array`, `wp.struct` 만 |
| 에러가 엉뚱한 곳에서 뜸 | 비동기 실행 | `wp.config.mode = "debug"` + `wp.synchronize_device()` |
| `print` 안 됨 | 커널 안 | `wp.printf` 사용 |
| 첫 launch 느림 | 컴파일 캐시 | warmup 후 측정 |
| Variable 이 Python 객체 | 커널 인자 타입 명시 안 함 | 모든 인자에 dtype 명시 |

### GPU 공통 디버깅 팁

1. **무조건 numpy 답과 매칭부터.** GPU 코드를 먼저 짜고 검증을 미루면 알고리즘 버그와 GPU 버그가 섞여서 디버깅 불가능.
2. **작은 사이즈로 먼저.** $n = 4, m = 2$ 짜리 toy 문제로 step-by-step 출력. element 단위로 비교.
3. **타이밍은 항상 동기화 후.** 비동기 launch 때문에 동기화 안 하면 시간 측정이 무의미.
4. **dtype 명시.** float32 vs float64 가 섞이면 미묘한 정확도 손실. 처음에는 float64 로 통일하고 나중에 float32 로.

---

## 학습 트랙에 끼워 넣는 방법

기존 `optimization_study_plan.md` 에 GPU 단계를 추가하는 두 가지 시나리오:

### 시나리오 A: CuPy 우선

```
Phase 1: ADMM (numpy)        ← 알고리즘 검증
   ↓
Phase 2: Ruiz                ← Phase 1 에 plug-in
   ↓
[NEW] Phase 1.5: CuPy 포팅   ← 1–2 일, 즉각적 가속 효과
   ↓
Phase 3: SQP                ← CPU 로 정확성 우선
   ↓
Phase 4: RTI + NMPC         ← CuPy batched 로 수천 env
```

장점: 빠르게 GPU 효과 체감. numpy 자산 거의 그대로 재사용.
단점: 본 논문 같은 specialized solver 까지는 거리가 있음.

### 시나리오 B: Warp 본격 도입

```
Phase 1: ADMM (numpy)        ← 알고리즘 검증
   ↓
Phase 2: Ruiz
   ↓
[NEW] Phase 1.5: Warp 입문   ← 반나절, 기본 예제
   ↓
[NEW] Phase 1.6: Warp ADMM   ← 1주, element-wise 부분 + CuPy backsolve
   ↓
[NEW] Phase 1.7: CUDA Graph  ← 1–2 일
   ↓
Phase 3: SQP (CasADi/JAX)   ← 자동미분 필요
   ↓
Phase 4: Batched NMPC (Warp) ← 본 논문 방향
   ↓
[선택] Phase 5: Block-tridiagonal solver 직접 구현
```

장점: 본 논문 방향과 정렬. Warp + IsaacLab 로 자연스럽게 확장.
단점: 시간 투자 큼. 학습 자료 부족.

### 권장: 하이브리드

가장 실용적인 진행은:

1. **Phase 1–4 까지는 numpy/CasADi 로 알고리즘 검증.** GPU 는 일단 배제.
2. **Phase 1.5 로 CuPy 포팅을 짧게 (1–2 일).** 즉각적 효과 체감.
3. **Phase 5 로 Warp 로 본격 재구현.** 본 논문 방향이 확실해진 후.

이렇게 가면 각 단계가 명확한 학습 목표를 갖고, 도구 선택을 너무 일찍 결정하지 않아도 됩니다.

---

## 참고 자료

### CuPy
- 공식 문서: docs.cupy.dev
- GitHub: github.com/cupy/cupy
- 예제: docs.cupy.dev/en/stable/user_guide/basic.html

### NVIDIA Warp
- 공식 문서: nvidia.github.io/warp/
- GitHub: github.com/NVIDIA/warp
- 튜토리얼: 공식 문서의 "Getting Started"
- 예제 모음: GitHub repo 의 `warp/examples/`

### GPU 일반
- "Programming Massively Parallel Processors" (Kirk & Hwu): GPU 프로그래밍 표준 교재
- NVIDIA CUDA C Programming Guide: 메모리 모델, warp, occupancy 등 하드웨어 이해

### 본 논문 관련
- CusADi (논문 [29]): GPU 병렬화된 CasADi 함수 코드 생성
- cuDSS: NVIDIA 의 batched sparse direct solver
- IsaacLab: 본 논문의 IsaacGym 후속작, Warp 와 통합 진행 중
