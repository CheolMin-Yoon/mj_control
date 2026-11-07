"""Indy7 v3 SE(3) 임피던스 항 CUDA 코드 생성 + 4096 lane GPU 검증, FP64·FP32 커널을 CPU double 기준과 비교"""

import argparse
import subprocess
import sys
from pathlib import Path

import casadi as ca
import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.control.se3_impedance_casadi import TERM_NAMES, se3_impedance_terms  # noqa: E402
from source.dynamics import se3_pin  # noqa: E402
from source.robots.indy7_v3.constants import INDY7_V3_FRAME_NAMES, INDY7_V3_URDF  # noqa: E402

# lane 그룹별 목표 상대 회전 크기 ‖ξ‖ [rad]
XI_SCALES = (0.0, 1e-6, 1e-3, 1e-2, 0.1, 1.0, 2.5, 3.0)

HARNESS = r"""
#include "{source}"
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t err, const char* what) {{
  if (err != cudaSuccess) {{ std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err)); std::exit(1); }}
}}

int main(int argc, char** argv) {{
  const char* dir = argv[1];
  const int n = std::atoi(argv[2]);
  const int reps = std::atoi(argv[3]);
  const int nnz_in[] = {{{nnz_in}}};
  const int nnz_out[] = {{{nnz_out}}};
  casadi_real* d_in[{n_in}];
  casadi_real* d_out[{n_out}];
  char path[4096];
  for (int i = 0; i < {n_in}; ++i) {{
    std::vector<casadi_real> host((size_t)n * nnz_in[i]);
    std::snprintf(path, sizeof(path), "%s/in%d.bin", dir, i);
    FILE* f = std::fopen(path, "rb");
    if (std::fread(host.data(), sizeof(casadi_real), host.size(), f) != host.size()) {{
      std::fprintf(stderr, "short read %s\n", path);
      return 1;
    }}
    std::fclose(f);
    check(cudaMalloc(&d_in[i], host.size() * sizeof(casadi_real)), "cudaMalloc in");
    check(cudaMemcpy(d_in[i], host.data(), host.size() * sizeof(casadi_real), cudaMemcpyHostToDevice), "memcpy in");
  }}
  for (int j = 0; j < {n_out}; ++j) {{
    check(cudaMalloc(&d_out[j], (size_t)n * nnz_out[j] * sizeof(casadi_real)), "cudaMalloc out");
  }}
  check(cudaDeviceSetLimit(cudaLimitStackSize, {stack_bytes}), "stack limit");
  const int block = 128, grid = (n + block - 1) / block;
  for (int r = 0; r < 10; ++r) {kernel}<<<grid, block>>>({args}, n);
  check(cudaGetLastError(), "launch");
  check(cudaDeviceSynchronize(), "warmup sync");
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int r = 0; r < reps; ++r) {kernel}<<<grid, block>>>({args}, n);
  cudaEventRecord(stop);
  check(cudaEventSynchronize(stop), "timed sync");
  float ms = 0.f;
  cudaEventElapsedTime(&ms, start, stop);
  std::printf("%.6f\n", ms / reps);
  for (int j = 0; j < {n_out}; ++j) {{
    std::vector<casadi_real> host((size_t)n * nnz_out[j]);
    check(cudaMemcpy(host.data(), d_out[j], host.size() * sizeof(casadi_real), cudaMemcpyDeviceToHost), "memcpy out");
    std::snprintf(path, sizeof(path), "%s/out%d.bin", dir, j);
    FILE* f = std::fopen(path, "wb");
    std::fwrite(host.data(), sizeof(casadi_real), host.size(), f);
    std::fclose(f);
  }}
  return 0;
}}
"""


def sample_inputs(model, frame_id, num_lanes, rng):
    """lane마다 (q, dq, pose_des, twist_des), 목표 = 현재 TCP·exp(λ_target), ‖ξ_target‖는 XI_SCALES 순환"""
    data = model.createData()
    inputs = [np.zeros((num_lanes, n)) for n in (model.nq, model.nv, 12, 6)]
    scales = np.array([XI_SCALES[i % len(XI_SCALES)] for i in range(num_lanes)])
    for i in range(num_lanes):
        q = rng.uniform(-0.9, 0.9, size=model.nq) * model.upperPositionLimit
        pin.framesForwardKinematics(model, data, q)
        xi = rng.normal(size=3)
        xi *= scales[i] / np.linalg.norm(xi)
        pose_des = data.oMf[frame_id] * pin.exp6(np.r_[rng.normal(scale=0.3, size=3), xi])
        inputs[0][i], inputs[1][i] = q, rng.normal(size=model.nv)
        inputs[2][i] = np.r_[pose_des.rotation.ravel(order="F"), pose_des.translation]
        inputs[3][i] = rng.normal(scale=0.5, size=6)
    return inputs, scales


def cpu_reference(function, inputs):
    """CPU double 평가, 출력마다 (num_lanes, nnz) column-major"""
    num_lanes = inputs[0].shape[0]
    outputs = [np.zeros((num_lanes, function.nnz_out(j))) for j in range(function.n_out())]
    for i in range(num_lanes):
        for j, value in enumerate(function(*(x[i] for x in inputs))):
            outputs[j][i] = np.asarray(value).ravel(order="F")
    return outputs


def pinocchio_crosscheck(model, frame_id, inputs, reference, num_lanes):
    """numpy Pinocchio·se3_pin과 lane 일부 대조, 출력별 최대 절대 오차"""
    data = model.createData()
    worst = dict.fromkeys(TERM_NAMES, 0.0)
    for i in range(num_lanes):
        q, dq, pose_des, twist_des = (x[i] for x in inputs)
        pin.computeAllTerms(model, data, q, dq)
        pin.updateFramePlacements(model, data)
        jacobian_b = pin.getFrameJacobian(model, data, frame_id, pin.ReferenceFrame.LOCAL)
        jdot_dq_b = pin.getFrameJacobianTimeVariation(model, data, frame_id, pin.ReferenceFrame.LOCAL) @ dq
        mass_matrix = np.triu(data.M) + np.triu(data.M, 1).T
        rel = data.oMf[frame_id].inverse() * pin.SE3(pose_des[:9].reshape(3, 3, order="F"), pose_des[9:])
        lam = pin.log6(rel).vector
        lam_dot = se3_pin.dexp6_inv(lam) @ (rel.action @ twist_des - jacobian_b @ dq)
        expected = {
            "lam": lam,
            "lam_dot": lam_dot,
            "dexp": se3_pin.dexp6(lam),
            "dexp_dot": se3_pin.dexp6_dot(lam, lam_dot),
            "mass_matrix": mass_matrix,
            "bias_forces": data.nle,
            "jacobian_b": jacobian_b,
            "jdot_dq_b": jdot_dq_b,
        }
        for j, key in enumerate(TERM_NAMES):
            value = np.asarray(expected[key]).ravel(order="F")
            worst[key] = max(worst[key], float(np.abs(reference[j][i] - value).max()))
    return worst


def build_and_run(function, inputs, work_dir, real, nvcc, cuda_arch, reps):
    """CUDA 생성·컴파일·실행, (lane 출력 목록, launch당 ms) 반환"""
    work_dir.mkdir(parents=True, exist_ok=True)
    dtype = np.float64 if real == "double" else np.float32
    name = function.name()
    generator = ca.CodeGenerator(
        f"{name}_{real}",
        {"cuda": True, "casadi_real": real, "cuda_kernels": {name: {"batch_inputs": list(range(function.n_in()))}}},
    )
    generator.add(function)
    source = Path(generator.generate(str(work_dir) + "/"))
    args = ", ".join([f"d_in[{i}]" for i in range(function.n_in())] + [f"d_out[{j}]" for j in range(function.n_out())])
    stack_bytes = 2 * (function.sz_w() * np.dtype(dtype).itemsize) + 4096
    harness = work_dir / "harness.cu"
    harness.write_text(
        HARNESS.format(
            source=source.name,
            nnz_in=", ".join(str(function.nnz_in(i)) for i in range(function.n_in())),
            nnz_out=", ".join(str(function.nnz_out(j)) for j in range(function.n_out())),
            n_in=function.n_in(),
            n_out=function.n_out(),
            stack_bytes=stack_bytes,
            kernel=f"{name}_kernel",
            args=args,
        )
    )
    binary = work_dir / "harness"
    subprocess.run(
        [str(nvcc), "-O3", f"-arch=sm_{cuda_arch}", "-diag-suppress", "177", str(harness), "-o", str(binary)],
        check=True,
        cwd=work_dir,
    )
    for i, values in enumerate(inputs):
        np.ascontiguousarray(values, dtype=dtype).tofile(work_dir / f"in{i}.bin")
    num_lanes = inputs[0].shape[0]
    run = subprocess.run(
        [str(binary), str(work_dir), str(num_lanes), str(reps)], check=True, capture_output=True, text=True
    )
    outputs = [
        np.fromfile(work_dir / f"out{j}.bin", dtype=dtype).reshape(num_lanes, function.nnz_out(j)).astype(np.float64)
        for j in range(function.n_out())
    ]
    return outputs, float(run.stdout.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-lanes", type=int, default=4096)
    parser.add_argument(
        "--output-dir", type=Path, default=Path.home() / ".local/share/mj-control-casadi/codegen/indy7_v3_se3"
    )
    parser.add_argument("--nvcc", type=Path, default=Path("/usr/local/cuda-13.4/bin/nvcc"))
    parser.add_argument("--cuda-arch", default="120")
    parser.add_argument("--reps", type=int, default=200)
    args = parser.parse_args()

    model = pin.buildModelFromUrdf(str(INDY7_V3_URDF))
    frame_name = INDY7_V3_FRAME_NAMES["ee"]
    frame_id = model.getFrameId(frame_name)
    function = se3_impedance_terms(model, frame_name)
    print(f"{function.name()}: {function.n_instructions()} instructions, sz_w {function.sz_w()}")

    inputs, scales = sample_inputs(model, frame_id, args.num_lanes, np.random.default_rng(0))
    reference = cpu_reference(function, inputs)
    crosscheck = pinocchio_crosscheck(model, frame_id, inputs, reference, num_lanes=256)

    results = {}
    for real in ("double", "float"):
        outputs, ms = build_and_run(
            function, inputs, args.output_dir / real, real, args.nvcc, args.cuda_arch, args.reps
        )
        results[real] = (outputs, ms)
        print(f"{real}: {ms:.4f} ms / launch, {args.num_lanes / ms * 1e3:.3e} lanes/s")

    print(f"\n{'term':<12}{'CPU vs pin':>12}{'fp64 abs':>12}{'fp32 abs':>12}{'fp32 rel':>12}")
    for j, key in enumerate(TERM_NAMES):
        ref = reference[j]
        err64 = np.abs(results["double"][0][j] - ref).max()
        err32 = np.abs(results["float"][0][j] - ref)
        rel32 = (err32.max(axis=1) / np.maximum(np.abs(ref).max(axis=1), 1e-12)).max()
        print(f"{key:<12}{crosscheck[key]:>12.1e}{err64:>12.1e}{err32.max():>12.1e}{rel32:>12.1e}")

    print(
        f"\nfp32 normwise relative error by |xi| group (max over lanes)\n{'|xi|':<8}"
        + "".join(f"{k:>11}" for k in TERM_NAMES)
    )
    for scale in XI_SCALES:
        mask = scales == scale
        row = []
        for j in range(len(TERM_NAMES)):
            ref = reference[j][mask]
            err = np.abs(results["float"][0][j][mask] - ref).max(axis=1)
            row.append((err / np.maximum(np.abs(ref).max(axis=1), 1e-12)).max())
        print(f"{scale:<8g}" + "".join(f"{value:>11.1e}" for value in row))


if __name__ == "__main__":
    main()
