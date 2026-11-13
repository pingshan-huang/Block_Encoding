"""Measure one PMR circuit implementation under a fixed, bounded configuration.

Run each implementation in a separate process so its historical top-level
module names cannot collide with the other implementation. Both versions use
identical explicit bounds, shifts, precisions, and transpiler settings. The
source root may be this checkout or an exported original Antonio snapshot.
Results are JSON; write them outside the checkout to keep artifacts out of Git.
Timing samples describe one machine and are not a statistical speedup claim.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter

import numpy as np
import qiskit
from qiskit import transpile


def main() -> None:
    """Measure a fixed fixture and reject deviations from its quantized reference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--implementation", choices=["monolithic", "modular"], required=True)
    parser.add_argument("--case", choices=["example", "two_qubit"], required=True)
    parser.add_argument("--theta-bits", type=int, default=3)
    parser.add_argument("--phase-bits", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if not (1 <= args.theta_bits <= 8 and 1 <= args.phase_bits <= 8):
        raise ValueError("benchmark precision fields must be between 1 and 8")
    system_bits = 1 if args.case == "example" else 2
    move_bits = system_bits
    if 1 + args.theta_bits + args.phase_bits + move_bits + system_bits > 13:
        raise ValueError("benchmark limits total width to 13 qubits before construction")
    source_root = Path(args.source_root).resolve()
    source_dir = source_root / ("code" if args.implementation == "monolithic" else "Block Encoding Circuit")
    sys.path.insert(0, str(source_dir))
    implementation = importlib.import_module("qiskit_fig5" if args.implementation == "monolithic" else "main")
    builder = implementation.build_qiskit_fig5_circuit if args.implementation == "monolithic" else implementation.build_fig5_circuit

    if args.case == "example":
        expected_h = np.array([[1.5, .3-.4j], [.3+.4j, -.5]], dtype=complex)
        shift, alpha = .5, 1.5
        coefficients = [[1.5, -.5], [.3+.4j, .3-.4j]]
        masks = [0, 1]
        # Cover the represented 0.3+0.4j conservatively in both old and new models.
        # Identical supplied bounds isolate circuit changes from core validation.
        expected_bounds = [1., float(np.nextafter(.5, np.inf))]
        PMRModel = importlib.import_module("pmr_model" if args.implementation == "monolithic" else "model").PMRModel
        model = PMRModel.from_coefficients(1, masks, coefficients, bounds=expected_bounds, shift=shift)
    else:
        PMRModel = importlib.import_module("pmr_model" if args.implementation == "monolithic" else "model").PMRModel
        coefficients = [[3, -1, 1, 0], [3+4j, 3-4j, -4+3j, -4-3j], [2j, 2, 2, -2j]]
        masks = [0, 1, 3]
        expected_bounds = [2., 5., 2.]
        shift, alpha = 1., 9.
        model = PMRModel.from_coefficients(2, masks, coefficients, bounds=expected_bounds, shift=shift)
        expected_h = np.array([
            [3, 3-4j, 0, -2j],
            [3+4j, -1, 2, 0],
            [0, 2, 1, -4-3j],
            [2j, 0, -4+3j, 0],
        ], dtype=complex)
    expected = (expected_h-shift*np.eye(expected_h.shape[0]))/alpha

    # The independently specified target also audits the input-model adapter.
    np.testing.assert_allclose(model.hamiltonian(), expected_h, atol=1e-14, rtol=0)
    np.testing.assert_allclose(model.target_block(), expected, atol=1e-14, rtol=0)
    elapsed = []
    for _ in range(args.repeats):
        start = perf_counter()
        build = builder(model, theta_bits=args.theta_bits, phase_bits=args.phase_bits, barriers=False)
        elapsed.append(perf_counter()-start)
    circuit = build.circuit
    if circuit.num_qubits > 13:
        raise ValueError("Benchmark limits statevector simulations to 13 qubits")
    start = perf_counter()
    encoded = implementation.encoded_block_from_statevector(build, max_qubits=13)
    simulation_seconds = perf_counter()-start
    spectral_error = float(np.linalg.norm(encoded-expected, ord=2))

    # This bound separates expected rounding error from incorrect circuit logic.
    # theta rounding: |delta theta| <= pi/(2*(2**t-1)); cos(theta/2) is 1/2-Lipschitz.
    # phase rounding: |delta phi| <= pi/2**p. The PMR row weights sum to one.
    quantization_bound = float(np.pi/(4*((1 << args.theta_bits)-1)) + np.pi/(1 << args.phase_bits))
    assert np.isfinite(spectral_error) and spectral_error <= quantization_bound+1e-12

    # Form a quantized target independently of VALUE, codec.decode_word and
    # production Hamiltonian reconstruction. Its indexing is source -> target.
    centered = np.array(coefficients, dtype=complex)
    centered[masks.index(0)] -= shift
    quantized = np.zeros_like(expected_h)
    theta_levels = (1 << args.theta_bits)-1
    phase_levels = 1 << args.phase_bits
    for row, mask in enumerate(masks):
        for x in range(expected_h.shape[0]):
            z = centered[row, x] / expected_bounds[row]
            theta = 2*np.arccos(min(1., abs(z)))
            phase = float(np.angle(z)) if abs(z) > 0 else 0.
            theta_code = round(float(theta)*theta_levels/np.pi)
            phase_code = round((phase % (2*np.pi))*phase_levels/(2*np.pi)) % phase_levels
            decoded = np.cos(theta_code*np.pi/(2*theta_levels))*np.exp(2j*np.pi*phase_code/phase_levels)
            quantized[x ^ mask, x] += expected_bounds[row]/alpha*decoded
    logic_error = float(np.linalg.norm(encoded-quantized, ord=2))
    assert logic_error <= 1e-12, f"Quantized-target circuit error: {logic_error}"

    start = perf_counter()
    lowered = transpile(circuit, basis_gates=["u", "cx"], optimization_level=1, seed_transpiler=7)
    transpile_seconds = perf_counter()-start
    two_qubit_count = sum(len(instruction.qubits) == 2 for instruction in lowered.data)
    record = {
        "source_root": str(source_root), "implementation": args.implementation,
        "case": args.case, "python": platform.python_version(), "qiskit": qiskit.__version__,
        "numpy": np.__version__, "declared_numpy_requirement": ">=1.26,<2.5",
        "theta_bits": args.theta_bits, "phase_bits": args.phase_bits, "barriers": False,
        "transpiler": {"basis_gates": ["u", "cx"], "optimization_level": 1, "seed_transpiler": 7},
        "system_qubits": model.n_qubits, "move_count": model.move_count, "move_qubits": model.move_qubits,
        "logical_qubits": circuit.num_qubits, "workspace_qubits": args.theta_bits+args.phase_bits,
        "ancilla_qubits": circuit.num_qubits-model.n_qubits,
        "logical_depth": circuit.depth(), "logical_gate_count": circuit.size(),
        "logical_operations": dict(circuit.count_ops()),
        "transpiled_depth": lowered.depth(), "transpiled_gate_count": lowered.size(),
        "transpiled_two_qubit_count": two_qubit_count, "transpiled_cx_count": lowered.count_ops().get("cx", 0),
        "transpiled_operations": dict(lowered.count_ops()), "construction_seconds": elapsed,
        "construction_median_seconds": statistics.median(elapsed), "simulation_seconds": simulation_seconds,
        "transpile_seconds": transpile_seconds, "spectral_error": spectral_error,
        "quantized_target_logic_error": logic_error, "quantized_target_tolerance": 1e-12,
        "quantization_error_bound": quantization_bound, "masks": masks,
        "model_bounds": model.bounds.tolist(), "model_shift": model.shift, "model_alpha": model.alpha,
        "coefficients": [[[float(z.real), float(z.imag)] for z in row] for row in np.asarray(coefficients, dtype=complex)],
        "independent_target": [[[float(z.real), float(z.imag)] for z in row] for row in expected],
        "encoded_block": [[[float(z.real), float(z.imag)] for z in row] for row in encoded],
    }
    Path(args.output).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
