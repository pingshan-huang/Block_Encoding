# Qiskit PMR circuit review and local integration

Review date: 2026-11-12. Original circuits: Antonio commit
`9b091ffdd2f4daec9b39bfe5fc1dc1b7c3f5d667`. Integration baseline: main commit
`f303c046b5b2a488c1c32293a66dd131f9355dea`. Local feature branch:
`circuit-review-optimization`. The original review was completed locally before
the separate integration onto the existing Zayar history.

## A. Original implementation

Antonio supplied two front ends for the same grouped PMR construction. The
function-oriented `code/qiskit_fig5.py` builds the complete circuit directly.
`Block Encoding Circuit/` exposes PREP, VALUE, r, R, and MOVE as individual
objects, plus an inspection utility. Their original circuits and bounded
resource counts match. They are complementary interfaces with duplicated
implementation, rather than different block-encoding algorithms.

The source-indexed convention is

    H = sum_mu,x c_mu(x) |x XOR mu><x|.
    c_tilde_mu(x) = c_mu(x) - s if mu == 0, otherwise c_mu(x).
    alpha = sum_mu b_mu; v_mu(x) = c_tilde_mu(x) / b_mu.

For a zero bound, the whole centered row must be exactly zero and VALUE is
defined as zero without division. A zero row with a positive bound retains its
preparation weight. These are distinct cases. The target is `(H - s I)/alpha`.

The circuit uses flag, workspace, move, and system registers, in that order.
Each register is little endian. The workspace stores theta in low bits and
phase in high bits. The six stages are:

1. **PREP:** unitary StatePreparation maps the zero move register to
   `sum_mu sqrt(b_mu/alpha)|mu>`, with zero amplitude on padded moves.
2. **r:** a classical VALUE table is quantized and used to synthesize controlled
   X gates. The resulting reversible operation is
   `|b,mu,x> -> |b XOR word(mu,x),mu,x>`. VALUE itself is not a quantum gate.
3. **R:** for each used word, apply `diag(exp(i phi),1) Ry(theta)` to the flag.
   Its flag-zero amplitude is `cos(theta/2) exp(i phi)`. The phase belongs on
   flag zero and is not a removable global phase. Unused words act as identity.
4. **r dagger:** repeat the XOR lookup. R never changes workspace bits, so this
   clears workspace initialized to zero, including for either flag input.
5. **MOVE:** apply the row's XOR mask to the source system state. Padded moves
   act as identity. This happens after uncomputing the source-indexed word.
6. **PREP dagger:** invert preparation. Projecting the move register onto zero
   supplies a second factor of `sqrt(b_mu/alpha)`.

Consequently, the zero-ancilla block is the PMR sum with quantized VALUE
coefficients and approximates `(H - s I)/alpha`. The finite-precision block
need not be exactly Hermitian. The model checks numerical Hermiticity within
its tolerance, rather than projecting inputs onto exact Hermitian matrices.

The original extractor's offset arithmetic was correct for its fixed layout.
The new extractor obtains actual qubit positions through `circuit.find_bit`,
so interleaved or reordered system registers also work. No postselection
renormalization or per-column phase alignment is performed. This agrees with
[Qiskit's bit-order convention](https://quantum.cloud.ibm.com/docs/en/qiskit/bit-ordering).
PREP uses the unitary
[StatePreparation API](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.StatePreparation),
not reset-based initialization.

## B. Issues discovered

Confirmed issues:

- Both original R builders used `np.isclose` to decide whether to omit rotations.
  This discards legitimate nonzero fixed-point codes. For example, one phase
  step at 30 phase bits is about `5.85e-9` radians. New regressions require that
  this phase and a similarly small theta produce nonzero amplitudes.
- The modular PMR model duplicated older validation, rejected zero bounds,
  lacked the shared core's ownership protections, and used different bound and
  Hermiticity checks. Its VALUE implementation divided directly by row bounds.
  Reusing the current core resolves this integration mismatch.
- The old codec could corrupt packed fields at extreme accepted precision:
  54 theta bits and one phase bit can encode zero with a theta code of `2**54`,
  spilling into phase and decoding near minus one. The circuit wrapper now
  rejects unsupported field widths; the existing sparse codec is untouched.
- Exact-pattern gates repeatedly opened and closed the same controls with X.
  This creates redundant logical gates. Reusing inversion masks removes many
  of them without dropping the required r dagger step.
- Circuit logic was duplicated between the two front ends. The extractors
  repeatedly constructed/synthesized a circuit for each input column.

Limitations, rather than demonstrated incorrect low-precision circuits:

- Lookup construction enumerates `M * 2**n` entries. This is explicit table
  synthesis, not a scalable QRAM or arithmetic VALUE implementation.
- Multi-controlled operations remain expensive after decomposition. Fewer
  logical X gates do not guarantee fewer CX gates or lower physical depth.
- Default precision uses 28 workspace qubits. The one-system-qubit example has
  31 total qubits, requiring about 32 GiB for just one complex128 statevector.
- The flat `main.py` module names can collide when both directories are imported
  indiscriminately. The original standalone entry points are retained; tests
  isolate their module names. A package-wide import redesign was not attempted.

## C. Changes implemented

No previously tracked main-baseline source, mathematical-core test, sparse
simulator, or environment file was modified. These circuit sources are new
working-tree files relative to main, selectively imported from Antonio and
then reviewed. Original contributor attribution is retained.

| Files | Changes and relevant entry points |
| --- | --- |
| `code/qiskit_oracles.py` | New shared `QiskitAnglePhaseCodec`, owned `QiskitRTable`, register validation, `_pattern_walk`, `append_prep`, `append_r`, `append_flag_rotation`, `append_move`, and bounded `extract_encoded_block`. |
| `code/qiskit_fig5.py` | Retains `QiskitFig5Build`, `build_qiskit_fig5_circuit`, CLI, and extractor interface; delegates primitives and extraction to shared helpers. |
| `code/requirements-qiskit.txt` | Original optional requirements imported unchanged. No installation performed. |
| `Block Encoding Circuit/_shared_core.py` | Locates the sibling validated core without replacing the modular CLI's `main` import. |
| `Block Encoding Circuit/model.py` | Reexports the shared `PMRModel` and `next_power_of_two`. |
| `Block Encoding Circuit/value_oracle.py` | Shared VALUE subclass retaining `.value(move_index, x)`. |
| `Block Encoding Circuit/angle_phase_codec.py` | Reexports the circuit codec and shared decoded rotation data. |
| `Block Encoding Circuit/prep_oracle.py` | Keeps `PrepOracle`, `target_state`, and append interface; delegates circuit synthesis. |
| `Block Encoding Circuit/r_oracle.py` | Keeps `LowercaseROracle`, `.word`, `.rotation`, append and inverse; uses owned shared lookup data. |
| `Block Encoding Circuit/flag_rotation_oracle.py` | Keeps `FlagRotationOracle.append`; delegates corrected integer-code selection and shared rotations. |
| `Block Encoding Circuit/move_oracle.py` | Keeps `MoveOracle.append`; delegates mask synthesis and register checks. |
| `Block Encoding Circuit/main.py` | Keeps register/oracle/build containers and CLI; uses shared extraction and integration. |
| `Block Encoding Circuit/circuit_utils.py`, `inspect_oracle.py`, `test.py` | Imported with documentation improvements; executable ASTs, including original test assertions, unchanged from Antonio. |
| `Block Encoding Circuit/requirements.txt` | Original requirements imported unchanged. |
| `tests/test_qiskit_circuit.py` | 20 bounded circuit and primitive regressions. |
| `tests/test_qiskit_modular.py` | Nine shared-core and cross-interface regressions. |
| `tests/benchmark_qiskit.py` | Reproducible, bounded before/after driver with independent matrix references. |
| `doc/qiskit_review.md` | This report and reproduction instructions. |

For each primitive, exact control patterns define orthogonal subspaces, so
their target operations commute. `_pattern_walk` orders the selected patterns
along a Gray path, retains the current X inversion mask, changes only differing
bits, and restores the final mask. These masks telescope on arbitrary inputs,
including superpositions. The algorithm sorts the selected patterns instead
of enumerating every possible workspace word. Controlled operations and
ancilla counts are retained. This proves the control traversal's equivalence;
complete small operators provide an independent numerical check.

The extractor lowers once to `u,cx` at optimization level 0, seed 7, then evolves
each system input using the same lowered circuit. It checks the caller's qubit
limit before synthesis/allocation and additionally caps the dense output block
at 64 MiB. This reduces repeated synthesis and avoids dense controlled-gate
matrices during evolution; it does not remove exponential statevector cost.

Intentional compatibility changes: circuit precision fields must be integers
other than booleans, each in 1..52, with total width at most 64. Invalid register
widths, overlapping qubits, or foreign qubits raise errors. Modular model/VALUE
and lookup data now follow the shared immutable ownership rules. The monolithic
build's `r_table` is now a classical `QiskitRTable`; lookup/word inspection stays
available, but incidental sparse `RBlock.apply_basis`, `apply`, and
`apply_inverse` methods are not provided. The sparse backend remains accessible
through its existing modules. No reviewed circuit consumer used these methods.

## D. Verification results

Before behavioral edits: all 58 mathematical-core tests, seven sparse tests,
and seven original modular tests passed. Both original circuit interfaces
produced matching bounded benchmark results. The original source snapshot and
SHA256 manifest were preserved outside the checkout.

Historical review results (before the separate Zayar integration verification):

| Suite | Actual result |
| --- | --- |
| Mathematical core | 58 passed, 0 failures/errors/skips |
| Existing sparse simulator | 7 passed, 0 failures/errors/skips |
| Original modular circuit tests | 7 passed, 0 failures/errors/skips |
| New circuit/primitive tests | 20 passed, 0 failures/errors/skips |
| New modular integration tests | 9 passed, 0 failures/errors/skips |

A combined discovery run of all 87 tests under `tests/` also passed in the
Qiskit environment, checking that the two interfaces do not contaminate test
imports. NumPy 2.5.2 emitted two deprecation warnings from the existing ownership
tests' intentional shape/dtype mutation probes. These were warnings, not test
failures, and reinforce the environment-version limitation below.

New checks cover X/Y/Z and negative X, complex attenuation at multiple precisions,
unsorted masks, multiqubit matrices, padded states, zero-bound and positive-bound
zero rows, explicit-positive all-zero targets, different bounds, invalid masks,
public ownership paths, and numerical Hermiticity. Primitive checks compare
complete small operators, including arbitrary initial workspace, both flag
inputs, unused words, and inverse operations. Extraction checks interleaved and
reversed system qubits plus an explicit global phase. Guard tests verify that
oversized circuits reach neither transpilation nor statevector construction.

The main new block checks use elementwise `atol=2e-10, rtol=0` against an
independently quantized reference. Primitive and modular comparisons use
`atol=1e-12` or `2e-12, rtol=0`; tiny-phase/theta checks use `1e-14` and separate
nonzero assertions. Benchmark spectral-norm logic error must be at most `1e-12`.
No existing assertion or tolerance was weakened.

For both interfaces, full original/reviewed operators were compared with
theta=1, phase=2, on six- and eight-qubit fixtures. Frobenius differences were
`3.590730762045541e-15` and `5.555621819780098e-14`, respectively. Maximum
`||U^dagger U-I||_F` across original and reviewed operators was
`2.385061279881218e-12`. Every metric passed the explicit `1e-10` acceptance
tolerance; no phase alignment was used. These are bounded checks, not proof
by simulation for arbitrary system sizes.

The unchanged sparse demonstration at theta=12, phase=16 reports spectral error
`1.277e-6`, below its `5e-5` acceptance threshold. It uses sparse states, not a
31-qubit dense simulation. Both Qiskit CLI demonstrations at theta=3, phase=4
run successfully and report `4.726e-2`; the independent benchmark gives the
more precise values below.

Environment: core tests used the existing Python 3.12.14 / NumPy 2.3.5 runtime.
Qiskit checks used existing Python 3.14.7 / Qiskit 2.5.2 / NumPy 2.5.2. The latter
NumPy version is outside the declared `numpy>=1.26,<2.5` range, so these are not
evidence of a fully requirements-compliant Qiskit environment. No dependencies
were installed or upgraded and requirements were not relaxed. Aer 0.17.2 could
not import because Windows Application Control blocked its native DLL. No Aer
tests, hardware runs, or default-workspace dense simulations were performed.
The reported Qiskit suites use built-in `quantum_info`, which ran successfully.

### Final verification for integration onto Zayar

The separate integration was based directly on Zayar commit
`8a6b1c2c785fd9dc446111f52f4954e81e393428`, which already contains mathematical-core
commit `a23775993f9ddf7af7a7ca0239bb7fc8fe44ddbc`. Its starting file tree matches
the review baseline. Only the 20 reviewed circuit, test, requirements, and
documentation files were transferred; the shared mathematical core is unchanged.

Fresh verification in that integration checkout passed 58 mathematical-core
tests, seven sparse-simulator tests, seven original modular tests, 20 new
circuit tests, and nine modular integration tests: **101 tests, zero failures,
errors, or skips**. These are newly run results, separate from the historical
review counts above. Both circuit interfaces also reran the independent
benchmark checks at theta=3, phase=4. Their spectral errors against the
independently quantized target were `2.112747379758738e-14` (one-qubit fixture)
and `1.0263745364313173e-13` (two-qubit fixture), each below `1e-12`.
Errors against the ideal target were `.047259346727128226` and
`.08645366973212082`, respectively, consistent with the quantization results.

Version probes confirmed Python 3.12.14 / NumPy 2.3.5 for the core and sparse
suites, and Python 3.14.7 / Qiskit 2.5.2 / NumPy 2.5.2 for circuit verification.
Qiskit was unavailable in the first environment. A requirements-compliant
Qiskit environment was not available among these existing runtimes; the
`numpy<2.5` compatibility limitation remains unresolved. No dependencies or
assertions were changed to obtain these results.

## E. Optimization measurements

All entries use theta=3, phase=4, barriers disabled, and Qiskit transpilation
to `u,cx`, optimization level 1, seed 7, without a hardware coupling map.
This resource transpilation is distinct from the extractor's level-0 lowering.
Both interfaces give the same deterministic counts and errors.

The one-qubit fixture is `[[1.5, .3-.4j], [.3+.4j, -.5]]`, shift .5, with
explicit bounds `[1, nextafter(.5,+infinity)]` in both original and reviewed
models, alpha 1.5. Identical conservative bounds isolate circuit changes from
the core's upward rounding of automatic bounds. The two-qubit fixture is
specified directly in the benchmark source, with masks `[0,1,3]`, bounds
`[2,5,2]`, shift 1, alpha 9. Its Hamiltonian is:

    [[3,   3-4j,  0,    -2j],
     [3+4j, -1,   2,      0],
     [0,     2,   1,  -4-3j],
     [2j,    0, -4+3j,    0]]

| Metric | One qubit: original -> reviewed | Two qubits: original -> reviewed |
| --- | --- | --- |
| Total qubits | 10 -> 10 | 12 -> 12 |
| Ancillas (including workspace) | 9 -> 9 | 10 -> 10 |
| Workspace qubits | 7 -> 7 | 7 -> 7 |
| Logical depth | 26 -> 27 | 103 -> 88 |
| Logical gates | 62 -> 46 | 241 -> 135 |
| Logical X gates | 46 -> 30 | 190 -> 84 |
| Transpiled depth | 1014 -> 1017 | 4417 -> 4417 |
| Transpiled gates | 1544 -> 1548 | 5894 -> 5890 |
| Transpiled two-qubit gates (all CX) | 719 -> 721 | 2732 -> 2732 |
| Spectral error to ideal target | .04725934672711928 -> .047259346727128226 | .08645366973212518 -> .08645366973212082 |
| Spectral error to independently quantized target | 2.45144e-14 -> 2.11275e-14 | 1.53867e-13 -> 1.02637e-13 |

Logical gate counts decrease about 25.8% and 44.0%. This does **not** establish
a two-qubit-gate improvement: the small fixture gains two CX gates and three
depth layers after transpilation; the larger fixture's CX count and depth are
unchanged. This is a logical control simplification, not a general hardware
resource improvement.

| Timing sample | One qubit original -> reviewed | Two qubits original -> reviewed |
| --- | --- | --- |
| Function interface construction median, 5 builds (ms) | .8031 -> .7970 | 4.0170 -> 1.4756 |
| Modular interface construction median, 5 builds (ms) | .8268 -> .9191 | 2.8366 -> 2.3711 |
| Function interface extraction, single run (s) | .12775 -> .16245 | .82720 -> .78180 |
| Modular interface extraction, single run (s) | .13299 -> .15377 | .81348 -> .86778 |

These short, machine-local timing samples are noisy and include first-use
costs. They support no general construction or simulation speedup claim.

For nearest angle/phase rounding in exact arithmetic, with `L=2**theta_bits-1`
and `P=2**phase_bits`, a conservative spectral error bound is

    pi/(4 L) + pi/P.

Each row is a permutation times a diagonal coefficient error; its norm is
bounded by the largest entry error. Triangle inequality and row weights
summing to one give the displayed bound. At 3+4 bits this is
`.30854927847756897`. Both measured ideal-target errors satisfy that bound,
while the separate `1e-12` quantized-target check distinguishes logic error
from coarse finite-precision error. Float rounding contributes additional
machine-scale error; this bound is not an arbitrary-precision guarantee.

## F. Documentation and reproduction

Docstrings explain source indexing, centering, weights, zero rows, word packing,
R's flag-zero phase, full-space XOR behavior, control-mask cancellation,
uncomputation, actual-qubit extraction, and memory/precision limits. Nontrivial
builders and extraction use NumPy-style parameter/return documentation. Every
test has a short explanation. Antonio's attribution remains distinct from the
local integration and verification work.

Run from the repository root in PowerShell. Set the two interpreter variables
to existing environments suitable for each suite; the review used separate
core and Qiskit environments with the versions and limitation stated above:

```powershell
$corePython = 'python'  # Interpreter for the NumPy core and sparse suites.
$circuitPython = 'python'  # Interpreter with Qiskit available.
& $corePython -B -m unittest discover -s tests -p 'test_pmr_*.py' -v
& $corePython -B -m unittest discover -s code -p test.py -v
& $circuitPython -B -m unittest discover -s tests -p test_qiskit_circuit.py -v
& $circuitPython -B -m unittest discover -s tests -p test_qiskit_modular.py -v
& $circuitPython -B -m unittest discover -s 'Block Encoding Circuit' -p test.py -v
& $corePython -B code/main.py
& $circuitPython -B code/qiskit_fig5.py --theta-bits 3 --phase-bits 4 --simulate
& $circuitPython -B 'Block Encoding Circuit/main.py' --theta-bits 3 --phase-bits 4 --simulate
git diff --check
```

The benchmark driver supports both a reviewed checkout and an exported source
snapshot of Antonio's pinned commit given above. Export the Python sources in
`code/` and `Block Encoding Circuit/`, retaining their relative layout and the
original mathematical model used by that historical version. Exclude virtual
environments, caches, and compiled files. Keep benchmark outputs outside the
repository.

For each source root, interface, and case, run:

```powershell
& $circuitPython -B tests/benchmark_qiskit.py --source-root '<source root>' --implementation monolithic --case example --theta-bits 3 --phase-bits 4 --repeats 5 --output '<external output directory>/result.json'
```

Repeat with `modular` and `two_qubit` for all four combinations. The same driver
supplies identical explicit coefficients, bounds, shifts, precisions, and
transpiler settings to both versions. It rejects configurations above 13 total
qubits before circuit construction. The tables above record the historical
measurements; timings will vary on another machine.

## G. Remaining limitations and next work

The next practical verification task is to run the same suite in a
requirements-compliant Qiskit environment. No dependency installation is part
of this checkpoint. Subsequent optimization should target measured
multi-controlled synthesis costs, using full-space equivalence checks and a
specified hardware basis/coupling map.

The explicit lookup and classical coefficient table still scale exponentially
with system size. Default precision can be constructed but was not densely
simulated. `max_qubits` is caller-adjustable, so increasing it still requires
memory judgment; the default is 16. No hardware, noise, or Aer result is claimed.

Float64 limits remain separate from fixed-point width. At 52 theta bits, zero
can round one code below its ideal endpoint and decode to a magnitude about
`2.83e-16`. Even lower-precision evaluation of `cos(pi/2)` is not bitwise zero.
The wrapper prevents catastrophic field spill but does not promise all 52 bits
of useful numerical precision. No new numerical backend or sparse pruning
change was introduced.

The integration baseline already contains tracked `.venv` files. This circuit
contribution adds, modifies, and removes none of them. Environment cleanup
remains separate from the mathematical and circuit work. The reviewed sources
are integrated onto the existing Zayar history without merging Antonio or main;
the original review checkout is preserved separately.

## H. Explanation for Professor Hen

Antonio's circuit implements grouped PMR block encoding by preparing move
weights, reversibly loading quantized normalized coefficients, rotating a flag
with the appropriate complex phase, clearing the lookup workspace, applying
the XOR move, and undoing preparation. The local integration keeps that
construction and his two interfaces, connects them to the validated PMR core,
fixes the omission of tiny nonzero rotations, and cancels repeated control X
gates. Independent matrix checks separate quantization error from circuit
error, and complete small-unitary comparisons check unused states as well as
the encoded block. Logical gate counts improve on both examples; decomposed
two-qubit counts do not improve. The result is a tested small-system research
implementation with explicit precision and environment limitations, not yet a
scalable or hardware-validated block encoder.
