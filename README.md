# Grouped PMR block encoding

This directory contains a modular Qrisp implementation of the
grouped block-encoding circuit.

The source files correspond to the circuit primitives:

- `prep.py`: uses Qrisp `prepare` for `sqrt(b_mu / alpha_PMR)`.
- `value.py`: evaluates the normalized centered coefficient.
- `r_compute.py`: uses Qrisp controls to reversibly load angle/phase data.
- `flag_rotation.py`: uses controlled Qrisp `ry` and `p` gates for uppercase `R`.
- `move.py`: applies move-controlled Qrisp `x` gates to the system register.
- `pmr_model.py`: validates and stores the PMR Hamiltonian data.
- `main.py`: uses Qrisp conjugation and inversion environments to assemble
  `PREP -> r -> R -> r_dagger -> MOVE -> PREP_dagger`.
- `test_fig5.py`: checks every primitive and the final encoded block.

Run the demonstration:

```powershell
.\.venv\Scripts\python.exe main.py
```

On Windows, the included launcher selects the correct interpreter automatically:

```powershell
.\run.cmd
```

Run the tests:

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

If `ModuleNotFoundError: No module named 'qrisp'` appears, the program is being
run with a different Python interpreter. Select `.venv\Scripts\python.exe` as
the project interpreter, or invoke it explicitly as shown above.

## Scope

The circuit itself is built with Qrisp quantum variables, control environments,
state preparation, rotations, phases, and inversion. Verification simulates one
system-basis input at a time and extracts only the all-zero-ancilla amplitudes;
it does not materialize the exponentially larger full unitary. The `b` register
currently uses reversible symbolic lookup labels for exact `(theta, phi)`
values. Replacing that lookup with Qrisp fixed-point arithmetic changes the
resource cost but not the circuit order.
