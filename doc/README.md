# Grouped PMR block encoding

The `code` directory contains a framework-independent NumPy implementation of
the grouped block-encoding circuit:

`PREP -> r -> R -> r_dagger -> MOVE -> PREP_dagger`

The implementation does not use Qrisp. Lowercase `r` stores fixed-point
representations of `theta` and `phase` directly in the `b` register.

## Code files

- `main.py`: assembles and verifies the complete circuit flow.
- `pmr_model.py`: validates the PMR Hamiltonian input.
- `value.py`: evaluates normalized centered coefficients.
- `rotation_codec.py`: encodes and decodes fixed-point theta and phase.
- `r_block.py`: reversibly XOR-loads the encoded data into `b`.
- `flag_rotation.py`: applies uppercase `R` from the decoded `b` value.
- `move.py`: applies the move-controlled system permutation.
- `prep.py`: implements PREP and PREP dagger.
- `state.py`: provides the sparse `(f,b,m,x)` state representation.
- `test.py`: tests every primitive and the complete encoded block.

The sparse simulator stores only active basis states, so increasing the number
of fixed-point workspace bits does not allocate an array of size `2**b_bits`.

## Setup

From the project root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\code\requirements.txt
```

## Run

```powershell
.\code\run.cmd
```

## Test

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\code -p "test.py" -v
```
