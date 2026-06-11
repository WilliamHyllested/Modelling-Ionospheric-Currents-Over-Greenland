# Thesis Code — Ionospheric Current Mapping over Greenland

This repository contains the Python code developed during my MSc thesis at the Technical University of Denmark (DTU). The work focuses on modelling and mapping ionospheric currents over Greenland, with comparisons between SuperMAG and DTU modelling configurations.

> **Note:** This code was written as research tooling, not as a distributable package. Expect rough edges — notebooks in particular may have hardcoded paths, undocumented assumptions, or intermediate experiments that were never cleaned up.

---

## Repository structure

- **Notebooks (`.ipynb`)** — The bulk of the analysis, figure generation, and model comparisons. Roughly organised by topic but not strictly sequential.
- **Helper modules (`.py`)** — Reusable functions for:
  - File loading and data I/O
  - Plotting and visualisation
  - Coordinate and time transformations
- **`resolution.py`** — Contains functions adapted from the [EZIE](https://github.com/klaundal/ezie) repository.

---

## Dependencies

### Python packages
Install via pip or conda as needed. No `requirements.txt` is provided, but the main third-party libraries used are listed below.

### External repositories
Several dependencies are not on PyPI and must be installed directly from GitHub:

| Package | Repository |
|---|---|
| `secsy` | https://github.com/klaundal/secsy |
| `polplot` | https://github.com/klaundal/polplot |
| `pyAMPS` | https://github.com/klaundal/pyAMPS |
| `baseline` | https://github.com/klaundal/baseline |
| `lompe` | https://github.com/klaundal/lompe |
| `chaosmagpy` | https://github.com/ancklo/ChaosMagPy |

Most of these can be installed with:
```bash
pip install git+https://github.com/<org>/<repo>.git
```

---

## Usage

There is no unified entry point. The intended way to explore the code is to open the notebooks directly. Some notebooks depend on data files that are not included in this repository (observation data, model outputs) — if you need access to specific datasets, see the contact section below.

---

## Contact

This repository is not actively maintained, but if you have questions about the code or the underlying research, feel free to reach out.

**William [Last Name]**
📧 [your.email@example.com]

I'll do my best to respond, though reply times may be slow now that the thesis is submitted.

---

## Acknowledgements

This work was carried out at the Technical University of Denmark (DTU). Functions in `resolution.py` are adapted from the [EZIE repository](https://github.com/klaundal/ezie).
