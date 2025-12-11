# Quasi-Symmetric Nets: A Constructive Approach to the Equimodular Elliptic Type of Kokotsakis Polyhedra

This repository accompanies the paper  
**“Quasi-Symmetric Nets: A Constructive Approach to the Equimodular Elliptic Type of Kokotsakis Polyhedra.”**  
It provides constructive criteria, verified examples (closed-form and numerical), and code to **search, verify, construct, and visualize** flexible Kokotsakis polyhedra of **equimodular elliptic type** for the quadrangular base case ($n=4$).

## Video Abstract

[![Watch the video](https://img.youtube.com/vi/nnBfM4qHzR8/0.jpg)](https://www.youtube.com/watch?v=nnBfM4qHzR8)


The repository includes:
- **Symbolic verification notebooks** (Wolfram Mathematica) that check the existence criterion and the examples.
- **Interactive Python visualizations** (Matplotlib) that animate flexion in $\mathbb{R}^3$.
- **A numerical search & verification pipeline** implementing the algebraic conditions from the paper.
- **Appendix documentation** explaining the structure and usage of visualization scripts in each example folder.

---

## Repository structure

Top-level folders (as in this repo):

- **Criterion/**  
  `helper.nb` — Mathematica companion for the existence criterion.  
  `helper.pdf` — static print of the executed notebook.

- **Algorithm/**  
  `numerical_search.py` — multi-start root-finding with domain-aware initialization and a rigorous verification pipeline.

- **Example 1/**  
  `helper1.nb` / `helper1.pdf` — closed-form checks and flexion formulas (Example 1).  
  `visualization1.py` — interactive 3D viewer for Example 1.

- **Example 2/**  
  `helper2.nb` / `helper2.pdf` — closed-form checks (Example 2).  
  `visualization2.py` — interactive 3D viewer for Example 2.

- **Example 3/**  
  `helper3.nb` / `helper3.pdf` — numerical example verification (Example 3).  
  `visualization3.py` — interactive 3D viewer for Example 3.

- **Example 4/**  
  `helper4.nb` / `helper4.pdf` — numerical example verification (Example 4).  
  `visualization4.py` — interactive 3D viewer for Example 4.

- **Appendix/**  
  Documentation folder providing detailed explanations of the **visualization scripts** (`visualization1.py`–`visualization4.py`).

> The `*.pdf` files are static versions of the executed Mathematica notebooks for quick reading without Mathematica.

---

## Requirements

### Python (visualization & numerical search)
- Python ≥ 3.9  
- Packages:
  - `numpy`
  - `matplotlib`
  - `scipy` (root finding, optional but recommended)
  - `sympy` (optional, for algebraic helpers)

Create a virtual environment and install dependencies:
```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

python -m pip install -U pip numpy matplotlib scipy sympy
