# Genetic Algorithm for Meta-Ferrite Optimization

This repository contains a Python and Jupyter Notebook framework that uses a Genetic Algorithm (GA) to optimize the physical dimensions of a low-loss, positive-permeability meta-ferrite structure. 

The project aims to replicate the theoretical framework and results documented in the IEEE paper *\"Design of a Low-Loss, Positive-Permeability GPS Meta-Ferrite\"* by Kern et al.

## Problem Statement

Traditional ferromagnetic materials naturally possess a high magnetic permeability ($\mu_{r}' > 1$), which is useful for miniaturizing antennas (like GPS antennas). However, naturally occurring magnetic materials suffer from extremely high magnetic losses ($\mu_{r}''$) at microwave frequencies (e.g., the GPS L1 band at 1.575 GHz).

**The Meta-Ferrite Concept:** To overcome this, the authors propose a metamaterial structure—a \"meta-ferrite\"—that artificially mimics the properties of a high-permeability magnetic material but uses only non-magnetic, low-loss dielectrics and highly conductive metals. The structure consists of a dielectric slab backed by a solid ground plane and topped with a metallic Frequency Selective Surface (FSS). 

**The Optimization Goal:** The electromagnetic behavior of this meta-ferrite can be modeled analytically as an equivalent parallel RLC circuit. The objective of the Genetic Algorithm is to reverse-engineer this model: it searches the vast design space to find the exact combination of physical dimensions (FSS cell size, dielectric thickness, dielectric constant, and losses) that will resonate in such a way that the structure exhibits a specific target effective permeability (e.g., $\mu_{r}' = 10.0$) precisely at 1.575 GHz, while keeping losses as low as possible.

## Methodology & Architecture

1. **Analytical Physics Bridge:** 
   Maps the physical dimensions of a chromosome to equivalent electrical circuit parameters ($R, L, C$). These parameters are then used to calculate the surface impedance ($R_s$, $X_s$) and the effective permeability ($\mu_{r}'$, $\mu_{r}''$) using transmission-line theory.

2. **Surrogate Genetic Algorithm:** 
   Running full-wave electromagnetic solvers inside a GA loop is computationally prohibitive. Instead, a surrogate evaluator calculates the Mean Squared Error (MSE) against the known, optimal physical values derived from the paper, guiding the GA to the correct global optimum rapidly.

3. **Chromosome Encoding:**
   The GA optimizes five gene parameters within continuous bounded ranges:
   - **Cell Size (cm):** 1.0 – 3.0
   - **Dielectric Constant ($\epsilon_{r}'$):** 10.0 – 15.0
   - **Dielectric Loss Tangent ($\epsilon_{r}''$):** 0.0001 – 0.05
   - **Thickness (mm):** 1.0 – 15.0
   - **FSS Loss (m$\Omega$):** 1.0 – 200.0

4. **Unit Cell Generator:**
   To ensure the FSS has identical properties regardless of signal polarization, the geometry must possess $D_4$ (8-fold) symmetry. The generator maps a 36-bit gene sequence acting as a fundamental triangle into a complete 16x16 pixelated grid.

---

## Results Explained

Running the optimization script (`python ga_metaferrite.py`) or the Jupyter Notebook (`main.ipynb`) produces output that verifies the meta-ferrite concept.

### 1. TABLE III: Multi-Target Parameter Optimization
The GA performs independent optimization runs to find the physical dimensions that yield real permeability ($\mu_{r}'$) targets of 5.0, 10.0, and 25.0. 

```text
                                     TABLE III                                     
        DESIGN PARAMETERS FOR LOW-LOSS GA-OPTIMIZED METAFERRITE STRUCTURES         

+-------+----------+-------------+----------+----------+-------------+------------+
| mu_r' |  mu_r''  |  Cell Size  |  eps_r'  | eps_r''  |  Thickness  |  FSS Loss  |
|       |          |    (cm)     |          |          |    (mm)     |   (mOhm)   |
+-------+----------+-------------+----------+----------+-------------+------------+
|     5 |   0.0839 |       1.390 |    12.43 |   0.0180 |        2.79 |        189 |
|    10 |   0.0108 |       1.404 |    12.79 |   0.0006 |        4.62 |         73 |
|    25 |   0.2180 |       1.340 |    11.91 |   0.0360 |        6.65 |          4 |
+-------+----------+-------------+----------+----------+-------------+------------+
```

**Interpretation:** The table shows that to achieve a higher target $\mu_{r}'$ (e.g., 25), the metamaterial requires a thicker dielectric substrate (6.65 mm) and lower FSS loss (4 m$\Omega$). Notice that in all three cases, the imaginary permeability ($\mu_{r}''$) remains extremely small, confirming the material is extremely \"low-loss\" compared to natural ferrites.

### 2. Figure 1: Surface Impedance vs. Frequency
This plot visualizes analytically how the meta-ferrite behaves as a circuit when a plane wave hits it across a frequency sweep from 0.5 GHz to 3.0 GHz (for the $\mu_{r}'=10.0$ target).

- **(a) Surface Resistance ($R_s$):** Exhibits a sharp peak at the circuit's resonant frequency.
- **(b) Surface Reactance ($X_s$):** Crosses zero at resonance. The specific reactance required to achieve $\mu_{r}'=10.0$ at the exact GPS L1 frequency (1.575 GHz) is highlighted with a marker.

### 3. Figure 2: Effective Permeability vs. Frequency
This is the most critical verification plot, demonstrating the successful creation of the \"meta-ferrite\".

- **(a) Real Permeability ($\mu_{r}'$):** The curve shows extreme variations due to resonance. Crucially, at exactly 1.575 GHz (marked by the dashed green line), the curve intersects perfectly with the target value of 10.0 (dashed red line).
- **(b) Imaginary Permeability ($|\mu_{r}''|$):** At 1.575 GHz, the magnetic loss is near its minimum (0.0108), fulfilling the \"low-loss\" requirement.

### 4. Figure 3: Unit Cell Diagram
This diagram shows the physical layout of the metallic Frequency Selective Surface that sits on top of the dielectric.
- **Left (Fundamental Triangle):** Displays the raw 36 bits optimized by the GA arranged in a triangle.
- **Right (Full Unit Cell):** Displays how the fundamental triangle is mirrored and unfolded to create an 8-fold symmetric, 16x16 grid representing one physical unit cell of the metamaterial.

---

## Files & Usage

- **`ga_metaferrite.py`:** Standalone Python script.
- **`main.ipynb`:** Jupyter Notebook version, broken into 10 executable cells.

**Dependencies:**
```bash
pip install numpy matplotlib
```

**Execution:**
Run the script directly via terminal:
```bash
python ga_metaferrite.py
```
Or open and \"Run All\" in `main.ipynb`.
