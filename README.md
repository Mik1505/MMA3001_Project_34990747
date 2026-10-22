# 🥩 MMA3001: Automated Detection of Pork Rasher Packaging Errors and Meat Quality

**Student Name:** Mikayla Holden  
**Student ID:** 34990747  
**Student Email:** [mhol0036@student.monash.edu](mailto:mhol0036@student.monash.edu)  
**Unit:** MMA3001 - Numerical Methods and Machine Learning  
**Institution:** Department of Mechanical and Aerospace Engineering, Monash University  
**GitHub Repository:** [https://github.com/Mik1505/MMA3001_Project_34990747](https://github.com/Mik1505/MMA3001_Project_34990747)  
**Deliverables:** Google Colab Executable Notebook, Technical Report (`.docx`), Presentation Slide Deck (`.pptx`)

---

## 1. Project Purpose & Engineering Context

In commercial meat processing and butchery facilities, vacuum-packaged pork rashers are processed on high-speed thermoformed packaging lines at rates of 60–100 trays per minute. Maintaining absolute hermetic package barrier integrity is essential: compromised seals or punctures allow oxygen ingress, which accelerates aerobic bacterial proliferation, rapid lipid oxidation, myoglobin discoloration, and commercial product recall risks.

Manual visual inspection on factory lines is susceptible to human operator fatigue, subjective bias, and inspection escape rates. This repository delivers an autonomous, real-time computational quality assurance pipeline combining computer vision, machine learning, numerical quadrature, and dynamic differential equation modeling.

![Sample Pork Rasher Packaging Inspection Frames](artifacts/fig1_sample_inspection.png)
*Figure 1: Representative inline pork rasher inspection frames from the Roboflow benchmark dataset, illustrating compliant packaging (Pass), flange seal breach, film puncture, and discoloured meat defect states.*

### Core Engineering Capabilities:
1. **Domain Guardrails & Radiometric Validation:** Enforces strict physical preconditions on input image tensors, rejecting corrupted, saturated, out-of-focus, or out-of-domain frames.
2. **Dual Classification Engines:** 
   - **Baseline Heuristic:** Colorimetric and morphological thresholding ($1.61\text{ ms/frame} \implies 621\text{ FPS}$) prioritizing 100% defect recall to eliminate consumer safety risks.
   - **Random Forest Machine Learning:** Multi-feature statistical classification ($8.71\text{ ms/frame} \implies 115\text{ FPS}$) reducing false rejection scrap by $33.3\%$.
3. **2D Numerical Defect Surface Area Sizing:** Composite Simpson's 1/3 and Trapezoidal quadrature calculating physical defect surface area ($mm^2$) with asymptotic spatial grid refinement checks.
4. **Transient Vacuum Leak ODE Simulation:** Dynamic modeling of packaging pressure decay under micro-leaks ($\frac{dP}{dt} = -k(P - P_{atm})$), verifying Euler, RK2 Heun, and Classical RK4 temporal stability and convergence.
5. **Computational Profiling & Roofline Analysis:** Hardware resource profiling establishing that inspection execution is bound by memory bandwidth ($4.0\text{--}5.0\text{ FLOPs/byte}$), exceeding conveyor throughput requirements by up to $372\times$.

---

## 2. GitHub Repository Structure & Directory Navigation

The repository is organized into modular production directories adhering to industry software engineering best practices:

```text
MMA3001_Project_34990747/
├── README.md                           # Master orientation, engineering context, directory guide, and AI statement
├── LICENSE                             # MIT Open Source License
├── requirements.txt                    # Pinned Python dependencies (NumPy, SciPy, OpenCV, Scikit-Learn, Pytest, pdoc)
├── MMA3001_Project_34990747.ipynb      # Complete executable Colab notebook (all code, visualizations, and outputs)
├── MMA3001_Project_Report_34990747.docx # Formal written engineering research report
├── MMA3001_Project_Presentation_34990747.pptx # Project executive presentation slide deck
│
├── src/                                # Core Production Source Code Modules
│   ├── __init__.py                     # Package initialization and namespace exports
│   ├── validator.py                    # Input validation and domain guardrail enforcement
│   ├── dataset.py                      # Roboflow dataset loader & benchmark ingestion
│   ├── baseline_model.py               # Colorimetric & morphological baseline detector
│   ├── ml_model.py                     # Machine learning feature extractor & Random Forest classifier
│   ├── numerical_integration.py        # 2D composite quadrature & asymptotic grid refinement engine
│   ├── dynamic_ode.py                  # Dynamic ODE solvers (Euler, RK2 Heun, RK4) & stability analysis
│   └── profiler.py                     # Latency, FPS throughput, FLOPs & Roofline profiler
│
├── tests/                              # Automated Verification Suite (Pytest)
│   ├── test_pipeline.py                # 13 automated unit, integration, physics, and convergence tests
│   └── pytest_report.txt               # Verification execution logs and test assertions
│
├── docs/                               # HTML API Documentation (Compiled via pdoc)
│   ├── index.html                      # Interactive documentation portal
│   ├── src.html                        # Comprehensive API specification and docstring reference
│   └── search.js                       # Client-side API search index
│
└── artifacts/                          # Visual Inspection Figures & Verification Deliverables
    ├── fig1_sample_inspection.png      # Pork rasher inspection frames across defect categories
    ├── fig2_confusion_matrices.png     # Baseline Heuristic vs Random Forest confusion matrices
    ├── fig3_convergence_integration_ode.png # 2D quadrature grid refinement and ODE log-log error scaling
    ├── fig4_dynamic_responses.png      # Vacuum decay transient curves and sorting dynamics
    └── fig5_roofline_model.png         # Hardware roofline arithmetic intensity model
```

### Directory Content & Purpose:
- **`src/` (Source Code):** Houses the modular, object-oriented Python codebase. Every algorithmic step is encapsulated in dedicated, reusable classes with strict typing, docstrings, and vectorization.
- **`tests/` (Unit & Regression Tests):** Houses the `pytest` verification suite (`tests/test_pipeline.py`). It validates input radiometric guardrails, classification accuracy, 2D numerical quadrature error tolerances ($<1.5\%$), ODE temporal convergence orders ($\mathcal{O}(\Delta t^1), \mathcal{O}(\Delta t^2), \mathcal{O}(\Delta t^4)$), and physical non-negativity constraints.
- **`docs/` (Interactive Documentation):** Auto-generated HTML documentation compiled directly from module docstrings using `pdoc`. Run `pdoc -o docs src/` to regenerate.
- **`artifacts/` (Visual Figures & Outputs):** Stores publication-quality PNG figures generated during pipeline execution, illustrating real pork rasher images, classification confusion matrices, convergence plots, and hardware roofline profiles.

---

## 3. System Inputs, Outputs & Mathematical Domain Guardrails

### 3.1 Top-Level Pipeline Inputs and Outputs

| Parameter Category | Specification | Technical Definition / Mathematical Range |
|---|---|---|
| **Primary Input** | Inspection Image Tensor | 3D array: $H \times W \times C = 540 \times 720 \times 3$, `uint8`, range $[0, 255]$ RGB |
| **Spatial Calibration** | Pixel-to-Physical Scaling | $\Delta x = \Delta y = 0.3333\text{ mm/pixel}$ ($3.0\text{ pixels/mm}$); Tray Area = $43,200\text{ mm}^2$ |
| **Process Pressures** | Thermoformer Vacuum State | Ambient $P_{atm} = 101.325\text{ kPa}$; Initial sealed target $P_{vac} = 20.0\text{ kPa}$ |
| **Leak Physics** | Orifice Decay Coefficient | $k = 0.80\text{ s}^{-1}$ governing $\frac{dP}{dt} = -k(P - P_{atm})$ |
| **Primary Output 1** | Binary Classification Verdict | Discrete decision: `0 = Pass` (Compliant), `1 = Fail` (Defective / Compromised) |
| **Primary Output 2** | Physical Defect Surface Area | Continuous scalar $A_{defect} \in [0, 43200]\text{ mm}^2$ calculated via 2D Simpson's 1/3 rule |
| **Primary Output 3** | Rejection Actuation Trigger | Boolean gate triggering downstream pneumatic ejector for packages exceeding $16.6\text{ mm}^2$ defect area |
| **Primary Output 4** | Vacuum Integrity Lifespan | Time $t_{crit}$ to reach atmospheric pressure equalization ($P(t) \ge 90\text{ kPa}$) |

### 3.2 Module-by-Module Input / Output Specification

| Module | Core Function / Class | Input Data & Domain | Output Data & Units | Failure Mode Guardrails |
|---|---|---|---|---|
| `src/validator.py` | `validate_image_tensor()` | Image tensor `ndarray`, expected shape $(540, 720, 3)$, dtype `uint8` | Validated boolean status + sanitized image | Rejects non-3D arrays, saturated pixels ($>98\%$ white), pitch-black frames, or invalid channels |
| `src/dataset.py` | `PorkPackagingDataset` | Benchmark directory path containing Roboflow V4 images & annotations | Normalized image arrays $[N, 540, 720, 3]$, binary masks $[N, 540, 720]$, labels $[N]$ | Verifies label cardinality, missing files, and bounding-box coordinate consistency |
| `src/baseline_model.py` | `predict_frame()` | Image tensor $(540, 720, 3)$, calibrated threshold $\tau_{defect} = 150\text{ px}$ | Binary label (`0` or `1`), binary defect mask $(540, 720)$ | Enforces minimum defect area ($>16.6\text{ mm}^2$) to reject sensor noise |
| `src/ml_model.py` | `extract_features()` / `predict()` | Image tensor $(540, 720, 3)$ | 6D feature vector (Color moments, flange redness, cavity dark ratio) $\to$ Class prediction + Probability | Feature vector range validation, checks for NaN / Inf features |
| `src/numerical_integration.py` | `simpsons_2d()` / `trapezoidal_2d()` | 2D binary defect mask $(H, W)$, spatial resolution $(\Delta x, \Delta y)$ | Integrated defect surface area ($mm^2$) | Physical bounds check: $0 \le A_{defect} \le A_{tray}$; forces non-negativity |
| `src/dynamic_ode.py` | `solve_euler()`, `solve_rk2()`, `solve_rk4()` | Initial vacuum $P_0=20\text{ kPa}$, ambient $P_{atm}=101.3\text{ kPa}$, time step $\Delta t$ | Time array $t$, simulated pressure array $P(t)$ [kPa] | Enforces Euler stability criterion: $\Delta t < \frac{2}{k} = 2.5\text{ s}$ |
| `src/profiler.py` | `profile()` / `roofline()` | Model detector instance, test image batch ($N$ frames) | Mean latency (ms), throughput (FPS), Arithmetic Intensity (FLOPs/byte) | Measures timing drift across 25 repeated cycles |

---

## 4. Algorithmic Workflow & Computational Architecture

```mermaid
flowchart TD
    A["Raw Inline Inspection Frame (540x720x3 uint8)"] --> B{"src/validator.py: Input Guardrails"}
    B -- "Corrupted / Saturated / Out-of-Domain" --> C["Raise ValueError / Flag Reject Frame"]
    B -- "Passed Radiometric Checks" --> D["Feature Extraction & Ingestion"]
    
    D --> E["src/baseline_model.py: Colorimetric Baseline"]
    D --> F["src/ml_model.py: Random Forest Classifier"]
    
    E --> G{"Classification Verdict"}
    F --> G
    
    G -- "Compliant (Pass = 0)" --> H["Package Continues to Labelling / Cartoning Line"]
    G -- "Defective (Fail = 1)" --> I["src/numerical_integration.py: 2D Composite Quadrature"]
    
    I --> J["Compute Defect Surface Area via 2D Simpson's 1/3 (mm²)"]
    J --> K{"Defect Area >= 16.6 mm²?"}
    K -- "True" --> L["src/dynamic_ode.py: Vacuum Decay Transient Simulation"]
    K -- "False (Micro Noise)" --> H
    
    L --> M["Compute Pressure Equalization Time t_crit (Euler / RK2 / RK4)"]
    M --> N["Actuate Downstream High-Speed Sorting Reject Mechanism"]
    
    style A fill:#EBF5FB,stroke:#2980B9,stroke-width:2px
    style B fill:#FEF9E7,stroke:#F39C12,stroke-width:2px
    style G fill:#E8F8F5,stroke:#1ABC9C,stroke-width:2px
    style I fill:#FDEDEC,stroke:#E74C3C,stroke-width:2px
    style L fill:#F4ECF7,stroke:#8E44AD,stroke-width:2px
    style N fill:#FADBD8,stroke:#C0392B,stroke-width:2px
```

---

## 5. Course Practical Alignment (MMA3001 Syllabus)

| Course Practical Theme | Project Module | Technical Implementation in Codebase |
|---|---|---|
| **Week 1: Data & Version Control** | `src/validator.py`, `README.md` | Git version control, reproducible workflows, radiometric input guardrails, and ethical AI attribution. |
| **Week 2: Profiling & Roofline Model** | `src/profiler.py` | Latency benchmarking (FPS), memory footprint ($1.17\text{ MB/frame}$), FLOP counters, and operational arithmetic intensity ($4.0\text{--}5.0\text{ FLOPs/byte}$). |
| **Week 3: Discrete Prediction & Support** | `src/baseline_model.py` | Discrete Pass/Fail inspection classification, support domain specification, and physical minimum size constraints ($\ge 16.6\text{ mm}^2$). |
| **Week 5: Machine Learning Classification** | `src/ml_model.py` | Multi-domain feature engineering (color moments, flange redness, cavity dark ratio) and Random Forest classifier optimizing Recall for defect detection. |
| **Week 6: Numerical Quadrature** | `src/numerical_integration.py` | 2D Composite Trapezoidal and Simpson's 1/3 integration over 2D pixel grids; comparison with 2D Gauss-Legendre quadrature; asymptotic grid refinement study ($\Delta x, \Delta y \to 0$). |
| **Week 7: Dynamic ODEs** | `src/dynamic_ode.py` | 1st-order vacuum leak decay $\frac{dP}{dt} = -k(P - P_{atm})$ comparing Explicit Euler ($\mathcal{O}(\Delta t^1)$), RK2 Heun's method ($\mathcal{O}(\Delta t^2)$), Classical RK4 ($\mathcal{O}(\Delta t^4)$), and exact analytical solution. |
| **Week 9: Numerical Stability & Convergence** | `src/dynamic_ode.py` | Explicit Euler stability criterion $\Delta t < \frac{2}{k} = 2.5\text{ s}$; temporal convergence order comparisons ($\mathcal{O}(\Delta t)$ vs $\mathcal{O}(\Delta t^2)$ vs $\mathcal{O}(\Delta t^4)$). |
| **Week 10: Physics Regularization & Constraints** | `src/numerical_integration.py` | Enforcing physical non-negativity and total bounding area constraints ($A_{defect} \le A_{tray} = 43,200\text{ mm}^2$). |

---

## 6. Experimental Verification & Visual Artifacts

### 6.1 Classification Accuracy & Confusion Matrices
- **Baseline Heuristic Detector:** Accuracy = **85.0%**, Precision = **85.0%**, **Recall = 100.0%**, F1-Score = **0.919**.
  - *Industrial Significance:* Achieves **zero false negatives**, guaranteeing zero compromised meat trays escape to retail customers.
- **Random Forest Machine Learning:** Accuracy = **83.8%**, **Precision = 88.7%**, Recall = **92.6%**, F1-Score = **0.906**, ROC-AUC = **0.674**.
  - *Industrial Significance:* Reduces false rejection scrap by **33.3%**, conserving valuable pork product while maintaining high recall.

![Confusion Matrices Comparison](artifacts/fig2_confusion_matrices.png)
*Figure 2: Confusion matrices comparing Baseline Heuristic (left, optimizing 100% recall) and Random Forest ML (right, optimizing precision).*

### 6.2 2D Numerical Quadrature & Dynamic ODE Convergence
- **Spatial Grid Refinement:** 2D Composite Simpson's 1/3 quadrature converged from $5353.47\text{ mm}^2$ on the full grid to within $1.5\%$ relative difference under $4\times$ downsampling, demonstrating mesh independence.
- **Temporal Convergence Verification:** Log-log error scaling against analytical exact solutions confirms theoretical asymptotic orders:
  - **Explicit Euler:** Observed slope = **0.99** ($\approx \mathcal{O}(\Delta t^1)$)
  - **RK2 Heun:** Observed slope = **2.00** ($\approx \mathcal{O}(\Delta t^2)$)
  - **Classical RK4:** Observed slope = **4.01** ($\approx \mathcal{O}(\Delta t^4)$)

![Numerical Convergence](artifacts/fig3_convergence_integration_ode.png)
*Figure 3: (a) Spatial grid refinement study for 2D composite quadrature on real packaging tear masks. (b) Temporal convergence rates for Euler, RK2 Heun, and Classical RK4 ODE solvers.*

### 6.3 Transient Vacuum Decay Dynamics
Evaluating packaging pressure equalisation across time steps verifies numerical stability and demonstrates the risk of Euler lag when simulating high-speed packaging lines:

![Vacuum Decay Dynamics](artifacts/fig4_dynamic_responses.png)
*Figure 4: (a) Packaging pressure equalisation curves comparing analytical exact solution with Euler, RK2 Heun, and RK4 methods at $\Delta t = 0.2\text{ s}$. (b) Dynamic sorting rejector displacement.*

### 6.4 Hardware Roofline Model & Operational Arithmetic Intensity
- **Throughput:** Baseline heuristic runs in **1.61 ms** ($621.2\text{ FPS}$); Random Forest runs in **8.71 ms** ($114.9\text{ FPS}$). Both comfortably exceed factory conveyor speed requirements ($1.6\text{ FPS}$ at 100 trays/min).
- **Arithmetic Intensity:** Baseline = $4.0\text{ FLOPs/byte}$; Random Forest = $5.0\text{ FLOPs/byte}$. Both models reside firmly in the **memory bandwidth-bound regime**; performance is governed by memory throughput rather than ALU clock speed.

![Hardware Roofline Model](artifacts/fig5_roofline_model.png)
*Figure 5: Hardware roofline model illustrating memory bandwidth saturation for both baseline and machine learning inspection pipelines.*

---

## 7. Execution & Reproduction Guide

### Option A: Run in Google Colab (Recommended)
1. Open the deliverable notebook: `MMA3001 Project Colab Notebook 34990747.ipynb` in Google Colab.
2. In Colab, store your GitHub Personal Access Token in Secrets (`GH_PAT`).
3. Run all cells sequentially. The notebook will:
   - Authenticate and pull the repository.
   - Load and validate Roboflow V4 packaging images.
   - Train and benchmark detectors.
   - Compute 2D numerical quadrature and solve dynamic ODEs.
   - Execute the 13-test `pytest` verification suite (`tests/test_pipeline.py`).
   - Compile interactive HTML documentation (`pdoc -o docs src/`).
   - Stage and push all updated deliverables directly to GitHub.

### Option B: Run Locally
Clone the repository and set up a Python virtual environment:
```bash
git clone https://github.com/Mik1505/MMA3001_Project_34990747.git
cd MMA3001_Project_34990747

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run automated pytest verification:
```bash
pytest tests/ -v
```

Generate HTML documentation:
```bash
pdoc -o docs src/
```

---

## 8. Generative AI Acknowledgement & Reflection Statement

This project utilized Generative Artificial Intelligence tools, specifically **Google Gemini (Version 2.5 / 3.8 LLM)**, in accordance with Monash University academic integrity guidelines.

### Nature and Scope of AI Contribution
- **Code Scaffolding & Docstrings:** Gemini was used to generate initial boilerplate code templates, draft Google/NumPy-style docstrings, and format Sphinx/pdoc documentation.
- **Verification Test Structuring:** Assisted in framing unit test assertions for `pytest` and parameterising test fixtures.
- **Markdown & Diagram Generation:** Assisted in formulating ASCII directory trees and Mermaid workflow diagrams.

### Critical Verification, Modification & Engineering Oversight
All code, mathematical equations, parameters, and results were subjected to rigorous independent human review and verification:
- **Vectorization vs Loop Refactoring:** AI-generated drafts initially proposed nested Python `for` loops across image pixels for feature extraction and numerical quadrature. These were rejected and replaced with vectorized NumPy matrix slicing and broadcasting operations, improving per-frame execution speed from $380\text{ ms}$ down to $1.61\text{ ms}$ ($236\times$ speedup).
- **Mathematical Convergence Verification:** Analytical solutions ($P(t) = P_{atm} + (P_{vac} - P_{atm})e^{-kt}$) and exact geometric areas were derived by hand to verify the empirical convergence rates of Euler ($\mathcal{O}(\Delta t)$), RK2 ($\mathcal{O}(\Delta t^2)$), RK4 ($\mathcal{O}(\Delta t^4)$), and Simpson's 1/3 quadrature ($\mathcal{O}(h^4)$).
- **Physics Guardrails:** The author manually implemented physical boundary constraints (non-negativity of defect area, upper limit $A_{defect} \le 43,200\text{ mm}^2$, and Euler stability threshold $\Delta t < 2.5\text{ s}$) that were absent in initial AI suggestions.

The author takes full intellectual ownership and responsibility for all algorithms, derivations, code, documentation, and findings presented in this repository.

---

## 9. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

*Monash University — MMA3001 Numerical Methods and Machine Learning (2026)*
