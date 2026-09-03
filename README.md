# Balanced Weight Routing (BWR) + Verified Residual Routing (VRR)

Balanced Weight Routing (BWR) investigates whether heterogeneous language models can be dynamically allocated according to verified residual task difficulty, reducing aggregate inference compute while preserving verified task success.

The repository provides a research-grade experimental framework to empirically test whether a ladder of local Ollama models can match the verified quality of the strongest model while consuming substantially less compute.

> **Engineering Analogy Note:**
> BWR is inspired by mechanical balancing principles:
> $$\text{task load} - \text{allocated capability} = \text{residual}$$
> This is strictly an engineering and optimization analogy, **not** a claim that language model cognition literally obeys Newtonian force laws.

---

## 1. Research Question & Formal Optimization

We test the null hypothesis:

$$H_0: C_{\text{BWR}} \ge C_{\text{strongest}}$$

against the alternative hypothesis:

$$H_1: C_{\text{BWR}} < C_{\text{strongest}}$$

subject to approximately equal verified task quality:

$$Q_{\text{BWR}} \approx Q_{\text{strongest}}$$

### Principal Optimization Problem

$$\min_{\pi} \mathbb{E}_{\pi}[C_{\text{total}}] \quad \text{subject to} \quad P_{\pi}(\text{verified success}) \ge \tau$$

where:
* $\pi$: routing policy
* $C_{\text{total}}$: aggregate inference compute/cost
* $\tau$: required reliability threshold

The primary reported metric is the normalized cost reduction:

$$\text{CostSaving} = 1 - \frac{C_{\text{router}}}{C_{\text{strongest baseline}}}$$

---

## 2. Core Paradigm: Verification Over Self-Confidence

Traditional cascading approaches often rely on the model self-reporting its own uncertainty (e.g. *"I am confident"* or *"I cannot solve this"*). Under rigorous empirical testing on adversarial traps, **models frequently exhibit high confidence on hallucinated errors**.

BWR enforces a strict principle:

$$\text{Attempt cheaply} \longrightarrow \text{Externally verify} \longrightarrow \text{Measure residual failure} \longrightarrow \text{Isolate unresolved work} \longrightarrow \text{Escalate only what requires greater capability} \longrightarrow \text{Verify again}$$

Given external verifier $V(x, y_i) \in [0, 1]$, the observed residual error is:

$$R_{\text{obs}} = 1 - V(x, y_i)$$

---

## 3. Architecture & 5-Model Ladder

```
                                 [ Task Load D ]
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │    Router Policy π (FV-BWR / VRR / etc)   │
                  │    - Initial dispatch via objective J_i   │
                  │    - Closed-loop rebalancing on R_obs     │
                  └─────────────────────┬─────────────────────┘
                                        │ Dispatch
                                        ▼
                             ┌─────────────────────┐
                             │    Selected Model   │
                             │  (Local Ollama/Mock)│
                             └──────────┬──────────┘
                                        │ Attempt y_i
                                        ▼
                           ┌─────────────────────────┐
                           │   External Verifier V   │
                           │ (AST, SymPy, Mechanics) │
                           └────────────┬────────────┘
                                        │
                     ┌──────────────────┴──────────────────┐
                Passed (V=1)                          Failed (V<1)
                     │                                     │
                     ▼                                     ▼
             [ Verified Success ]                 [ Residual Extractor ]
             - Return Result                      - Extract Minimal Delta Δ
             - Record Telemetry                   - Context Reduction S_context
                                                  - Update D_{t+1} <- R_obs
```

### 5-Model Ladder Specification (`configs/models.yaml`)

Models are ordered conceptually from cheapest/weakest to strongest/most expensive, but capability is measured empirically rather than inferred from parameter count alone:

| Tier | Identifier | Default Ollama Tag | Context Window | Relative Cost Tier |
| :--- | :--- | :--- | :--- | :--- |
| $M_1$ | `model_1` | `qwen2.5:0.5b` | 4,096 | 1 (Micro) |
| $M_2$ | `model_2` | `qwen2.5-coder:3b` | 8,192 | 2 (Mini) |
| $M_3$ | `model_3` | `qwen3:4b-instruct` | 8,192 | 3 (Small) |
| $M_4$ | `model_4` | `qwen2.5:7b` | 16,384 | 4 (Medium) |
| $M_5$ | `model_5` | `llama3.1:8b` | 16,384 | 5 (Large/Strongest) |

---

## 4. Feature-Vector Balanced Weight Routing (FV-BWR)

Rather than treating routing as a coarse domain-level classifier ($\text{router} = f(\text{domain})$), **FV-BWR** formulates routing directly over multidimensional task requirement vectors:

$$\boxed{\mathbf{D} = [d_{\text{math}}, d_{\text{reasoning}}, d_{\text{code}}, d_{\text{language}}, d_{\text{mechanics}}, d_{\text{planning}}, d_{\text{trap}}] \in [0, 1]^7}$$

Two tasks nominally categorized as `mechanics` may impose completely different dimensional loads:
* **Numerical Mass Balancing Problem**: $\mathbf{D}_A = [0.80_{\text{math}}, 0.70_{\text{reasoning}}, 0.00_{\text{code}}, 0.20_{\text{language}}, 0.85_{\text{mechanics}}, 0.20_{\text{planning}}, 0.00_{\text{trap}}]$
* **Dynamic Simulation Script**: $\mathbf{D}_B = [0.60_{\text{math}}, 0.70_{\text{reasoning}}, 0.85_{\text{code}}, 0.40_{\text{language}}, 0.75_{\text{mechanics}}, 0.60_{\text{planning}}, 0.00_{\text{trap}}]$

### 4.1 Empirical Capability Vectors ($\mathbf{C}_i$)
Each model $M_i$ in the ladder possesses an empirical verified capability vector calibrated on training tasks:

$$\mathbf{C}_i = [c_{i,\text{math}}, c_{i,\text{reasoning}}, c_{i,\text{code}}, c_{i,\text{language}}, c_{i,\text{mechanics}}, c_{i,\text{planning}}, c_{i,\text{trap}}] \in [0, 1]^7$$

### 4.2 Component-Wise Deficit & Imbalance Norm
The remaining unabsorbed capability deficit for model $M_i$ is computed component-wise:

$$\mathbf{R}_i = (\mathbf{D} - \mathbf{C}_i)_+ = \max(0, \mathbf{D} - \mathbf{C}_i)$$

The scalar imbalance norm is the weighted Euclidean norm:

$$R_i = \|\mathbf{W} (\mathbf{D} - \mathbf{C}_i)_+\|_2 = \sqrt{\sum_{k=1}^7 w_k \cdot \max(0, d_k - c_{ik})^2}$$

### 4.3 Joint Balancing Objective ($J_i$)
The router selects the model $i^*$ that minimizes the joint residual imbalance and normalized execution cost:

$$J_i = \lambda_R R_i^2 + \lambda_K K_i \implies i^* = \arg\min_{i} J_i$$

where $K_i$ is the empirical cost proxy of model $M_i$, $\lambda_R$ penalizes capability deficits, and $\lambda_K$ penalizes excessive compute expenditure.

### 4.4 Closed-Loop Dynamic Verifier Feedback
When an external verifier $V$ detects an error, it extracts the deterministic failure modes (e.g., AST syntax error $\to$ code deficit, numerical delta $\to$ math deficit, unbalance residue $\to$ mechanics deficit, hallucination trigger $\to$ trap deficit).

The observed residual vector $\mathbf{R}_{\text{obs}}$ updates the dynamic demand:

$$\mathbf{D}_0 \xrightarrow{\text{select } M_i} \text{Execute} \xrightarrow{\text{Verifier } V} \mathbf{R}_{\text{obs}} \implies \mathbf{D}_{t+1} \leftarrow \mathbf{R}_{\text{obs}} \xrightarrow{\text{rebalance}} \text{select } M_j \dots$$

---

## 5. 2×2 Factorial Ablation Matrix

| Configuration | Model Selection Policy | Context Passed on Escalation | Focus / Ablation |
| :--- | :--- | :--- | :--- |
| **Policy A** | Fixed Cascade ($M_1 \to M_2 \to \dots \to M_5$) | Full Original Context ($x$) | Baseline Cascading Control |
| **Policy B (VRR)** | Fixed Cascade ($M_1 \to M_2 \to \dots \to M_5$) | Minimal Verified Residual ($\Delta_t$) | Tests Context Minimization Effect |
| **Policy C (Coarse BWR)** | BWR Empirical Efficiency $\eta_i$ | Full Original Context ($x$) | Tests Domain-level Skipping |
| **Policy D (BWR + VRR)** | BWR Empirical Efficiency $\eta_i$ | Minimal Verified Residual ($\Delta_t$) | Full Dual-Mechanism System |
| **Policy E (FV-BWR Full)** | Feature-Vector BWR $f(\mathbf{D})$ | Full Original Context ($x$) | Tests 7D Vector Load Balancing |
| **Policy F (FV-BWR + VRR)** | Feature-Vector BWR $f(\mathbf{D})$ | Minimal Verified Residual ($\Delta_t$) | Closed-Loop Vector Balancing + VRR |

---

## 6. Generalization Benchmarks on Unseen Tasks

In `experiments/exp07_feature_vector_bwr.py`, empirical capability vectors $\mathbf{C}_i$ are calibrated on a **60% Training Split** and evaluated on a **40% Unseen Test Split** stratified across all 5 benchmark domains:

| Routing Policy | Test Success ($Q$) | Avg Tokens | Avg Cost ($K$) | Cost Saving ($S_C$) | Token Saving ($S_T$) | $M_5$ Util % | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$M_5$ Only (Baseline)** | 80.0% | 202.5 | 0.03447 | 0.0% (Base) | 0.0% (Base) | 100.0% | 1.38s |
| **Policy A: Fixed Cascade** | 80.0% | 228.1 | 0.01008 | **+70.8%** | -12.6% | 0.0% | 0.65s |
| **Policy B: VRR (Fixed + Residual)** | 80.0% | 307.8 | 0.01397 | **+59.5%** | -52.0% | 0.0% | 0.77s |
| **Policy C: Coarse BWR (Domain)** | 70.0% | 260.0 | 0.02514 | +27.1% | -28.4% | 50.0% | 0.98s |
| **Policy E: FV-BWR (Full)** | 80.0% | 234.7 | 0.01552 | **+55.0%** | -15.9% | 10.0% | 0.83s |
| **Policy F: FV-BWR + VRR** | 80.0% | 235.0 | 0.01082 | **+68.6%** | -16.0% | **0.0%** | 0.60s |

### Key Scientific Findings:
1. **Generalization Over Coarse Domain Routing**: Coarse domain routing achieved only $70.0\%$ accuracy on unseen tasks due to within-domain heterogeneity. Feature-Vector BWR matched the full $80.0\%$ baseline accuracy while reducing normalized compute cost by **$+68.6\%$**.
2. **Zero Strongest-Model Bottleneck**: Policy F resolved all unseen tasks using balanced smaller models ($M_1\dots M_4$), achieving **$0.0\% \ M_5 \text{ utilization}$**.
3. **Verification Superiority Over Confidence**: Self-reported confidence router collapsed to $0.0\%$ on adversarial trap tasks, proving that deterministic external verification is mandatory for robust cost reduction.

---

## 7. Domain Verifiers

* **Code (`verifiers/code_verifier.py`)**: Sandboxed execution with timeout, AST syntax validation, unit test assertion runner, stdout verification, and structured exception isolation (`syntax_error`, `runtime_error`, `unit_test_failure`, `timeout_error`).
* **Mathematics (`verifiers/math_verifier.py`)**: Exact SymPy symbolic simplification ($\text{simplify}(y_{\text{pred}} - y_{\text{true}}) = 0$), numerical tolerance verification ($|y_{\text{pred}} - y_{\text{true}}| \le 10^{-5}$), and algebraic system consistency.
* **Mechanics (`verifiers/mechanics_verifier.py`)**:
  * Rotating mass single/multi-plane dynamic balancing ($\sum m r \cos\theta = 0, \sum m r \sin\theta = 0, \sum m r l \cos\theta = 0, \sum m r l \sin\theta = 0$).
  * Static equilibrium ($\sum F_x = 0, \sum F_y = 0, \sum M = 0$).
  * Natural vibrations ($\omega_n = \sqrt{k/m}$).
  * Statically determinate beam reactions and First Law thermodynamic conservation ($Q - W = \Delta U$).
* **Structured Data & Hallucination Traps (`verifiers/structured_verifier.py`)**: Schema compliance and adversarial trap detection (buoyancy omission, extraneous radical roots, velocity/pressure sign inversion, Python mutable default argument traps).

---

## 8. Quickstart & Experiment Execution

### Installation & Environment Setup
```bash
git clone https://github.com/ChamanPrakashKanth/Balanced-Weight-Routing.git
cd Balanced-Weight-Routing
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

### Run Unit Test Suite
```bash
pytest tests/ -v
```

### Run Feature-Vector BWR Generalization Experiment (Mock Engine)
```bash
python experiments/exp07_feature_vector_bwr.py --seed 42 --train-ratio 0.60
```

### Run Full Master Experiment Suite (All Policies + Ablations)
```bash
python experiments/run_all_experiments.py
```

### Run Live Local Ollama Benchmarks
Ensure Ollama server is running locally with installed models (`qwen2.5:0.5b`, `qwen2.5-coder:3b`, `qwen3:4b-instruct`, `qwen2.5:7b`, `llama3.1:8b`):
```bash
python experiments/exp00_individual_models.py --live
python experiments/exp07_feature_vector_bwr.py --live
```

---

## 9. Critical Falsification Criteria & Limitations

The research harness is explicitly designed to identify conditions where BWR fails:

1. **Verifier Overhead vs Savings**: If external verification latency or execution costs exceed the token savings of routing to smaller models, simple single-model dispatch remains optimal.
2. **Context Fragmentation in Residuals**: If isolating the error strips crucial problem background, the larger model may fail to repair the code.
3. **Cascading Latency**: Multi-hop escalations ($M_1 \to M_3 \to M_5$) increase interactive turnaround time compared to direct single-shot dispatch.
4. **False Acceptance Vulnerability**: Unverified confidence routing fails catastrophically on adversarial traps, accepting invalid answers 100% of the time when models hallucinate with high self-reported certainty.

---

## 10. License

MIT License. See [LICENSE](LICENSE) for details.
