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
                  │    Router Policy π (BWR / VRR / etc)      │
                  │    - Initial dispatch via efficiency η_i  │
                  │    - Marginal escalation E_ij & skipping  │
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
                                                  - Select Next Model or Stop
```

### 5-Model Ladder Specification (`configs/models.yaml`)

Models are ordered conceptually from cheapest/weakest to strongest/most expensive, but capability is measured empirically rather than inferred from parameter count alone:

| Tier | Identifier | Default Ollama Tag | Context Window | Relative Cost Tier |
| :--- | :--- | :--- | :--- | :--- |
| $M_1$ | `model_1` | `qwen2.5:0.5b` | 4,096 | 1 (Micro) |
| $M_2$ | `model_2` | `llama3.2:1b` | 8,192 | 2 (Mini) |
| $M_3$ | `model_3` | `qwen2.5:3b` | 8,192 | 3 (Small) |
| $M_4$ | `model_4` | `mistral:7b` | 16,384 | 4 (Medium) |
| $M_5$ | `model_5` | `qwen2.5:14b` | 32,768 | 5 (Large/Strongest) |

---

## 4. Mathematical Formulations

### Multidimensional Task Load & Residual
A task demand is represented as $\mathbf{D} \in \mathbb{R}^k$:

$$\mathbf{D} = \begin{bmatrix} d_{\text{math}} \\ d_{\text{code}} \\ d_{\text{mechanics}} \\ d_{\text{structured}} \\ d_{\text{trap}} \end{bmatrix}, \quad \mathbf{C}_{\text{total}} = \sum_{i=1}^N a_i \mathbf{C}_i, \quad \mathbf{R} = \mathbf{D} - \sum_{i=1}^N a_i \mathbf{C}_i$$

Routing terminates when $\|\mathbf{R}\| \le \epsilon$ according to external verification.

### Normalized Inference Cost Proxy
For local inference, cost is measured using exact Ollama evaluation metadata:

$$K_i = \left( \alpha T_i^{\text{in}} + \beta T_i^{\text{out}} + \gamma t_i + \delta E_i \right) \cdot \mu_i$$

where $T_i^{\text{in}}$ is prompt tokens, $T_i^{\text{out}}$ is completion tokens, $t_i$ is wall-clock latency in seconds, and $\mu_i$ is the tier multiplier.

### Capability-Cost Efficiency & Marginal Escalation
1. **Initial Efficiency**:
   $$\eta_i = \frac{P(\text{verified success} \mid M_i, \text{domain})}{\mathbb{E}[K_i]}$$

2. **Marginal Escalation Efficiency**:
   $$E_{ij} = \frac{\Delta Q_{ij}}{\Delta K_{ij}} = \frac{Q_j - Q_i}{K_j - K_i}$$
   When $E_{ij}$ indicates intermediate models cannot economically resolve the observed residual failure, BWR skips directly (e.g. $M_1 \to M_4$).

3. **Context Token Reduction**:
   $$S_{\text{context}} = 1 - \frac{T_{\text{residual-context}}}{T_{\text{full-context}}}$$

---

## 5. Domain Verifiers

* **Code (`verifiers/code_verifier.py`)**: Sandboxed execution with timeout, AST syntax validation, unit test assertion runner, stdout verification, and structured exception isolation (`syntax_error`, `runtime_error`, `unit_test_failure`, `timeout_error`).
* **Mathematics (`verifiers/math_verifier.py`)**: Exact SymPy symbolic simplification ($\text{simplify}(y_{\text{pred}} - y_{\text{true}}) = 0$), numerical tolerance verification ($|y_{\text{pred}} - y_{\text{true}}| \le 10^{-5}$), and algebraic system consistency.
* **Mechanics (`verifiers/mechanics_verifier.py`)**:
  * Rotating mass single/multi-plane dynamic balancing ($\sum m r \cos\theta = 0, \sum m r \sin\theta = 0, \sum m r l \cos\theta = 0, \sum m r l \sin\theta = 0$).
  * Static equilibrium ($\sum F_x = 0, \sum F_y = 0, \sum M = 0$).
  * Natural vibrations ($\omega_n = \sqrt{k/m}$).
  * Statically determinate beam reactions and First Law thermodynamic conservation ($Q - W = \Delta U$).
* **Structured Data & Hallucination Traps (`verifiers/structured_verifier.py`)**: Schema compliance and adversarial trap detection (buoyancy omission, extraneous radical roots, velocity/pressure sign inversion, Python mutable default argument traps).

---

## 6. Experimental Baselines & Ablations

| Experiment | Method | Escalation Strategy | Context Transmitted |
| :--- | :--- | :--- | :--- |
| **`exp00`** | Individual Models | Standalone benchmark per tier | Full Task |
| **`exp01`** | Strongest-Only | $D \to M_5$ | Full Task |
| **`exp02`** | Fixed Cascade | $M_1 \to M_2 \to M_3 \to M_4 \to M_5$ | Full Task |
| **`exp03`** | Confidence Router | Terminate if self-reported confidence $\ge \tau_{\text{conf}}$ | Full Task |
| **`exp04`** | Verified Full Escalation | Escalate on verifier failure | Full Task |
| **`exp05`** | Verified Residual Routing (VRR) | Escalate on verifier failure | Minimal Residual Context |
| **`exp06`** | Balanced Weight Routing (BWR) | Empirical matrix $\mathbf{w}_i$ + marginal $E_{ij}$ + skip logic | Minimal Residual Context |

---

## 7. Installation & Quickstart

### Prerequisites
* Python 3.10+
* (Optional) Ollama running locally for live local LLM benchmarking.

### Installation
```bash
# Clone the repository
git clone https://github.com/your-org/balanced-weight-router.git
cd balanced-weight-router

# Install dependencies and editable package
pip install -e .
```

### Running Tests
```bash
pytest tests/ -v
```

### Running the Full Benchmark Suite (Deterministic Mock Engine)
```bash
# Executes all 6 baseline experiments, ablations, and generates summary reports
python experiments/run_all_experiments.py --mock
```

### Running with Live Local Ollama Models
```bash
# Ensure Ollama is running and models are pulled (e.g. ollama pull qwen2.5:0.5b)
python experiments/run_all_experiments.py --live
```

---

## 8. Experimental Results & Telemetry

Benchmark runs write immutable, append-only JSONL event streams to `results/logs/<run_id>.jsonl` tracking:
* Exact prompt/completion tokens
* Wall-clock latency
* Verification pass/fail scores and failure categories
* Context reduction percentages
* Accumulated inference cost

### Example Summary Output

```
============================== FINAL SCIENTIFIC COMPARISON ==============================
| Routing Policy                     | Success (Q)   |   Avg Tokens |   Avg Cost (K) | Cost Saving (S_C)   | Token Saving (S_T)   | Context Red.   | Escalation %   | M5 Util %   |
|------------------------------------|---------------|--------------|----------------|---------------------|----------------------|----------------|----------------|-------------|
| Strongest Only (M5)                | 82.6%         |        110.7 |        0.02055 | 0.0% (Base)         | 0.0% (Base)          | 0.0%           | 0.0%           | 100.0%      |
| Fixed Cascade                      | 87.0%         |        232.6 |        0.01054 | +48.7%              | -110.1%              | 0.0%           | 69.6%          | 0.0%        |
| Confidence Router (Control)        | 0.0%          |        307.7 |        0.01852 | +9.9%               | -177.9%              | 0.0%           | 78.3%          | 0.0%        |
| Verified Escalation (Full Context) | 87.0%         |        231.5 |        0.01031 | +49.9%              | -109.1%              | 0.0%           | 82.6%          | 0.0%        |
| Verified Residual Routing (VRR)    | 91.3%         |        280.5 |        0.01218 | +40.7%              | -153.3%              | 12.6%          | 69.6%          | 0.0%        |
| Balanced Weight Router (BWR)       | 91.3%         |        213.3 |        0.01665 | +19.0%              | -92.7%               | 13.7%          | 60.9%          | 39.1%       |
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
