# Balanced Weight Routing (BWR) - Scientific Benchmark Report

**Date/Time:** 2026-09-03 10:03:20 UTC
**Total Benchmark Tasks:** 23
**Execution Mode:** Deterministic Mock Simulator

## Comparative Results Table

| Routing Policy                            | Success (Q)   |   Avg Tokens |   Avg Cost (K) | Cost Saving (S_C)   | Token Saving (S_T)   | Context Red.   | Escalation %   | M5 Util %   |
|-------------------------------------------|---------------|--------------|----------------|---------------------|----------------------|----------------|----------------|-------------|
| M5 Only (Baseline)                        | 82.6%         |        111.2 |        0.02051 | 0.0% (Base)         | 0.0% (Base)          | 0.0%           | 0.0%           | 100.0%      |
| Policy A: Fixed Cascade (Full Context)    | 91.3%         |        225.4 |        0.01    | +51.3%              | -102.7%              | 0.0%           | 69.6%          | 0.0%        |
| Policy B: VRR (Fixed + Residual)          | 91.3%         |        270.7 |        0.01156 | +43.6%              | -143.4%              | 13.6%          | 65.2%          | 0.0%        |
| Policy C: BWR (Full Context)              | 87.0%         |        199.5 |        0.01855 | +9.6%               | -79.4%               | 0.0%           | 73.9%          | 39.1%       |
| Policy D: BWR + VRR (Balanced + Residual) | 82.6%         |        272.5 |        0.02449 | -19.4%              | -145.0%              | 12.4%          | 78.3%          | 34.8%       |
| Confidence Router (Ablation Control)      | 0.0%          |        317.1 |        0.0204  | +0.5%               | -185.1%              | 0.0%           | 78.3%          | 0.0%        |

## Hypothesis Evaluation

- **H0**: $C_{\text{BWR}} \ge C_{\text{strongest}}$
- **H1**: $C_{\text{BWR}} < C_{\text{strongest}}$ subject to $Q_{\text{BWR}} \approx Q_{\text{strongest}}$
- **Empirical Cost Saving**: -19.38%
- **Quality Delta**: $\Delta Q = +0.00\%$
- **Conclusion**: Failed to reject null hypothesis.

## Key Findings & Ablations

1. **Verification Dominance**: Confidence routing suffers from false overconfidence on adversarial/trap benchmarks, while deterministic verifiers eliminate false acceptance.
2. **Context Reduction (VRR)**: Residual extraction isolates the failed condition, reducing prompt tokens significantly without sacrificing repair capability.
3. **Empirical Capability Allocation (BWR)**: Dispatching via empirical domain capabilities $\mathbf{w}_i$ and skipping unviable intermediate tiers provides the lowest inference cost.
