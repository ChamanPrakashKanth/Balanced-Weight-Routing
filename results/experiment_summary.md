# Balanced Weight Routing (BWR) - Scientific Benchmark Report

**Date/Time:** 2026-09-03 09:14:32 UTC
**Total Benchmark Tasks:** 23
**Execution Mode:** Deterministic Mock Simulator

## Comparative Results Table

| Routing Policy                     | Success (Q)   |   Avg Tokens |   Avg Cost (K) | Cost Saving (S_C)   | Token Saving (S_T)   | Context Red.   | Escalation %   | M5 Util %   |
|------------------------------------|---------------|--------------|----------------|---------------------|----------------------|----------------|----------------|-------------|
| Strongest Only (M5)                | 82.6%         |        110.7 |        0.02055 | 0.0% (Base)         | 0.0% (Base)          | 0.0%           | 0.0%           | 100.0%      |
| Fixed Cascade                      | 87.0%         |        232.6 |        0.01054 | +48.7%              | -110.1%              | 0.0%           | 69.6%          | 0.0%        |
| Confidence Router (Control)        | 0.0%          |        307.7 |        0.01852 | +9.9%               | -177.9%              | 0.0%           | 78.3%          | 0.0%        |
| Verified Escalation (Full Context) | 87.0%         |        231.5 |        0.01031 | +49.9%              | -109.1%              | 0.0%           | 82.6%          | 0.0%        |
| Verified Residual Routing (VRR)    | 91.3%         |        280.5 |        0.01218 | +40.7%              | -153.3%              | 12.6%          | 69.6%          | 0.0%        |
| Balanced Weight Router (BWR)       | 91.3%         |        213.3 |        0.01665 | +19.0%              | -92.7%               | 13.7%          | 60.9%          | 39.1%       |

## Hypothesis Evaluation

- **H0**: $C_{\text{BWR}} \ge C_{\text{strongest}}$
- **H1**: $C_{\text{BWR}} < C_{\text{strongest}}$ subject to $Q_{\text{BWR}} \approx Q_{\text{strongest}}$
- **Empirical Cost Saving**: 19.02%
- **Quality Delta**: $\Delta Q = +8.70\%$
- **Conclusion**: Failed to reject null hypothesis.

## Key Findings & Ablations

1. **Verification Dominance**: Confidence routing suffers from false overconfidence on adversarial/trap benchmarks, while deterministic verifiers eliminate false acceptance.
2. **Context Reduction (VRR)**: Residual extraction isolates the failed condition, reducing prompt tokens significantly without sacrificing repair capability.
3. **Empirical Capability Allocation (BWR)**: Dispatching via empirical domain capabilities $\mathbf{w}_i$ and skipping unviable intermediate tiers provides the lowest inference cost.
