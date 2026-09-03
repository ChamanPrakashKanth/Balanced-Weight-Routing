# Feature-Vector Balanced Weight Routing (FV-BWR) Generalization Report

**Dataset**: 24 tasks total | 14 Train | 10 Unseen Test

### Generalization Comparison on Unseen Tasks

| Routing Policy                                     | Test Success (Q)   |   Avg Tokens |   Avg Cost (K) | Cost Saving (S_C)   | Token Saving (S_T)   | M5 Util %   | Latency   |
|----------------------------------------------------|--------------------|--------------|----------------|---------------------|----------------------|-------------|-----------|
| M5 Only (Baseline)                                 | 80.0%              |        152.7 |        0.02715 | 0.0% (Base)         | 0.0% (Base)          | 100.0%      | 1.15s     |
| Policy A: Fixed Cascade (Full Context)             | 80.0%              |        239.8 |        0.01139 | +58.0%              | -57.0%               | 0.0%        | 0.73s     |
| Policy B: VRR (Fixed + Residual)                   | 80.0%              |        303.3 |        0.01333 | +50.9%              | -98.6%               | 0.0%        | 0.77s     |
| Policy C: Coarse BWR (Domain-only)                 | 90.0%              |        212   |        0.02232 | +17.8%              | -38.8%               | 20.0%       | 0.99s     |
| Policy E: FV-BWR (Full Context)                    | 90.0%              |        202   |        0.01488 | +45.2%              | -32.3%               | 20.0%       | 0.85s     |
| Policy F: FV-BWR + VRR (Feature Vector + Residual) | 90.0%              |        269.2 |        0.01754 | +35.4%              | -76.3%               | 10.0%       | 0.88s     |

### Scientific Conclusions
- **M5 Only Success**: 80.0%
- **FV-BWR Test Success**: 90.0%
- **FV-BWR Cost Savings**: +35.4%
