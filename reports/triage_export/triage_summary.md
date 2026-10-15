| Security Posture Metric | Value | Context |
| :--- | :---: | :--- |
| **Total Scanned Identities** | `86` | 51 Users, 35 Roles |
| **Evaluated Policy Documents** | `31` | 26 Monitored Cloud Resources |
| **Flagged Latent Escalation Risks** | `1` | 1 Critical, 0 High |
| **Misleading Safe (Static Analysis Blind)** | `1` | Level 1: 0.0, Level 2: >0.70 |
| **Average Latent Blast Radius** | `1.00` | Crown Jewel Assets per Flagged Principal |
| **Pareto Dominance (RQ2 Tradeoff)** | `CERTIFIED` | Uncertainty-Aware Repair strictly optimal |

### Top High-Risk Identities (Calibrated Latent Privilege Escalation)

| Principal | Type | Dept | Observed (Go) | Latent P̂ | Severity | Blind Spot? | Target Asset |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `user:secops-standard-0` | User | SecOps | 0.00 | 0.88 | **CRITICAL** | **YES (⚠️ Blind)** | `s3:::enterprise-data-lake-prod` |
| `user:secops-standard-1` | User | SecOps | 0.00 | 0.00 | **LOW** | No | `None` |
| `user:secops-restricted-2` | User | SecOps | 0.00 | 0.00 | **LOW** | No | `None` |
| `user:secops-standard-3` | User | SecOps | 0.00 | 0.00 | **LOW** | No | `None` |
| `user:secops-restricted-4` | User | SecOps | 0.00 | 0.00 | **LOW** | No | `None` |
