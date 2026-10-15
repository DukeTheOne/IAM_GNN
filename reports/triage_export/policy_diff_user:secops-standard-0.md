# Remediation Diff: `user:secops-standard-0`

- **Severity Tier:** `CRITICAL` (Latent Attack P̂ = `0.88`)
- **Capability Status:** `VALID (Admissible under C(a,r,c))`
- **Blast Radius Reduction:** `100.0%`
- **Historical T_1 Conformance:** `100.0% (Zero regressions)`
- **Holdout T_2 Operational Breakage:** `1.33%`

### Unified Policy AST Diff

```diff
--- before/Policy_DeveloperExecution.json
+++ after/Policy_DeveloperExecution.json
@@ -10,7 +10,7 @@
       ],
       "NotAction": [],
       "Resource": [
-        "*"
+        "arn:aws:iam::123456789012:role/ScopedBuildRole"
       ],
       "NotResource": [],
       "Condition": {}
```

### Standalone Graph Canvas

Open [`ego_graph_user:secops-standard-0.html`](./ego_graph_user:secops-standard-0.html) in any browser.
