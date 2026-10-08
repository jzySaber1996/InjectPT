# Skill Graph REINFORCE Training Flow

## Flowchart

```mermaid
flowchart TD
    A[Load normalized PPO prompt train/val data] --> B[Load base model and SFT adapter]
    B --> C[Sample prompt batch]
    C --> D[Generate candidate Mermaid graphs]
    D --> E[Normalize Mermaid text]
    E --> F[Run structural validator and Z3 witness search]
    F --> G[Compute base reward from reward_config]
    F --> H[Compute z3_reachability_score]
    F --> I[Compute mermaid_completeness_score]
    G --> J[Build per-trajectory optimization signals]
    H --> J
    I --> J
    J --> K[Group baseline and policy-gradient advantage]
    K --> L[Assemble total loss]
    L --> M[Backward and optimizer step]
    M --> N[Validation / checkpoint / plots]
    N --> C
```

## Loss Design

The updated training loss keeps the original REINFORCE skeleton, but adds two explicit structural auxiliary terms:

- `L_pg = -A_hat * mean_logprob`
- `L_z3 = -lambda_z3 * s_z3 * mean_logprob`
- `L_mermaid = -lambda_mermaid * s_mermaid * mean_logprob`
- `L_anchor = lambda_sft * CE(reference_graph)`
- `L_buffer = lambda_buffer * CE(success_buffer_graph)`
- `L_total = L_pg + L_z3 + L_mermaid + L_anchor + L_buffer - lambda_entropy * H`

Where:

- `s_z3` is a normalized reachability score in `[0, 1]`, composed from `path_exists`, `witness_path`, `all_nodes_reachable_from_start`, `all_nodes_can_reach_finish`, and witness coverage.
- `s_mermaid` is a normalized completeness score in `[0, 1]`, composed from `graph TD` header, edge markers, start/finish markers, resolved start/finish nodes, and basic node/edge density.

## Tuning Guidance

- Start with `lambda_z3 > lambda_mermaid`, because path reachability is the harder structural constraint.
- The local config now uses:
  - `z3_reachability_loss_coef: 0.30`
  - `mermaid_completeness_loss_coef: 0.15`
- If the model emits valid Mermaid syntax but still fails Z3, raise `z3_reachability_loss_coef` first.
- If the model often drops `graph TD`, start/finish nodes, or edge structure, raise `mermaid_completeness_loss_coef` first.
