# Agent readiness

## Current stance

This project is not building agents in the MVP.

The current goal is to build a deterministic, testable pipeline with clean
interfaces so that an agentic copilot, web UI, or workflow orchestrator can be
attached later without rewriting the processing code.

## Future agent role

A future agent should act as an external copilot or controller, not as an
untracked decision-maker inside numerical processing. It may eventually:

- assemble or edit pipeline configuration;
- choose a backend implementation for a stage;
- launch deterministic pipeline stages;
- inspect logs, artifacts, registry state, and QC summaries;
- suggest parameter forks after failures;
- compare candidate runs against benchmark expectations;
- generate small one-off analysis scripts around the pipeline.

The pipeline stages themselves should remain explicit, reproducible functions or
commands with typed inputs, typed outputs, and recorded parameters.

## Design requirements today

To keep future agent integration cheap, every stage should expose:

- structured inputs: manifests, project config, artifact registry, backend
  context;
- structured outputs: artifacts, logs, QC records, and updated registry state;
- deterministic behavior for the same inputs and parameters;
- explicit backend selection rather than hidden package-specific behavior;
- machine-readable failure modes where possible;
- enough lineage to rerun, compare, or delete recomputable artifacts.

## What not to do now

- Do not add autonomous retries before stage-level success criteria exist.
- Do not let an agent modify data or parameters without recording the change.
- Do not couple pipeline logic to a chat interface.
- Do not require a web UI to run or test core processing.
- Do not make a backend choice irreversible in project state.

## Practical implication

The near-term pipeline should first become a reliable API that both humans and
future agents can drive:

```text
manifest + registry + backend config
  -> deterministic stage command
  -> artifacts + logs + QC + updated registry
```

Only after these contracts are stable should agentic troubleshooting or
parameter forking be layered on top.
