---
name: skill-distill
description: Distill a successful coding-agent session into a reproducible, tested Hanzo skill for a chosen model and agent harness. Use when asked to distill a session, teach a smaller model a workflow, or reproduce a successful agent run.
---

# Skill Distill for Hanzo

This Hanzo-native entry is adapted conceptually from [sybil-solutions/skill-distill](https://github.com/sybil-solutions/skill-distill) (MIT license). It is **not** a vendored copy of the upstream scripts or a claim of tested equivalence. Review upstream for its fuller trial bookkeeping, session extractors and harness adapters.

## Inputs

Require a reference session or trace, original user prompt, baseline workspace commit, target student harness and model, and acceptance checks. Default to Hanzo's agent runtime for the student; use Enso to resolve the configured model. Zen may generate text; Kai provides typed decisions, not text generation. Never silently substitute another model for the requested student.

## Workflow

1. **Consent and redact.** Before reading or exporting session transcripts, get permission to use the trace. Remove secrets, credentials, tokens, personal information, proprietary code and irrelevant conversation context. Store raw traces in private, ignored, tenant-scoped storage; never commit them.
2. **Pin a clean baseline.** Record the starting Git commit, dependency versions, tool scopes and environment configuration. Do not use the finished reference result as a student input.
3. **Define observable acceptance.** Identify 4–10 assertions for the desired outcome, including tests and behavioral checks. Verify those checks against the reference before using them to judge the student.
4. **Write the skill.** Create a concise `SKILL.md` explaining the decisions, workflow, pitfalls, validation and recovery steps. Do not include the final answer or a script that simply reconstructs the reference output.
5. **Run the student in isolation.** Start each trial from the same clean baseline in a Hanzo sandbox with explicitly approved tools and resources. Use the original prompt plus the candidate skill, with unrelated context excluded. Record model, harness, skill revision, tool calls, costs and results.
6. **Compare and revise.** Find the earliest meaningful divergence between the reference and the student. Improve the instructions at that point; rerun from a fresh baseline. Use at least three independent trials per candidate version, not a single lucky run.
7. **Review and publish.** Present side-by-side results, failed assertions, trial variance and limitations. Require explicit user acceptance before publishing the generated skill into a shared catalog.

## Output

- `SKILL.md` with metadata, instructions and examples
- Reproducibility manifest (baseline SHA, model, harness, allowed tools, tests)
- Sanitized evaluation ledger and acceptance results
- Explicit reviewer acceptance or rejection

## Safety and quality gates

- No hidden network calls, unapproved tool scopes, copied credentials, or untrusted install hooks.
- No leakage of reference outputs, private traces, or evaluator answers into student instructions.
- Fail closed if the baseline cannot be reconstructed or the requested student cannot be run.
- For scheduled distillation, enforce per-tenant isolation, resource limits and cancellation.
- Keep the generated skill separate from model weights: skill distillation is **not** fine-tuning.
- Do not claim reproducibility without actual trial evidence.

## Upstream attribution

Inspired by `sybil-solutions/skill-distill`, © its contributors, distributed under MIT. Source: https://github.com/sybil-solutions/skill-distill . This Hanzo skill is an original integration guide and does not bundle upstream Python scripts. If those scripts are vendored later, preserve their license and copyright notices.
