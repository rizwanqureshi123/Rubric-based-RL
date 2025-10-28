rubric-eval-task
================

Rubric-driven reinforcement learning task where the model must grade a candidate postmortem against a set of strict criteria. The harness exposes the question, candidate response, evidence log, and golden analysis through tools. The grader enforces that the final verdict matches the rubric—down to specific evidence cited in the justifications—so the assistant must reason carefully rather than guess.

## What the task teaches

- Converting rubric text into a structured evaluation of another model’s answer.
- Referencing multiple sources (incident log + golden analysis) before issuing a verdict.
- Producing machine-checkable JSON that cites concrete evidence and proposes targeted remediation steps.

The dataset lives in `data/scenario.json` and contains the incident log, golden analysis, rubric definition, and grading expectations.

## Running the task

1. Install dependencies (requires [uv](https://github.com/astral-sh/uv) or any PEP 517 runner):
   ```bash
   uv sync
   ```
2. Supply an Anthropic API key via `ANTHROPIC_API_KEY` or `anthropic_key.txt`.
3. Execute the evaluation:
   ```bash
   uv run main.py --episodes 10 --model claude-3-5-haiku-20241022
   ```
4. Add `--concurrent` to parallelise multiple episodes when calibrating pass rates.

Each run prints pass/fail feedback per episode and a final pass-rate estimate so you can keep the task inside the 10–40 % window.

## Grading contract

- The final response must be a JSON object with `overall_decision`, `rubric_scores`, `summary`, and `improvement_plan` (`reject` or `fail` are both accepted for `overall_decision`).
- Every rubric entry is expected exactly once with the correct pass/fail label and a justification that mentions the specified evidence keywords and flags the deficiency.
- The summary must reference data leakage, class imbalance, recall monitoring, the 3.8% positive rate, and the 18% recall drop.
- The improvement plan must contain actionable steps that cover recreating the split, handling class imbalance, and adding recall monitoring.

Because every requirement is machine-checked, surface-level answers without evidence fail automatically; models must reason over the provided artifacts before responding.
