import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from anthropic import AsyncAnthropic


DATA_PATH = Path(__file__).parent / "data" / "scenario.json"


def _load_scenario() -> Dict[str, Any]:
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


SCENARIO = _load_scenario()
RUBRIC_INDEX: Dict[str, Dict[str, Any]] = {
    item["criterion_id"]: item for item in SCENARIO["rubric"]
}
ALLOWED_DECISIONS = {
    SCENARIO["expected_overall_decision"].strip().lower()
}
for decision in SCENARIO.get("acceptable_overall_decisions", []):
    ALLOWED_DECISIONS.add(decision.strip().lower())


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _contains_all(text: str, keywords: Sequence[Any]) -> bool:
    lowered = text.lower()
    for keyword in keywords:
        if isinstance(keyword, (list, tuple)):
            if not any(option.lower() in lowered for option in keyword):
                return False
        else:
            if keyword.lower() not in lowered:
                return False
    return True


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "show_question",
        "description": (
            "Return the grading question along with the candidate response that needs to be scored."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "show_incident_log",
        "description": (
            "Return bullet-point evidence extracted from the postmortem timeline. "
            "Use this to ground each rubric judgement."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "show_golden_analysis",
        "description": (
            "Return the golden incident analysis summarising the correct findings and recommended fixes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
]


PROMPT = """You are an ML code reviewer. A candidate wrote an incident analysis for the toxicity classifier outage.
Grade that candidate response strictly against the rubric:
1. Call both show_question and show_incident_log before drafting any verdict.
2. Optionally consult show_golden_analysis to confirm the expected solution.
3. When you are certain about every criterion, reply with a JSON object:
{"overall_decision": "...", "rubric_scores": [{"criterion_id": "...", "score": "...", "justification": "..."}], "summary": "...", "improvement_plan": ["...", "...", "..."]}

Rules:
- Only output the JSON object (no Markdown).
- The summary must mention data leakage, class imbalance, and recall monitoring.
- Every rubric justification must quote concrete evidence from the tools.
- The improvement_plan must contain concrete next steps aligned with the rubric gaps.
- Scores are case-insensitive strings; use pass/fail exactly.
"""


def _dispatch_tool(name: str, arguments: Dict[str, Any]) -> str:
    if name == "show_question":
        payload = {
            "scenario_id": SCENARIO["scenario_id"],
            "question": SCENARIO["question"],
            "candidate_response": SCENARIO["candidate_response"],
        }
        return json.dumps(payload)

    if name == "show_incident_log":
        payload = {
            "incident_log": SCENARIO["incident_log"],
        }
        return json.dumps(payload)

    if name == "show_golden_analysis":
        payload = {
            "golden_response": SCENARIO["golden_response"],
            "rubric": SCENARIO["rubric"],
        }
        return json.dumps(payload)

    raise ValueError(f"Unhandled tool name: {name}")


def _ensure_improvement_coverage(plan: Sequence[str]) -> Tuple[bool, str]:
    if len(plan) < len(SCENARIO["improvement_keyword_sets"]):
        return False, "Improvement plan must contain at least three actionable steps."
    plan_lower = [item.lower() for item in plan]
    for keyword_group in SCENARIO["improvement_keyword_sets"]:
        if not any(
            any(keyword.lower() in step for keyword in keyword_group)
            for step in plan_lower
        ):
            return (
                False,
                f"Improvement plan missing required theme covering keywords: {keyword_group}",
            )
    return True, ""


def grade_response(response_text: str) -> Tuple[bool, str]:
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as exc:
        return False, f"Response was not valid JSON: {exc}"

    required_fields = {"overall_decision", "rubric_scores", "summary", "improvement_plan"}
    missing = required_fields.difference(parsed.keys())
    if missing:
        return False, f"Missing required field(s): {', '.join(sorted(missing))}"

    overall = parsed["overall_decision"]
    if not isinstance(overall, str):
        return False, "overall_decision must be a string."
    if overall.strip().lower() not in ALLOWED_DECISIONS:
        return False, "overall_decision does not match the rubric expectation."

    rubric_scores = parsed["rubric_scores"]
    if not isinstance(rubric_scores, list) or not rubric_scores:
        return False, "rubric_scores must be a non-empty list."

    if len(rubric_scores) != len(SCENARIO["rubric"]):
        return False, "rubric_scores length does not match the rubric definition."

    seen_criteria: Dict[str, bool] = {}
    for entry in rubric_scores:
        if not isinstance(entry, dict):
            return False, "Each rubric score must be an object."
        for key in ("criterion_id", "score", "justification"):
            if key not in entry:
                return False, f"Rubric entry missing field '{key}'."
        criterion_id = entry["criterion_id"]
        if criterion_id not in RUBRIC_INDEX:
            return False, f"Unknown rubric criterion_id '{criterion_id}'."
        if criterion_id in seen_criteria:
            return False, f"Duplicate rubric criterion_id '{criterion_id}'."
        seen_criteria[criterion_id] = True

        expected = RUBRIC_INDEX[criterion_id]
        score = entry["score"]
        if not isinstance(score, str):
            return False, f"Score for criterion '{criterion_id}' must be a string."
        if score.strip().lower() != expected["expected_score"]:
            return False, f"Score mismatch for criterion '{criterion_id}'."

        justification = entry["justification"]
        if not isinstance(justification, str) or not justification.strip():
            return False, f"Justification for '{criterion_id}' must be a non-empty string."

        if not _contains_all(justification, expected["must_include"]):
            return (
                False,
                f"Justification for '{criterion_id}' missing required evidence keywords.",
            )

        if not _contains_any(justification, expected["failure_markers"]):
            return (
                False,
                f"Justification for '{criterion_id}' must explicitly flag the deficiency.",
            )

    summary = parsed["summary"]
    if not isinstance(summary, str) or not summary.strip():
        return False, "Summary must be a non-empty string."
    for phrase in SCENARIO["summary_requirements"]:
        if phrase.lower() not in summary.lower():
            return False, f"Summary must mention '{phrase}'."

    improvement_plan = parsed["improvement_plan"]
    if not isinstance(improvement_plan, list) or not improvement_plan:
        return False, "improvement_plan must be a non-empty list."
    if not all(isinstance(item, str) and item.strip() for item in improvement_plan):
        return False, "Each improvement plan item must be a non-empty string."
    ok, message = _ensure_improvement_coverage(improvement_plan)
    if not ok:
        return False, message

    return True, "Rubric evaluation accepted."


@dataclass
class EpisodeResult:
    episode: int
    success: bool
    duration_seconds: float
    final_message: str
    feedback: str


def _serialize_blocks(blocks: Sequence[Any]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for block in blocks:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            serialized.append({"type": "text", "text": block.text})
        elif block_type == "tool_use":
            serialized.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
        else:
            raise ValueError(f"Unsupported content block type: {block_type}")
    return serialized


async def run_episode(
    client: AsyncAnthropic,
    episode_index: int,
    model: str,
    max_turns: int = 6,
    max_tokens: int = 600,
    patience_tool_only: int = 2,
) -> EpisodeResult:
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": PROMPT}]}
    ]
    start = time.perf_counter()
    final_message = ""
    last_attempt = ""
    feedback = "Assistant did not produce a final answer."
    latest_feedback = feedback
    success = False
    tool_only_rounds = 0

    for turn_index in range(max_turns):
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )
        assistant_content = _serialize_blocks(response.content)
        messages.append({"role": "assistant", "content": assistant_content})

        tool_calls = [
            block for block in response.content if getattr(block, "type", None) == "tool_use"
        ]
        if tool_calls:
            text_blocks = [
                block.text
                for block in response.content
                if getattr(block, "type", None) == "text"
            ]
            for call in tool_calls:
                tool_output = _dispatch_tool(call.name, call.input)
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": call.id,
                                "content": [{"type": "text", "text": tool_output}],
                            }
                        ],
                    }
                )
            if text_blocks:
                tool_only_rounds = 0
            else:
                tool_only_rounds += 1
                should_remind = (
                    tool_only_rounds >= patience_tool_only
                    and turn_index + 1 < max_turns
                )
                if should_remind:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "You have the necessary evidence. Respond with the JSON verdict now, "
                                        "ensuring each rubric justification cites the required metrics."
                                    ),
                                }
                            ],
                        }
                    )
                    tool_only_rounds = 0
            continue

        text_blocks = [
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ]
        final_message = "".join(text_blocks).strip()
        if final_message:
            last_attempt = final_message
        success, feedback = grade_response(final_message)
        if success:
            latest_feedback = feedback
            break

        latest_feedback = feedback
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{feedback} Please correct the JSON response and resend it. "
                            "Use direct quotes or precise paraphrases from the tools."
                        ),
                    }
                ],
            }
        )
        final_message = ""
        feedback = "Assistant incorporated feedback and corrected the response."
        continue

    duration = time.perf_counter() - start
    return EpisodeResult(
        episode=episode_index,
        success=success,
        duration_seconds=duration,
        final_message=final_message or last_attempt,
        feedback=latest_feedback,
    )


def load_api_key() -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        return api_key

    key_file = Path("anthropic_key.txt")
    if key_file.exists():
        candidate = key_file.read_text(encoding="utf-8").strip()
        if candidate:
            return candidate

    raise RuntimeError(
        "Anthropic API key not found. Set ANTHROPIC_API_KEY or place the key in anthropic_key.txt."
    )


async def main(concurrent: bool, episodes: int, model: str) -> None:
    api_key = load_api_key()
    client = AsyncAnthropic(api_key=api_key)

    runners = [
        run_episode(client, idx + 1, model=model)
        for idx in range(episodes)
    ]
    if concurrent:
        results = await asyncio.gather(*runners)
    else:
        results = []
        for coroutine in runners:
            results.append(await coroutine)

    total = len(results)
    successes = sum(1 for result in results if result.success)
    pass_rate = successes / total if total else 0.0

    for result in results:
        status = "PASS" if result.success else "FAIL"
        print(
            f"[episode {result.episode:02d}] {status} "
            f"({result.duration_seconds:.1f}s) – {result.feedback}"
        )
        if not result.success:
            print(f"  Final message: {result.final_message}")

    print(
        f"\nCompleted {total} episode(s). Successes: {successes}. "
        f"Estimated pass rate: {pass_rate * 100:.1f}%."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the rubric-based evaluation task for the toxicity classifier postmortem."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate (default: 10).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-haiku-20241022",
        help="Anthropic model name to query.",
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Run episodes concurrently instead of sequentially.",
    )
    args = parser.parse_args()

    asyncio.run(main(concurrent=args.concurrent, episodes=args.episodes, model=args.model))
