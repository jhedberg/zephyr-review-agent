# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""The review loop against a scripted OpenRouter and a scripted GitHub, no network."""

from __future__ import annotations

import json

import httpx
import pytest

from zephyr_reviewer.intel import SourceIntel
from zephyr_reviewer.mcp_server import build_server
from zephyr_reviewer.review import publish as pub
from zephyr_reviewer.review.agent import run_review
from zephyr_reviewer.review.config import ReviewConfig
from zephyr_reviewer.review.github_api import GitHub, PullRequest
from zephyr_reviewer.review.mcp_client import ToolBridge
from zephyr_reviewer.review.openrouter import OpenRouter
from zephyr_reviewer.review.prompts import load_profile, opening_message, system_prompt
from zephyr_reviewer.review.render import inline_comments, render_comment
from zephyr_reviewer.review.schema import parse_review

GOOD_REVIEW = {
    "summary": "Adds bus recovery to the Foo I2C driver and a Kconfig symbol for it.",
    "verdict": "request_changes",
    "confidence": "high",
    "findings": [
        {
            "file": "drivers/i2c/i2c_foo.c",
            "line": 17,
            "end_line": 19,
            "severity": "high",
            "category": "concurrency",
            "title": "Early return leaks the bus lock",
            "problem": "foo_transfer returns on error before k_sem_give, so the semaphore is never released.",
            "trigger": "Any failed transfer.",
            "impact": "Every later transfer blocks forever.",
            "recommendation": "Release the lock before returning.",
            "suggested_code": "\tif (ret < 0) {\n\t\tk_sem_give(&data->lock);\n\t\treturn ret;\n\t}",
            "confidence": "high",
            "evidence": "read_file drivers/i2c/i2c_foo.c:9-21",
        },
        {
            "file": "drivers/i2c/Kconfig",
            "line": 20,
            "severity": "medium",
            "category": "build_config",
            "title": "select of I2C_BUS_RECOVERY without a definition",
            "problem": "I2C_BUS_RECOVERY is not defined anywhere in the tree.",
            "trigger": "CONFIG_I2C_FOO_RECOVERY=y",
            "impact": "Kconfig warning; recovery never enabled.",
            "recommendation": "Define the symbol or drop the select.",
            "confidence": "medium",
        },
        {
            "file": "kernel/sem.c",
            "line": 1,
            "severity": "suggestion",
            "category": "maintainability",
            "title": "Unrelated file cited",
            "problem": "Should be demoted to summary-only.",
            "trigger": "",
            "impact": "",
            "recommendation": "",
            "confidence": "low",
        },
        {
            "file": "drivers/i2c/i2c_foo.c",
            "line": 1,
            "severity": "low",
            "category": "documentation",
            "title": "Context line cited",
            "problem": "Line 1 is not part of the diff.",
            "trigger": "",
            "impact": "",
            "recommendation": "",
            "confidence": "low",
        },
    ],
    "test_assessment": "The recovery scenario enables the option but does not exercise foo_recover().",
    "commit_message_assessment": "",
}


def _tool_call(call_id: str, name: str, args: dict) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _completion(message: dict, prompt_tokens: int = 1000) -> dict:
    return {
        "id": "gen-1",
        "model": "anthropic/claude-sonnet-4.5",
        "choices": [{"message": {"role": "assistant", **message}, "finish_reason": "tool_calls"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 50,
            "total_tokens": prompt_tokens + 50,
            "cost": 0.0123,
        },
    }


class ScriptedOpenRouter:
    """Replies in order; records every request payload."""

    def __init__(self, replies: list[dict]):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        self.requests.append(payload)
        if not self.replies:
            return httpx.Response(500, json={"error": {"message": "script exhausted", "code": 500}})
        return httpx.Response(200, json=self.replies.pop(0))


def _llm(script: ScriptedOpenRouter, models=("anthropic/claude-sonnet-4.5", "openai/gpt-4.1")) -> OpenRouter:
    return OpenRouter(
        "key", list(models), client=httpx.Client(transport=httpx.MockTransport(script.handler)), max_attempts=2
    )


async def _run(mini_repo, script: ScriptedOpenRouter, **kw):
    intel = SourceIntel(str(mini_repo["root"]), mini_repo["head"], mini_repo["base"])
    pr = PullRequest(
        number=7,
        title="drivers: i2c: foo: add bus recovery",
        body="Adds recovery.",
        author="t",
        base_ref="main",
        base_sha=mini_repo["base"],
        head_ref="pr",
        head_sha=mini_repo["head"],
        head_repo="t/zephyr",
        draft=False,
    )
    system = system_prompt(load_profile("zephyr"), "English")
    opening = opening_message(pr, intel.overview(), intel.diff_text, "", "zephyrproject-rtos/zephyr")
    changed = {f.path: f.new_side_lines for f in intel.files}
    async with ToolBridge(build_server(intel)) as bridge:
        run = await run_review(_llm(script), bridge, system, opening, changed, log=lambda m: None, **kw)
    return run, pr, intel


async def test_agent_investigates_then_submits(mini_repo):
    script = ScriptedOpenRouter(
        [
            _completion(
                {
                    "content": "",
                    "tool_calls": [
                        _tool_call(
                            "c1", "read_file", {"path": "drivers/i2c/i2c_foo.c", "start_line": 9, "end_line": 21}
                        ),
                        _tool_call("c2", "kconfig_symbol", {"name": "I2C_BUS_RECOVERY"}),
                    ],
                }
            ),
            _completion({"content": "", "tool_calls": [_tool_call("c3", "submit_review", GOOD_REVIEW)]}),
        ]
    )
    run, pr, intel = await _run(mini_repo, script)
    assert run.review is not None and run.steps == 2 and not run.forced
    assert run.tool_call_count == 2
    assert run.usage.requests == 2 and run.usage.cost > 0.02

    first, second = script.requests
    assert first["model"] == "anthropic/claude-sonnet-4.5" and first["models"] == [
        "anthropic/claude-sonnet-4.5",
        "openai/gpt-4.1",
    ]
    assert {t["function"]["name"] for t in first["tools"]} >= {"read_file", "submit_review", "kconfig_symbol"}
    assert first["messages"][0]["role"] == "system" and "Zephyr Project" in first["messages"][0]["content"]
    assert "### Full diff" in first["messages"][1]["content"]
    # Tool results were fed back with their ids, in order.
    tool_msgs = [m for m in second["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]
    assert "   17+| \tif (ret < 0) {" in tool_msgs[0]["content"]
    assert "No Kconfig definition of I2C_BUS_RECOVERY" in tool_msgs[1]["content"]

    review = run.review
    assert review.verdict == "request_changes" and review.highest_severity() == "high"
    lock, kconfig, unrelated, context = review.findings
    assert lock.inline and lock.line == 17 and lock.end_line == 19
    assert kconfig.inline and kconfig.line == 20
    assert not unrelated.inline and unrelated.line == 0 and "not changed by this pull request" in unrelated.notes[0]
    assert not context.inline and "not on the new side" in context.notes[0]

    body = render_comment(review, run, pr, repo="zephyrproject-rtos/zephyr", tag="zephyr-reviewer")
    # The posted comment carries no run statistics unless asked for.
    assert "tool call(s)" not in body and "$0.0" not in body and "anthropic/claude" not in body
    assert body.rstrip().endswith("It is advisory; verify before acting on it.</sub>")
    body = render_comment(review, run, pr, repo="zephyrproject-rtos/zephyr", tag="zephyr-reviewer", details=True)
    assert body.startswith("<!-- zephyr-reviewer -->")
    assert "🛑 **Request changes**" in body
    assert "blob/" + mini_repo["head"] + "/drivers/i2c/i2c_foo.c#L17" in body
    assert "```c\n\tif (ret < 0) {" in body
    assert "Evidence" not in body and "**Trigger:**" not in body and "**Problem:**" not in body
    assert "shown in the summary only" not in body  # internal notes stay in review.json
    assert "### Test gaps" in body and "2 tool call(s)" in body and "$0.025" in body

    inline = inline_comments(review)
    assert len(inline) == 3 and inline[2]["subject_type"] == "file"
    assert inline[0] == {
        "path": "drivers/i2c/i2c_foo.c",
        "line": 19,
        "side": "RIGHT",
        "start_line": 17,
        "start_side": "RIGHT",
        "body": inline[0]["body"],
    }
    assert "```suggestion" in inline[0]["body"]


async def test_agent_rejects_invalid_submission_then_accepts(mini_repo):
    bad = {"summary": "", "verdict": "maybe", "findings": "nope"}
    script = ScriptedOpenRouter(
        [
            _completion({"content": "", "tool_calls": [_tool_call("c1", "submit_review", bad)]}),
            _completion({"content": "", "tool_calls": [_tool_call("c2", "submit_review", GOOD_REVIEW)]}),
        ]
    )
    run, _, _ = await _run(mini_repo, script)
    assert run.review is not None and run.steps == 2
    rejection = [m for m in script.requests[1]["messages"] if m["role"] == "tool"][0]["content"]
    assert "summary is required" in rejection and "verdict must be one of" in rejection
    assert "findings must be an array" in rejection


async def test_agent_forces_submission_when_budget_runs_out(mini_repo):
    script = ScriptedOpenRouter(
        [
            _completion({"content": "", "tool_calls": [_tool_call("c1", "pr_overview", {})]}),
            _completion({"content": "", "tool_calls": [_tool_call("c2", "commit_messages", {})]}),
            _completion({"content": "", "tool_calls": [_tool_call("c3", "submit_review", GOOD_REVIEW)]}),
        ]
    )
    run, _, _ = await _run(mini_repo, script, max_steps=3)
    assert run.review is not None and run.forced
    last = script.requests[2]
    assert last["tool_choice"] == {"type": "function", "function": {"name": "submit_review"}}
    assert "budget is exhausted" in last["messages"][-1]["content"]


async def test_agent_nudges_prose_and_handles_unknown_tools_and_bad_json(mini_repo):
    script = ScriptedOpenRouter(
        [
            _completion({"content": "Let me think out loud instead of calling a tool."}),
            _completion(
                {
                    "content": "",
                    "tool_calls": [
                        {"id": "x1", "type": "function", "function": {"name": "read_file", "arguments": "{not json"}},
                        _tool_call("x2", "no_such_tool", {}),
                    ],
                }
            ),
            _completion({"content": "", "tool_calls": [_tool_call("c3", "submit_review", GOOD_REVIEW)]}),
        ]
    )
    run, _, _ = await _run(mini_repo, script)
    assert run.review is not None
    msgs = script.requests[2]["messages"]
    assert any(m["role"] == "user" and "prose instead of a tool call" in m["content"] for m in msgs)
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert "not valid JSON" in tool_msgs[0]["content"] and "unknown tool" in tool_msgs[1]["content"]


def test_arguments_sent_as_text_are_read_as_the_arguments_they_hold():
    from zephyr_reviewer.review.agent import tool_arguments

    args = {"concerns": [{"title": "t", "problem": "p"}], "dismissed": []}
    assert tool_arguments(json.dumps(args)) == args
    # What a model sends after being told that its arguments were not valid JSON: the same, as text.
    assert tool_arguments(json.dumps({"arguments": json.dumps(args)})) == args
    assert tool_arguments(json.dumps({"arguments": args})) == args
    assert tool_arguments(json.dumps(json.dumps(args))) == args
    assert tool_arguments({"arguments": json.dumps(args)}) == args
    # Anything that is not exactly one set of arguments is left as it came, for the tool to judge.
    assert tool_arguments('{"arguments": "not json"}') == {"arguments": "not json"}
    digits = "9" * 5000  # more than json converts: text that happens to be all digits stays the text it is
    assert tool_arguments(json.dumps({"arguments": digits})) == {"arguments": digits}
    assert tool_arguments(json.dumps(digits)) == digits
    assert tool_arguments('{"arguments": "[1, 2]"}') == {"arguments": "[1, 2]"}
    assert tool_arguments('{"arguments": "{}", "path": "a.c"}') == {"arguments": "{}", "path": "a.c"}
    assert tool_arguments('"plain text"') == "plain text" and tool_arguments("[1]") == [1]
    with pytest.raises(json.JSONDecodeError):
        tool_arguments('{"concerns": [')


async def test_a_submission_wrapped_up_as_text_is_accepted(mini_repo):
    wrapped = {"id": "c1", "type": "function", "function": {"name": "submit_review", "arguments": ""}}
    wrapped["function"]["arguments"] = json.dumps({"arguments": json.dumps(GOOD_REVIEW)})
    script = ScriptedOpenRouter([_completion({"content": "", "tool_calls": [wrapped]})])
    run, _, _ = await _run(mini_repo, script)
    assert run.review is not None and run.errors == [] and len(script.requests) == 1
    assert run.review.summary == GOOD_REVIEW["summary"]


async def test_a_call_cut_off_in_its_arguments_is_told_so(mini_repo):
    half = {"id": "c1", "type": "function", "function": {"name": "submit_review", "arguments": ""}}
    half["function"]["arguments"] = json.dumps(GOOD_REVIEW)[:-40]
    cut = _completion({"content": "", "tool_calls": [half]})
    cut["usage"]["completion_tokens"] = 8000  # the whole limit, though the answer says it ended for the tool call
    script = ScriptedOpenRouter(
        [cut, _completion({"content": "", "tool_calls": [_tool_call("c2", "submit_review", GOOD_REVIEW)]})]
    )
    run, _, _ = await _run(mini_repo, script)
    assert run.review is not None and len(script.requests) == 2
    (answer,) = [m for m in script.requests[1]["messages"] if m["role"] == "tool"]
    assert "reached the output limit before its arguments were complete" in answer["content"]
    assert "`submit_review`" in answer["content"] and "not valid JSON" not in answer["content"]

    # Cut off before the first argument, what is left can be an empty object, and cut off later, an object
    # with the fields that were complete. Neither is a submission to reject for what it lacks: more of them
    # than a run has rejections do not end it, and none is told which fields are missing.
    def at_limit(args: dict) -> dict:
        reply = _completion({"content": "", "tool_calls": [_tool_call("e", "submit_review", args)]})
        reply["usage"]["completion_tokens"] = 8000
        return reply

    good = _completion({"content": "", "tool_calls": [_tool_call("c2", "submit_review", GOOD_REVIEW)]})
    script = ScriptedOpenRouter([at_limit({}), at_limit({"summary": "s"}), at_limit({}), good])
    run, _, _ = await _run(mini_repo, script)
    assert run.review is not None and run.errors == [] and len(script.requests) == 4
    answers = [m["content"] for m in script.requests[3]["messages"] if m["role"] == "tool"]
    assert len(answers) == 3 and all("reached the output limit" in a for a in answers)
    assert not any("rejected" in a or "verdict must be" in a for a in answers)
    # The same submissions in answers with room left are rejected for what they lack, and counted.
    plain = _completion({"content": "", "tool_calls": [_tool_call("e", "submit_review", {})]})
    script = ScriptedOpenRouter([plain, plain, plain, good])
    run, _, _ = await _run(mini_repo, script)
    assert run.review is None and len(script.requests) == 3 and "summary is required" in run.errors
    # An answer that ends at the limit for a reason of its own is not one that was cut off.
    filtered = at_limit({})
    filtered["choices"][0]["finish_reason"] = "content_filter"
    script = ScriptedOpenRouter([filtered, good])
    run, _, _ = await _run(mini_repo, script)
    (answer,) = [m["content"] for m in script.requests[1]["messages"] if m["role"] == "tool"]
    assert run.review is not None and "rejected" in answer and "output limit" not in answer

    # The same broken arguments in an answer that had room left are the model's own mistake, as before.
    short = _completion({"content": "", "tool_calls": [half]})
    script = ScriptedOpenRouter(
        [short, _completion({"content": "", "tool_calls": [_tool_call("c2", "submit_review", GOOD_REVIEW)]})]
    )
    run, _, _ = await _run(mini_repo, script)
    (answer,) = [m for m in script.requests[1]["messages"] if m["role"] == "tool"]
    assert run.review is not None and "not valid JSON" in answer["content"] and "output limit" not in answer["content"]


async def test_only_the_last_call_of_an_answer_is_the_one_cut_off(mini_repo):
    # A complete submission with a mistake in it, then a lookup that the output limit cut off: the submission
    # is told its mistake, as in any answer, and only the lookup is told that it was cut off.
    half = {"id": "c2", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "drivers/i2c'}}
    both = _completion({"content": "", "tool_calls": [_tool_call("c1", "submit_review", {"summary": "s"}), half]})
    both["usage"]["completion_tokens"] = 8000
    both["choices"][0]["finish_reason"] = "length"
    good = _completion({"content": "", "tool_calls": [_tool_call("c3", "submit_review", GOOD_REVIEW)]})
    script = ScriptedOpenRouter([both, good])
    run, _, _ = await _run(mini_repo, script)
    assert run.review is not None
    first, second = [m["content"] for m in script.requests[1]["messages"] if m["role"] == "tool"]
    assert "rejected" in first and "verdict must be one of" in first and "output limit" not in first
    assert "`read_file` reached the output limit" in second


async def test_a_run_whose_submissions_were_all_cut_off_says_so(mini_repo):
    def cut_off():
        half = {"id": "c", "type": "function", "function": {"name": "submit_review", "arguments": ""}}
        half["function"]["arguments"] = json.dumps(GOOD_REVIEW)[:-40]
        reply = _completion({"content": "", "tool_calls": [half]})
        reply["usage"]["completion_tokens"] = 8000
        return reply

    script = ScriptedOpenRouter([cut_off() for _ in range(8)])
    run, _, _ = await _run(mini_repo, script, max_steps=1)
    assert run.review is None and len(script.requests) == 4  # the one step and its three forced turns
    assert run.errors == [
        "4 call(s) of submit_review were cut off at the output limit and none got through within 1 steps (+3 forced)"
    ]


async def test_a_lookup_cut_off_when_the_run_is_asked_to_submit_is_not_asked_for_again(mini_repo):
    half = {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "drivers/i2c'}}
    cut = _completion({"content": "", "tool_calls": [half]})
    cut["usage"]["completion_tokens"] = 8000
    good = _completion({"content": "", "tool_calls": [_tool_call("c2", "submit_review", GOOD_REVIEW)]})
    script = ScriptedOpenRouter([cut, good])
    run, _, _ = await _run(mini_repo, script, max_steps=1)  # the first turn is already the last: forced
    assert run.review is not None and run.forced
    (answer,) = [m["content"] for m in script.requests[1]["messages"] if m["role"] == "tool"]
    assert answer.startswith("Not run: the budget is exhausted.") and "`submit_review`" in answer
    assert "Call it again" not in answer and "not valid JSON" not in answer
    # With turns left, the same cut-off lookup is asked for again.
    script = ScriptedOpenRouter([cut, good])
    run, _, _ = await _run(mini_repo, script)
    (answer,) = [m["content"] for m in script.requests[1]["messages"] if m["role"] == "tool"]
    assert run.review is not None and "`read_file` reached the output limit" in answer and "Call it again" in answer


async def test_the_transcript_keeps_what_a_refused_submission_said(mini_repo):
    half = {"id": "c1", "type": "function", "function": {"name": "submit_review", "arguments": ""}}
    half["function"]["arguments"] = json.dumps(GOOD_REVIEW)[:-40]
    cut = _completion({"content": "", "tool_calls": [half]})
    cut["usage"]["completion_tokens"] = 8000
    long = {"summary": "x" * 9000}
    nothing = _completion({"content": "", "tool_calls": [_tool_call("c0", "submit_review", {})]})
    nothing["usage"]["completion_tokens"] = 8000
    script = ScriptedOpenRouter(
        [
            cut,
            _completion({"content": "", "tool_calls": [half]}),
            nothing,
            _completion({"content": "", "tool_calls": [_tool_call("c2", "submit_review", {"summary": "x"})]}),
            _completion({"content": "", "tool_calls": [_tool_call("c3", "submit_review", long)]}),
            _completion({"content": "", "tool_calls": [_tool_call("c4", "submit_review", GOOD_REVIEW)]}),
        ]
    )
    run, _, _ = await _run(mini_repo, script)
    assert run.review is not None
    unreadable = [t for t in run.transcript if t["kind"] == "tool_call_unreadable"]
    assert [(t["step"], t["name"], t["cut_off"]) for t in unreadable] == [
        (1, "submit_review", True),
        (2, "submit_review", False),
    ]
    assert all(t["arguments"] == half["function"]["arguments"] and t["error"] for t in unreadable)
    submissions = [t for t in run.transcript if t["kind"] == "tool_call" and t.get("terminal")]
    assert [(t["step"], t["ok"]) for t in submissions] == [(3, False), (4, False), (5, False), (6, True)]
    # Parsed, but nothing in it, from an answer at the output limit: marked as the cut-off call it was.
    assert (submissions[0]["arguments"], submissions[0]["cut_off"]) == ("{}", True) and submissions[0]["errors"]
    assert (submissions[1]["arguments"], submissions[1]["cut_off"]) == ('{"summary": "x"}', False)
    assert any(e.startswith("verdict must be one of") for e in submissions[1]["errors"])
    # A long one is kept to its beginning, with a note of how much is missing.
    assert submissions[2]["arguments"].startswith('{"summary": "xxxx') and len(submissions[2]["arguments"]) < 4100
    assert submissions[2]["arguments"].endswith("more characters]")
    assert "arguments" not in submissions[3] and "cut_off" not in submissions[3]  # accepted: it is in the result
    # Neither kind is a lookup.
    assert run.tool_call_count == 0


async def test_agent_gives_up_after_repeated_rejections(mini_repo):
    bad = _completion({"content": "", "tool_calls": [_tool_call("c", "submit_review", {"summary": "x"})]})
    script = ScriptedOpenRouter([bad, bad, bad, bad])
    run, _, _ = await _run(mini_repo, script, max_rejections=2)
    assert run.review is None and run.errors and "verdict" in run.errors[0]


def test_openrouter_retries_then_fails_cleanly():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503, text="busy", headers={"Retry-After": "0"})

    llm = OpenRouter("k", ["m"], client=httpx.Client(transport=httpx.MockTransport(handler)), max_attempts=2)
    import pytest

    from zephyr_reviewer.review.openrouter import OpenRouterError

    llm._sleep = lambda *a: None  # type: ignore[method-assign]
    with pytest.raises(OpenRouterError):
        llm.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 2


def test_a_parsed_review_with_a_finding_is_not_an_approval_and_not_all_clear():
    from zephyr_reviewer.review.schema import says_all_clear

    lines = {"drivers/i2c/i2c_foo.c": {17, 18, 19}}
    review, errors = parse_review({**GOOD_REVIEW, "verdict": "approve", "summary": "No issues found."}, lines)
    assert not errors and review.findings
    assert review.verdict == "comment" and review.summary == "Automated review completed; see findings."
    # A summary that names what is wrong stays, and so does a verdict that is not an approval.
    review, _ = parse_review({**GOOD_REVIEW, "verdict": "approve"}, lines)
    assert review.verdict == "comment" and review.summary == GOOD_REVIEW["summary"]
    review, _ = parse_review(GOOD_REVIEW, lines)
    assert review.verdict == "request_changes"
    # Without findings an approval that says all is clear is what it is.
    review, _ = parse_review(
        {**GOOD_REVIEW, "verdict": "approve", "summary": "No issues found.", "findings": []}, lines
    )
    assert review.verdict == "approve" and review.summary == "No issues found."
    assert says_all_clear("No defects or regressions identified.") is False  # not one of the known phrasings
    assert says_all_clear("Looks good.") and says_all_clear("No issues found; however the lock leaks.") is False
    assert not says_all_clear("") and not says_all_clear("No issues found. " + "x" * 300)


def test_parse_review_keeps_only_problems_in_assessments():
    base = {"summary": "ok", "verdict": "approve", "confidence": "high", "findings": []}
    clear = {
        "test_assessment": "Tests are adequate for this change.",
        "commit_message_assessment": "No issues found.",
        "discussion_assessment": "All threads have been addressed.",
    }
    review, _ = parse_review({**base, **clear}, {})
    assert review is not None
    assert (
        review.test_assessment == "" and review.commit_message_assessment == "" and review.discussion_assessment == ""
    )
    kept = {
        "test_assessment": "The recovery scenario enables the option but does not exercise foo_recover().",
        "commit_message_assessment": "The second commit mixes the Kconfig change with the fix; split it.",
        "discussion_assessment": "@bob's request to release the lock on the error path is still unaddressed.",
    }
    review, _ = parse_review({**base, **kept}, {})
    assert review is not None
    assert review.test_assessment == kept["test_assessment"]
    assert review.commit_message_assessment == kept["commit_message_assessment"]
    assert review.discussion_assessment == kept["discussion_assessment"]
    # Short but naming a problem: kept. Long prose: kept even if it opens positively.
    review, _ = parse_review({**base, "test_assessment": "Adequate, but the error path is untested."}, {})
    assert review is not None and review.test_assessment
    review, _ = parse_review({**base, "test_assessment": "Looks good. " + "x" * 300}, {})
    assert review is not None and review.test_assessment


def test_an_alias_of_another_field_is_not_a_value_of_this_one():
    finding = {"file": "a.c", "line": 3, "title": "t", "problem": "p"}
    # "major", "blocker" and "nit" are other words for a severity. Where the severity they stand for is not
    # a value of the field, they are nothing: "major" is no verdict, "blocker" and "nit" no confidence.
    review, errors = parse_review({"summary": "ok", "verdict": "major", "findings": []}, {})
    assert review is None and any(e.startswith("verdict must be one of") for e in errors)
    review, errors = parse_review(
        {
            "summary": "ok",
            "verdict": "changes requested",
            "confidence": "blocker",
            "findings": [{**finding, "severity": "major", "category": "nit", "confidence": "nit"}],
        },
        {"a.c": {3}},
    )
    assert not errors and review.verdict == "request_changes" and review.confidence == "medium"
    f = review.findings[0]
    assert (f.severity, f.category, f.confidence) == ("high", "bug", "medium")


def test_parse_review_normalizes_vocabulary():
    review, errors = parse_review(
        {
            "summary": "ok",
            "verdict": "request changes",
            "confidence": "HIGH",
            "findings": [
                {
                    "file": "b/a.c",
                    "line": "3",
                    "severity": "Suggestion",
                    "category": "weird",
                    "title": "t",
                    "problem": "p",
                    "confidence": "?",
                }
            ],
        },
        {"a.c": {3}},
    )
    assert not errors and review is not None
    assert review.verdict == "request_changes" and review.confidence == "high"
    f = review.findings[0]
    assert (
        f.file == "a.c" and f.inline and f.severity == "suggestion" and f.category == "bug" and f.confidence == "medium"
    )


def test_publish_upserts_comment_and_posts_inline_review(mini_repo, tmp_path, monkeypatch):
    seen: list[tuple[str, str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        seen.append((request.method, request.url.path, body))
        if request.method == "GET" and request.url.path.endswith("/issues/7/comments"):
            return httpx.Response(200, json=[{"id": 42, "body": "<!-- zephyr-reviewer -->\nold"}])
        if request.method == "GET" and request.url.path.endswith("/pulls/7/comments"):
            return httpx.Response(200, json=[])
        if request.method == "PATCH":
            return httpx.Response(200, json={"html_url": "https://github.com/o/r/pull/7#issuecomment-42"})
        if request.url.path.endswith("/reviews"):
            return httpx.Response(200, json={"html_url": "https://github.com/o/r/pull/7#pullrequestreview-1"})
        return httpx.Response(404, json={})

    gh = GitHub("tok", "o/r", client=httpx.Client(transport=httpx.MockTransport(handler)))
    review, _ = parse_review(GOOD_REVIEW, {"drivers/i2c/i2c_foo.c": {17, 18, 19}, "drivers/i2c/Kconfig": {20}})
    from zephyr_reviewer.review.agent import RunResult
    from zephyr_reviewer.review.openrouter import Usage

    run = RunResult(review, 3, Usage(), ["m"], False)
    pr = PullRequest(7, "t", "", "a", "main", "b" * 40, "pr", "h" * 40, "o/r", False)
    cfg = ReviewConfig(
        post_inline=True, comment_tag="zephyr-reviewer", fail_on_severity="high", output_dir=str(tmp_path / "out")
    )
    out_file = tmp_path / "outputs.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out_file))

    outputs = pub.publish(cfg, gh, pr, run, lambda exclude: "<!-- zephyr-reviewer -->\nnew", log=lambda m: None)
    pub.write_outputs(outputs)
    pub.save_artifacts(cfg.output_dir, run, "body", pr, cfg)

    methods = [(m, p) for m, p, _ in seen]
    assert ("PATCH", "/repos/o/r/issues/comments/42") in methods  # updated in place, not duplicated
    review_req = next(b for m, p, b in seen if p.endswith("/reviews"))
    assert review_req["commit_id"] == "h" * 40 and len(review_req["comments"]) == 2
    assert outputs["comment_url"].endswith("issuecomment-42") and outputs["highest_severity"] == "high"
    assert "verdict=request_changes" in out_file.read_text()
    assert pub.gate_fails(run, cfg) is not None
    assert (tmp_path / "out" / "review.json").exists() and (tmp_path / "out" / "transcript.jsonl").exists()
    saved = json.loads((tmp_path / "out" / "review.json").read_text())
    assert saved["review"]["findings"][0]["title"] == "Early return leaks the bus lock"


async def test_forced_submission_gets_grace_steps_and_falls_back_to_required(mini_repo):
    """A provider that answers a forced tool_choice with an empty message no longer loses the run."""
    script = ScriptedOpenRouter(
        [
            _completion({"content": "", "tool_calls": [_tool_call("c1", "pr_overview", {})]}),
            _completion({"content": "", "tool_calls": [_tool_call("c2", "commit_messages", {})]}),
            _completion({"content": ""}),  # step 3: forced, empty
            _completion({"content": ""}),  # step 4 (grace): forced, empty again
            _completion({"content": "", "tool_calls": [_tool_call("c5", "submit_review", GOOD_REVIEW)]}),  # step 5
        ]
    )
    run, _, _ = await _run(mini_repo, script, max_steps=3)
    assert run.review is not None and run.forced and run.steps == 5
    choices = [r["tool_choice"] for r in script.requests]
    assert choices[2] == {"type": "function", "function": {"name": "submit_review"}}
    assert choices[3] == {"type": "function", "function": {"name": "submit_review"}}
    assert choices[4] == "required"  # after two empty forced replies, any tool is accepted and steered

    # Grace is bounded: a model that never submits still ends, with the error naming the budget.
    never = ScriptedOpenRouter([_completion({"content": ""}) for _ in range(10)])
    run, _, _ = await _run(mini_repo, never, max_steps=2)
    assert run.review is None and "within 2 steps (+3 forced)" in run.errors[0]


async def test_tool_calls_in_one_step_run_concurrently_and_answer_in_order(mini_repo):
    import asyncio
    import time

    from zephyr_reviewer.review.agent import LocalTool

    async def slow(args):
        await asyncio.sleep(0.3)
        return f"answer for {args['n']}"

    tool = LocalTool(
        {
            "type": "function",
            "function": {"name": "slow", "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}}},
        },
        slow,
    )
    script = ScriptedOpenRouter(
        [
            _completion(
                {
                    "content": "",
                    "tool_calls": [
                        _tool_call("c1", "slow", {"n": 1}),
                        _tool_call("c2", "read_file", {"path": "drivers/i2c/i2c_foo.c"}),
                        _tool_call("c3", "slow", {"n": 3}),
                        _tool_call("c4", "nope", {}),
                    ],
                }
            ),
            _completion({"content": "", "tool_calls": [_tool_call("c5", "submit_review", GOOD_REVIEW)]}),
        ]
    )
    t0 = time.monotonic()
    run, _, _ = await _run(mini_repo, script, extra_tools=[tool])
    assert run.review is not None and time.monotonic() - t0 < 0.55  # the two 0.3s lookups overlapped
    answers = [m for m in script.requests[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in answers] == ["c1", "c2", "c3", "c4"]
    assert answers[0]["content"] == "answer for 1" and answers[2]["content"] == "answer for 3"
    assert "k_sem_take" in answers[1]["content"] and answers[3]["content"].startswith("ERROR: unknown tool")
    recorded = [t for t in run.transcript if t.get("kind") == "tool_call" and not t.get("terminal")]
    assert [t["name"] for t in recorded].count("slow") == 2 and all(t["seconds"] >= 0 for t in recorded)


def _reasoned_away(finish_reason: str = "length") -> dict:
    """A reply whose reasoning used the whole output limit: no text, no tool call."""
    message = {"content": "", "reasoning_details": [{"type": "reasoning.text", "text": "thinking " * 4000}]}
    reply = _completion(message)
    reply["choices"][0]["finish_reason"] = finish_reason
    reply["usage"]["completion_tokens"] = 8000
    reply["usage"]["completion_tokens_details"] = {"reasoning_tokens": 8000}
    return reply


def _submit(call_id: str = "s") -> dict:
    return _completion({"content": "", "tool_calls": [_tool_call(call_id, "submit_review", GOOD_REVIEW)]})


NAMED = {"type": "function", "function": {"name": "submit_review"}}


async def test_an_answer_reasoned_away_is_asked_for_again_with_less_reasoning(mini_repo):
    read = _tool_call("c1", "read_file", {"path": "drivers/i2c/i2c_foo.c", "start_line": 9, "end_line": 21})
    script = ScriptedOpenRouter([_completion({"content": "", "tool_calls": [read]}), _reasoned_away(), _submit()])
    run, _, _ = await _run(mini_repo, script)
    assert run.review is not None and run.errors == [] and run.steps == 3
    first, second, third = script.requests
    # The same request again: nothing of the lost turn is in the history, not its 8000 tokens of reasoning
    # and no word put in the model's mouth. It only asks for less reasoning, from here on.
    assert third["messages"] == second["messages"] and "thinking thinking" not in json.dumps(third["messages"])
    assert "reasoning" not in first and "reasoning" not in second and third["reasoning"] == {"effort": "low"}
    assert [r["tool_choice"] for r in script.requests] == ["auto", "auto", "auto"]
    cut = [t for t in run.transcript if t.get("cut_off")]
    assert len(cut) == 1 and cut[0]["step"] == 2 and cut[0]["usage"]["completion_tokens"] == 8000


async def test_a_model_that_keeps_reasoning_its_answers_away_is_asked_to_submit_and_then_stopped(mini_repo):
    script = ScriptedOpenRouter([_reasoned_away() for _ in range(40)])
    run, _, _ = await _run(mini_repo, script, max_steps=25)
    # Two tries at going on, four at submitting (twice the tool by name, then any tool): not 28 requests,
    # each longer than the last.
    assert run.review is None and len(script.requests) == 6
    assert [r["tool_choice"] for r in script.requests] == ["auto", "auto", NAMED, NAMED, "required", "required"]
    assert run.errors == [
        "6 answers in a row ended at the output limit with nothing in them, 4 of them when asked to submit"
    ]
    sizes = [len(json.dumps(r["messages"])) for r in script.requests]
    assert sizes[1] == sizes[0] and sizes[5] == sizes[2] and sizes[2] - sizes[1] < 500  # one sentence, once
    # Nothing was kept of the two turns, so the instruction joins the opening message: no two user
    # messages in a row, which some routes reject.
    assert [m["role"] for m in script.requests[2]["messages"]] == ["system", "user"]
    asked = script.requests[2]["messages"][-1]["content"]
    assert asked.startswith(script.requests[0]["messages"][-1]["content"])
    assert "reached the output limit before any text or tool call arrived" in asked
    assert "call `submit_review` now" in asked
    # One step down while it goes on, all the way down once it is asked to submit.
    assert [r.get("reasoning") for r in script.requests] == [None, {"effort": "low"}] + [{"effort": "none"}] * 4


async def test_a_forced_submission_reasoned_away_still_gets_every_way_through(mini_repo):
    # The step budget is used up; the model is asked for its review and reasons the answer away twice.
    # As for any empty answer to a forced choice, the third request takes any tool.
    script = ScriptedOpenRouter([_reasoned_away(), _reasoned_away(), _submit()])
    run, _, _ = await _run(mini_repo, script, max_steps=1)
    assert run.review is not None and run.forced
    assert [r["tool_choice"] for r in script.requests] == [NAMED, NAMED, "required"]
    # ... and the fourth is its last, as it is for a run without cut-off answers.
    script = ScriptedOpenRouter([_reasoned_away() for _ in range(10)])
    run, _, _ = await _run(mini_repo, script, max_steps=1)
    assert run.review is None and len(script.requests) == 4
    assert "4 of them when asked to submit" in run.errors[0]


async def test_only_an_empty_answer_at_the_output_limit_counts_and_only_in_a_row(mini_repo):
    read = _tool_call("c1", "read_file", {"path": "drivers/i2c/i2c_foo.c", "start_line": 9, "end_line": 21})
    cut_text = _completion({"content": "The lock is"})  # cut off, but something arrived: prose, as before
    cut_text["choices"][0]["finish_reason"] = "length"
    filtered = _reasoned_away("content_filter")  # empty for another reason, whatever its token count
    script = ScriptedOpenRouter(
        [
            _reasoned_away(),
            _completion({"content": "", "tool_calls": [read]}),  # an answer in between starts the count again
            _reasoned_away(""),  # no reason given, the whole limit used: the same thing
            cut_text,
            filtered,
            _submit(),
        ]
    )
    run, _, _ = await _run(mini_repo, script)
    assert run.review is not None and len(script.requests) == 6
    assert [t["step"] for t in run.transcript if t.get("cut_off")] == [1, 3]
    assert [r["tool_choice"] for r in script.requests] == ["auto"] * 6  # never two in a row: never forced
    nudges = [m["content"] for m in script.requests[5]["messages"] if m["role"] == "user"][-2:]
    assert all("You replied with prose instead of a tool call" in n for n in nudges)


async def test_a_route_that_refuses_the_reasoning_field_gets_the_request_without_it(mini_repo):
    replies = [_reasoned_away(), _submit()]
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if "reasoning" in payload:
            return httpx.Response(400, json={"error": {"message": "reasoning is not supported", "code": 400}})
        return httpx.Response(200, json=replies.pop(0))

    script = ScriptedOpenRouter([])
    script.handler = handler  # type: ignore[method-assign]
    run, _, _ = await _run(mini_repo, script)
    assert run.review is not None and run.errors == []
    assert ["reasoning" in r for r in requests] == [False, True, False]
    assert requests[2]["messages"] == requests[0]["messages"]


async def test_only_a_refusal_of_the_reasoning_field_is_retried_without_it(mini_repo):
    def scripted(status: int, message: str):
        requests: list[dict] = []
        replies = [_reasoned_away(), _submit()]

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            requests.append(payload)
            if "reasoning" in payload:
                return httpx.Response(status, json={"error": {"message": message, "code": status}})
            return httpx.Response(200, json=replies.pop(0))

        script = ScriptedOpenRouter([])
        script.handler = handler  # type: ignore[method-assign]
        return script, requests

    # A rejected key is a rejected key with or without the field: the run fails as it would have.
    script, requests = scripted(401, "No auth credentials found")
    run, _, _ = await _run(mini_repo, script)
    assert run.review is None and len(requests) == 2
    assert run.errors and run.errors[0].startswith("model request failed: HTTP 401")
    # So is an outage, after the attempts the client makes: no second round of them without the field.
    script, requests = scripted(503, "overloaded")
    run, _, _ = await _run(mini_repo, script)
    assert run.review is None and run.errors[0].startswith("model request failed: HTTP 503")
    assert ["reasoning" in r for r in requests] == [False, True, True]
    # A refusal worded like a size limit is still a refusal of the field.
    script, requests = scripted(400, "reasoning token limit is unsupported for this model")
    run, _, _ = await _run(mini_repo, script)
    assert run.review is not None and ["reasoning" in r for r in requests] == [False, True, False]


async def test_asked_to_submit_a_cut_off_answer_turns_reasoning_off_at_once(mini_repo):
    # A model that reasons its limit away at low effort as at default gets its submission through with
    # reasoning switched off, and a forced submission has few turns to find that out in.
    script = ScriptedOpenRouter([_reasoned_away(), _submit()])
    run, _, _ = await _run(mini_repo, script, max_steps=1)
    assert run.review is not None and run.steps == 2
    assert [r.get("reasoning") for r in script.requests] == [None, {"effort": "none"}]
    assert [r["tool_choice"] for r in script.requests] == [NAMED, NAMED]


async def test_a_setting_a_route_refuses_falls_back_within_the_step_and_is_not_asked_for_again(mini_repo):
    def scripted(refuse: dict, replies: list[dict]):
        requests: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            requests.append(payload)
            if payload.get("reasoning") == refuse:
                message = "Reasoning is mandatory for this endpoint and cannot be disabled."
                return httpx.Response(400, json={"error": {"message": message, "code": 400}})
            return httpx.Response(200, json=replies.pop(0))

        script = ScriptedOpenRouter([])
        script.handler = handler  # type: ignore[method-assign]
        return script, requests

    # Reasoning cannot be switched off on this route: asked to submit, the request falls back to low
    # effort in the same step, and a later cut-off does not try "none" again.
    script, requests = scripted({"effort": "none"}, [_reasoned_away(), _reasoned_away(), _submit()])
    run, _, _ = await _run(mini_repo, script, max_steps=1)
    assert run.review is not None and run.steps == 3
    assert [r.get("reasoning") for r in requests] == [None, {"effort": "none"}, {"effort": "low"}, {"effort": "low"}]
    # Low effort is not taken on this route: the request goes out without the field. A later cut-off while
    # the investigation goes on changes nothing, and one when asked to submit switches reasoning off.
    read = _tool_call("c1", "read_file", {"path": "drivers/i2c/i2c_foo.c", "start_line": 9, "end_line": 21})
    lookup = _completion({"content": "", "tool_calls": [read]})
    replies = [_reasoned_away(), lookup, _reasoned_away(), lookup, _reasoned_away(), _submit()]
    script, requests = scripted({"effort": "low"}, replies)
    run, _, _ = await _run(mini_repo, script, max_steps=5)
    assert run.review is not None and run.steps == 6
    assert [r.get("reasoning") for r in requests] == [
        None,
        {"effort": "low"},
        None,
        None,
        None,
        None,
        {"effort": "none"},
    ]
    # Both refused: the request is made a third time in the same step, as the model's default.
    requests = []

    def neither(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if "reasoning" in payload:
            return httpx.Response(400, json={"error": {"message": "Unsupported effort value", "code": 400}})
        return httpx.Response(200, json=(_submit() if len(requests) > 2 else _reasoned_away()))

    script = ScriptedOpenRouter([])
    script.handler = neither  # type: ignore[method-assign]
    run, _, _ = await _run(mini_repo, script, max_steps=1)
    assert run.review is not None and run.steps == 2
    assert [r.get("reasoning") for r in requests] == [None, {"effort": "none"}, {"effort": "low"}, None]


async def test_reasoning_is_switched_off_only_for_a_submission_that_is_asked_for(mini_repo):
    # Cut off again and again while it investigates, the model stays at low effort: with reasoning switched
    # off it would go on as another reviewer than the one that was asked for.
    read = _tool_call("c1", "read_file", {"path": "drivers/i2c/i2c_foo.c", "start_line": 9, "end_line": 21})
    lookup = _completion({"content": "", "tool_calls": [read]})
    script = ScriptedOpenRouter([_reasoned_away(), lookup, _reasoned_away(), lookup, _reasoned_away(), _submit()])
    run, _, _ = await _run(mini_repo, script)
    assert run.review is not None and not run.forced
    assert [r.get("reasoning") for r in script.requests] == [None] + [{"effort": "low"}] * 5


async def test_a_failure_that_is_not_about_the_setting_is_not_taken_for_a_refusal(mini_repo):
    def scripted(status: int, message: str):
        requests: list[dict] = []
        replies = [_reasoned_away(), _submit()]

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            requests.append(payload)
            if "reasoning" in payload:
                return httpx.Response(status, json={"error": {"message": message, "code": status}})
            return httpx.Response(200, json=replies.pop(0))

        script = ScriptedOpenRouter([])
        script.handler = handler  # type: ignore[method-assign]
        return script, requests

    # A rejected key and an outage are what they are, also when their text happens to say "reasoning".
    for status, message in ((401, "Reasoning service denied for this key"), (503, "reasoning backend overloaded")):
        script, requests = scripted(status, message)
        run, _, _ = await _run(mini_repo, script)
        assert run.review is None and run.errors[0].startswith(f"model request failed: HTTP {status}")
        assert all("reasoning" in r for r in requests[1:])  # never retried without the field
    # A client error about the effort value is a refusal, whether or not it says "reasoning".
    script, requests = scripted(400, "Unsupported effort value: low")
    run, _, _ = await _run(mini_repo, script)
    assert run.review is not None and ["reasoning" in r for r in requests] == [False, True, False]

    # The same holds for an error that arrives in the body of a 200 response, where the code is all there is.
    def in_band(code: int, message: str):
        requests: list[dict] = []
        replies = [_reasoned_away(), _submit()]

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            requests.append(payload)
            if "reasoning" in payload:
                return httpx.Response(200, json={"error": {"message": message, "code": code}})
            return httpx.Response(200, json=replies.pop(0))

        script = ScriptedOpenRouter([])
        script.handler = handler  # type: ignore[method-assign]
        return script, requests

    script, requests = in_band(401, "effort service denied")
    run, _, _ = await _run(mini_repo, script)
    assert run.review is None and ["reasoning" in r for r in requests] == [False, True]
    script, requests = in_band(400, "Unsupported effort value")
    run, _, _ = await _run(mini_repo, script)
    assert run.review is not None and ["reasoning" in r for r in requests] == [False, True, False]


async def test_a_lookup_asked_for_after_the_budget_is_not_run(mini_repo):
    # Two empty answers to the forced choice, then any tool is accepted, and the model asks to read a file.
    read = _tool_call("c1", "read_file", {"path": "drivers/i2c/i2c_foo.c", "start_line": 9, "end_line": 21})
    prose = _completion({"content": "I will submit now."})
    script = ScriptedOpenRouter([prose, prose, _completion({"content": "", "tool_calls": [read]}), _submit()])
    run, _, _ = await _run(mini_repo, script, max_steps=1)
    assert run.review is not None and run.steps == 4
    assert [r["tool_choice"] for r in script.requests] == [NAMED, NAMED, "required", "required"]
    answer = script.requests[3]["messages"][-1]
    assert answer["role"] == "tool" and answer["name"] == "read_file"
    assert answer["content"].startswith("Not run: the budget is exhausted.") and "submit_review" in answer["content"]
    assert run.tool_results == [] and run.tool_call_count == 0  # nothing was read, and nothing counts as read
    assert [t["name"] for t in run.transcript if t["kind"] == "tool_refused"] == ["read_file"]
    # With any tool accepted, the terminal tool is the only one offered.
    offered = [[t["function"]["name"] for t in r["tools"]] for r in script.requests]
    assert len(offered[0]) > 1 and offered[0] == offered[1] and offered[2] == offered[3] == ["submit_review"]
    # Before the budget is out the same call is run.
    script = ScriptedOpenRouter([_completion({"content": "", "tool_calls": [read]}), _submit()])
    run, _, _ = await _run(mini_repo, script)
    assert [r["name"] for r in run.tool_results] == ["read_file"]
