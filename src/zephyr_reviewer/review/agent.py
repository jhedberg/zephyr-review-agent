# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""The review loop: model proposes tool calls, the bridge answers, until submit_review."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .mcp_client import ToolBridge
from .openrouter import ContextLengthError, OpenRouter, OpenRouterError, Usage, assistant_message_for_history
from .prompts import budget_exhausted, cut_off, cut_off_call, nudge, rejection_message
from .schema import SUBMIT_REVIEW_TOOL, SUBMIT_TOOL_NAME, Review, parse_review
from .text import without_surrogates


def tool_arguments(raw: str | dict[str, Any]) -> dict[str, Any]:
    """A tool call's arguments as the model sent them, parsed and made encodable: a \\ud83d escape in the
    JSON becomes a lone surrogate on parsing, which no transport, file or request could carry.

    A model told that its arguments were not valid JSON may send them again as text inside an object of
    its own making, ``{"arguments": "{...}"}``, or as one JSON string. Either holds exactly one set of
    arguments, so it is read as those, and not answered with errors about fields that are in fact there."""
    args = json.loads(raw) if isinstance(raw, str) else dict(raw)
    inner = args["arguments"] if isinstance(args, dict) and set(args) == {"arguments"} else args
    if isinstance(inner, str):
        try:
            inner = json.loads(inner)
        except (ValueError, RecursionError):  # not JSON, or JSON that json will not load (a 5000-digit number)
            inner = None
    return without_surrogates(inner if isinstance(inner, dict) else args)


@dataclass
class LocalTool:
    """A tool answered by the orchestrator itself rather than the MCP server (e.g. PR discussion)."""

    schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[str]]

    @property
    def name(self) -> str:
        return self.schema["function"]["name"]


@dataclass
class TerminalTool:
    """The tool whose successful call ends an agent run: its schema, a parser returning (output, errors),
    and a one-line description of the output for the log."""

    schema: dict[str, Any]
    parse: Callable[[dict[str, Any]], tuple[Any, list[str]]]
    describe: Callable[[Any], str] = lambda out: "submitted"

    @property
    def name(self) -> str:
        return self.schema["function"]["name"]


@dataclass
class AgentResult:
    output: Any
    steps: int
    usage: Usage
    models_used: list[str]
    forced: bool
    errors: list[str] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    tool_results: list[dict[str, Any]] = field(default_factory=list)

    @property
    def tool_call_count(self) -> int:
        return sum(1 for t in self.transcript if t.get("kind") == "tool_call" and not t.get("terminal"))


@dataclass
class RunResult:
    review: Review | None
    steps: int
    usage: Usage
    models_used: list[str]
    forced: bool
    errors: list[str] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    tool_results: list[dict[str, Any]] = field(default_factory=list)  # name, arguments, content (for a later stage)
    stages: list[dict[str, Any]] = field(default_factory=list)  # per-stage summaries when the review ran in stages
    review_map: dict[str, Any] | None = None  # concern lineage when the review ran as a specialist pipeline

    @property
    def tool_call_count(self) -> int:
        return sum(
            1
            for t in self.transcript
            if t.get("kind") == "tool_call" and t.get("name") != SUBMIT_TOOL_NAME and not t.get("terminal")
        )

    def stage_summary(self, name: str) -> dict[str, Any]:
        return {
            "stage": name,
            "models": list(self.models_used),
            "steps": self.steps,
            "tool_calls": self.tool_call_count,
            "forced": self.forced,
            "submitted": self.review is not None,
            "seconds": round(self.seconds, 1),
            "usage": self.usage.as_dict(),
        }


async def run_agent(
    llm: OpenRouter,
    bridge: ToolBridge | None,
    system: str,
    opening: str,
    terminal: TerminalTool,
    *,
    max_steps: int = 40,
    max_context_tokens: int = 160_000,
    max_rejections: int = 2,
    max_seconds: float = 0,
    extra_tools: list[LocalTool] | None = None,
    log=print,
    label: str = "",
    quiet: bool = False,
    on_first_answer: Callable[[], None] | None = None,
) -> AgentResult:
    """Model proposes tool calls, the bridge and local tools answer, until the terminal tool is accepted.

    ``on_first_answer`` is called once, when the first model request has been answered or has failed:
    from then on the provider holds this run's opening in its prompt cache.

    ``max_seconds`` (0 = none) is a wall-clock budget: once exceeded, the next model call is forced to submit.
    While a model call is outstanding a heartbeat line is logged every ``HEARTBEAT_SECONDS``.
    ``quiet`` drops the per-step lines (model calls, tool calls, heartbeats) and logs one line when the
    agent finishes: what it submitted, or why it did not, with the step count and time."""
    started = time.monotonic()
    tag = f"[{label}] " if label else ""
    detail = (lambda msg: None) if quiet else log  # per-step chatter; errors still reach ``log`` via done()
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": opening}]
    local = {t.name: t for t in extra_tools or []}
    bridge_tools = list(bridge.tools) if bridge is not None else []
    bridge_names = bridge.tool_names if bridge is not None else set()
    tools = [*bridge_tools, *(t.schema for t in local.values()), terminal.schema]
    usage = Usage()
    models: list[str] = []
    transcript: list[dict[str, Any]] = []
    forced = False
    nudges = 0
    rejections = 0
    cut_off_submissions = 0  # calls of the terminal tool that the output limit cut off
    last_prompt_tokens = 0
    errors: list[str] = []
    tool_results: list[dict[str, Any]] = []
    compactions = 0
    forced_misses = 0
    cut_offs = 0  # answers in a row that ended at the output limit with nothing in them
    forced_cut_offs = 0  # ... of them while the model was asked to submit
    rung = -1  # how far reasoning is turned down, an index into LESS_REASONING; -1 is the model's default
    refused: set[int] = set()  # rungs a route rejected: not every model takes every setting

    def done(output: Any, step: int) -> AgentResult:
        seconds = time.monotonic() - started
        if quiet:
            what = terminal.describe(output) if output is not None else f"did not submit: {'; '.join(errors)}"
            log(f"{tag}{what} ({step} step(s), {seconds:.0f}s{', forced' if forced else ''})")
        return AgentResult(output, step, usage, models, forced, errors, transcript, seconds, tool_results)

    async def lookup(step: int, name: str, args: dict[str, Any]) -> str:
        """Answer one non-terminal tool call: a local tool or a bridge lookup, recorded in the transcript."""
        if name in local:
            t0 = time.monotonic()
            try:
                content = await local[name].handler(args)
            except Exception as exc:  # noqa: BLE001 - report to the model, keep reviewing
                content = f"ERROR: {type(exc).__name__}: {exc}"
            took = time.monotonic() - t0
            detail(f"{tag}[step {step}] {name}({_short(args)}) -> {len(content)} chars in {took:.1f}s")
            transcript.append(
                {
                    "kind": "tool_call",
                    "step": step,
                    "name": name,
                    "arguments": args,
                    "chars": len(content),
                    "seconds": took,
                }
            )
            tool_results.append({"name": name, "arguments": args, "content": content})
            return content
        assert bridge is not None
        label_call = f"{tag}[step {step}] still waiting for tool {name}({_short(args)})"
        content, rec = await _with_heartbeat(
            bridge.call_with_record(name, args),
            lambda waited, msg=label_call: detail(f"{msg} ({waited:.0f}s)"),
        )
        detail(
            f"{tag}[step {step}] {name}({_short(args)}) -> {rec.result_chars} chars"
            f"{' ERROR' if rec.is_error else ''} in {rec.seconds:.1f}s"
        )
        transcript.append(
            {
                "kind": "tool_call",
                "step": step,
                "name": name,
                "arguments": args,
                "chars": rec.result_chars,
                "error": rec.is_error,
                "seconds": rec.seconds,
            }
        )
        if not rec.is_error:
            tool_results.append({"name": name, "arguments": args, "content": content})
        return content

    # Once the budget is out the model is forced to submit; GRACE_STEPS more turns allow a rejected or
    # malformed forced submission to be retried instead of losing the whole run's work.
    for step in range(1, max_steps + GRACE_STEPS + 1):
        elapsed = time.monotonic() - started
        out_of_time = max_seconds > 0 and elapsed > max_seconds
        budget_out = step >= max_steps or last_prompt_tokens > max_context_tokens or out_of_time
        tool_choice: str | dict[str, Any] = "auto"
        if budget_out and not forced:
            forced = True
            why = f"time budget of {max_seconds:.0f}s" if out_of_time else "step or context budget"
            detail(f"{tag}[step {step}] {why} exhausted; forcing {terminal.name}")
            messages.append({"role": "user", "content": budget_exhausted(terminal.name)})
        if forced:
            # Some providers answer a named tool_choice with an empty message; after two such misses fall
            # back to "required" (any tool), which the loop then steers to the terminal tool by rejection.
            tool_choice = {"type": "function", "function": {"name": terminal.name}} if forced_misses < 2 else "required"

        # "required" means any tool, and the only one wanted by then is the terminal one: offer no other.
        offered = [terminal.schema] if tool_choice == "required" else tools
        approx_tokens = sum(len(str(m.get("content") or "")) for m in messages) // 4
        detail(f"{tag}[step {step}] waiting for {llm.models[0]} ({len(messages)} messages, ~{approx_tokens:,} tokens)")
        t_call = time.monotonic()
        try:
            while True:
                reasoning = LESS_REASONING[rung] if rung >= 0 else None
                try:
                    result = await _chat_with_heartbeat(
                        llm, messages, offered, tool_choice, detail, tag, step, reasoning
                    )
                    break
                except OpenRouterError as exc:
                    if reasoning is None or not _refuses_reasoning(exc):
                        raise
                    # Turning reasoning down is a remedy, not a requirement. A route that rejects a setting
                    # (one model cannot have reasoning switched off, another may not take the field at all)
                    # gets the request again with the setting below it, and is not asked for that one again.
                    refused.add(rung)
                    rung = max((r for r in range(rung) if r not in refused), default=-1)
                    detail(
                        f"{tag}[step {step}] reasoning {reasoning} refused ({str(exc)[:120]}); trying the next lower"
                    )
        except ContextLengthError as exc:
            compactions += 1
            # First elide older tool results and retry; then everything and force submission; then give up.
            elided = compact_history(messages, keep_recent=4) if compactions == 1 else 0
            if elided:
                transcript.append({"kind": "compaction", "step": step, "elided": elided, "error": str(exc)[:300]})
                detail(f"{tag}[step {step}] context window exceeded; elided {elided} older tool result(s) and retrying")
                continue
            elided = compact_history(messages, keep_recent=0)
            if elided:
                transcript.append({"kind": "compaction", "step": step, "elided": elided, "error": str(exc)[:300]})
                forced = True
                detail(f"{tag}[step {step}] context still too large; elided all tool results, forcing submission")
                continue
            errors.append(f"context window exceeded and nothing left to compact: {str(exc)[:300]}")
            detail(f"{tag}[step {step}] {errors[-1]}")
            return done(None, step)
        except OpenRouterError as exc:
            errors.append(f"model request failed: {str(exc)[:500]}")
            detail(f"{tag}[step {step}] {errors[-1]}")
            return done(None, step)
        finally:
            if on_first_answer is not None:
                on_first_answer()
                on_first_answer = None
        usage.add(result.usage)
        last_prompt_tokens = int((result.usage or {}).get("prompt_tokens") or 0)
        if result.model and result.model not in models:
            models.append(result.model)
        detail(
            f"{tag}[step {step}] {result.model or llm.models[0]} answered in {time.monotonic() - t_call:.1f}s "
            f"({len(result.tool_calls)} tool call(s), {last_prompt_tokens:,} prompt tokens)"
        )
        if _cut_off(result, llm):
            # The output limit covers reasoning and answer together, and nothing but reasoning fit. The turn
            # is not kept: its reasoning, sent back, makes every later request that much longer and leads
            # to the same end. The request is made again from the history as it was, with reasoning turned
            # down from here on (changing it once costs the cached prefix once; changing it back would cost
            # it again). How far depends on what is asked: see LESS_REASONING.
            cut_offs += 1
            transcript.append(
                {
                    "kind": "assistant",
                    "step": step,
                    "model": result.model,
                    "content": "",
                    "tool_calls": [],
                    "usage": result.usage,
                    "cut_off": True,
                }
            )
            detail(f"{tag}[step {step}] the answer ended at the output limit with nothing in it ({cut_offs} in a row)")
            if forced:
                # As for any empty answer to a forced choice: after two, any tool will do. A run that is
                # asked to submit this many times and never gets an answer through has nothing to give.
                forced_misses += 1
                forced_cut_offs += 1
                if forced_cut_offs >= MAX_FORCED_CUT_OFFS:
                    errors.append(
                        f"{cut_offs} answers in a row ended at the output limit with nothing in them, "
                        f"{forced_cut_offs} of them when asked to submit"
                    )
                    return done(None, step)
            elif cut_offs >= MAX_CUT_OFFS:
                # Investigating further is not getting anywhere; what the run has so far is still worth asking for.
                forced = True
                detail(f"{tag}[step {step}] forcing {terminal.name}")
                last = messages[-1]
                if last.get("role") == "user" and isinstance(last.get("content"), str):
                    # Nothing was kept of the cut-off turns, so the history may end in the opening message:
                    # the instruction joins it rather than follow it as a second user message in a row.
                    messages[-1] = {**last, "content": last["content"] + "\n\n" + cut_off(terminal.name)}
                else:
                    messages.append({"role": "user", "content": cut_off(terminal.name)})
            # While the investigation goes on, low effort is as far as it goes, however often an answer is
            # cut off: a model with reasoning switched off is not the reviewer that was asked for. Asked to
            # submit, the thinking is done and the turns are few: all the way down.
            last = len(LESS_REASONING) - 1 if forced else 0
            lower = [r for r in range(rung + 1, last + 1) if r not in refused]
            if lower:
                rung = lower[-1]
            continue
        cut_offs = forced_cut_offs = 0
        messages.append(assistant_message_for_history(result.message))
        transcript.append(
            {
                "kind": "assistant",
                "step": step,
                "model": result.model,
                "content": result.content[:4000],
                "tool_calls": [tc.get("function", {}).get("name") for tc in result.tool_calls],
                "usage": result.usage,
            }
        )

        if not result.tool_calls:
            nudges += 1
            if forced:
                forced_misses += 1
            despite = " despite forced tool_choice" if forced else ""
            detail(f"{tag}[step {step}] no tool call{despite}; nudging ({nudges})")
            if nudges > 2:
                forced = True
            messages.append(
                {"role": "user", "content": nudge(terminal.name) if not forced else budget_exhausted(terminal.name)}
            )
            continue

        # Parse every call first so a submission is judged before any lookups run; then run the lookups
        # concurrently (models often ask for several files at once) and answer them in the model's order.
        parsed: list[tuple[dict[str, Any], str, dict[str, Any] | None, str, bool]] = []
        # Only the last call of an answer can be the one that its output limit cut off.
        at_limit = _at_output_limit(result, llm)
        for n, tc in enumerate(result.tool_calls):
            cut = at_limit and n == len(result.tool_calls) - 1
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            raw_args = fn.get("arguments") or "{}"
            try:
                args = tool_arguments(raw_args)
                content = ""
            except json.JSONDecodeError as exc:
                args = None
                if not cut:
                    content = f"ERROR: arguments were not valid JSON ({exc}). Call the tool again with a JSON object."
                elif forced and name != terminal.name:
                    # A lookup is not run once the run is asked to submit, so this one is not asked for again.
                    content = f"Not run: the budget is exhausted. {budget_exhausted(terminal.name)}"
                else:
                    # Not a model that cannot write JSON: the answer ended in the middle of it. Told that its
                    # JSON was invalid, a model sends the same arguments again wrapped up as text.
                    content = cut_off_call(name)
                    cut_off_submissions += int(name == terminal.name)
                detail(f"{tag}[step {step}] {name}: unreadable arguments ({exc})")
                transcript.append(
                    {
                        "kind": "tool_call_unreadable",
                        "step": step,
                        "name": name,
                        "error": str(exc),
                        "cut_off": cut,
                        "arguments": _clipped(raw_args),
                    }
                )
            parsed.append((tc, name, args, content, cut))

        answers: dict[int, str] = {}
        for i, (tc, name, args, content, cut) in enumerate(parsed):
            if args is None:
                answers[i] = content
            elif name == terminal.name:
                output, errs = terminal.parse(args)
                entry = {
                    "kind": "tool_call",
                    "step": step,
                    "name": name,
                    "terminal": True,
                    "ok": not errs,
                    "errors": errs,
                }
                if errs:
                    # What was sent, as it was sent: the errors alone do not say what the model got wrong.
                    entry["arguments"] = _clipped((tc.get("function") or {}).get("arguments") or "")
                    entry["cut_off"] = cut
                transcript.append(entry)
                if output is not None and not errs:
                    detail(f"{tag}[step {step}] {terminal.describe(output)}")
                    return done(output, step)
                if cut:
                    # A submission cut off at the output limit can arrive as valid JSON: as an empty object,
                    # or with the fields it got to and the rest missing. What it lacks was never written, so
                    # it is answered as the cut-off call it is, and is not one of the rejections a run may
                    # have. The step budget still ends a run that keeps doing it.
                    detail(f"{tag}[step {step}] submission cut off at the output limit: {errs}")
                    cut_off_submissions += 1
                    answers[i] = cut_off_call(name)
                    continue
                rejections += 1
                detail(f"{tag}[step {step}] submission rejected: {errs}")
                if rejections > max_rejections:
                    errors.extend(errs)
                    return done(None, step)
                answers[i] = rejection_message(errs, terminal.name)
            elif name not in local and not (bridge is not None and name in bridge_names):
                # What was offered in this request: under "required" that is the terminal tool alone.
                available = ", ".join(sorted(t["function"]["name"] for t in offered))
                answers[i] = f"ERROR: unknown tool {name!r}. Available: {available}."
            elif forced:
                # The model is asked to submit. A lookup it asks for instead (the named choice is not binding
                # on every provider, and a model may call a tool it is not offered) is not run, or the last
                # turns go to reading instead of to the submission.
                answers[i] = f"Not run: the budget is exhausted. {budget_exhausted(terminal.name)}"
                transcript.append({"kind": "tool_refused", "step": step, "name": name, "arguments": args})

        pending = [i for i in range(len(parsed)) if i not in answers]
        results = await asyncio.gather(*(lookup(step, parsed[i][1], parsed[i][2] or {}) for i in pending))
        answers.update(zip(pending, results, strict=True))
        for i, (tc, name, _args, _content, _cut) in enumerate(parsed):
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "name": name, "content": answers[i]})

    budget = f"within {max_steps} steps (+{GRACE_STEPS} forced)"
    if cut_off_submissions:
        # It did call it: say what became of the calls, or the error sends a reader looking for a model
        # that would not submit.
        errors.append(
            f"{cut_off_submissions} call(s) of {terminal.name} were cut off at the output limit and "
            f"none got through {budget}"
        )
    else:
        errors.append(f"model did not call {terminal.name} {budget}")
    return done(None, max_steps + GRACE_STEPS)


MAX_CUT_OFFS = 2  # answers in a row cut off with nothing in them before the model is asked to submit
# OpenRouter's `reasoning` request field, in the order it is turned down after an answer is cut off. Low
# effort is documented as about a fifth of the output limit, but it is a request: one draft model reasons
# its 8000 tokens away at low effort as at default, and gets through with reasoning switched off; another
# gets through at low and refuses "none" ("Reasoning is mandatory for this endpoint"). So both, in order.
LESS_REASONING: tuple[dict[str, Any], ...] = ({"effort": "low"}, {"effort": "none"})


REFUSAL_STATUSES = (400, 404, 422)  # the request as made is not accepted


def _refuses_reasoning(exc: Exception) -> bool:
    """A failed request that was rejected for its reasoning setting: the route does not take it. That is a
    client error naming the field or its effort ("Reasoning is mandatory for this endpoint", "unsupported
    effort value"). A rejected key, an outage and a failure after the client's own attempts say nothing
    about the setting, whatever words they use."""
    if getattr(exc, "status", None) not in REFUSAL_STATUSES:
        return False
    text = str(exc).lower()
    return "reasoning" in text or "effort" in text


def _at_output_limit(result, llm) -> bool:
    """An answer that ended at its output limit. The reason it gives for ending does not always say so: a
    tool call cut off in the middle of its arguments has been seen to end with "tool_calls", so having used
    the whole limit says it too, unless the answer names another cause. An answer that was complete at
    exactly the limit is taken for one that was cut off; that matters only if its tool call is unusable."""
    if result.finish_reason == "length":
        return True
    if result.finish_reason in ("content_filter", "error"):
        return False
    limit = int(getattr(llm, "max_tokens", 0) or 0)
    return limit > 0 and int((result.usage or {}).get("completion_tokens") or 0) >= limit


KEPT_ARGUMENT_CHARS = 4000  # of arguments that were rejected or could not be read, in the transcript


def _clipped(raw: Any) -> str:
    text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, default=str)
    if len(text) <= KEPT_ARGUMENT_CHARS:
        return text
    return f"{text[:KEPT_ARGUMENT_CHARS]}... [{len(text) - KEPT_ARGUMENT_CHARS} more characters]"


def _cut_off(result, llm) -> bool:
    """An answer that ended at the output limit with neither text nor a tool call in it. The limit covers
    a model's reasoning too, so this is what reasoning that used all of it looks like. Another reason to
    end (a content filter, an error, a stop of the model's own) is not this, whatever the token count;
    without a reason given, having used the whole limit says it."""
    if result.tool_calls or result.content.strip():
        return False
    if result.finish_reason:
        return result.finish_reason == "length"
    limit = int(getattr(llm, "max_tokens", 0) or 0)
    return limit > 0 and int((result.usage or {}).get("completion_tokens") or 0) >= limit


HEARTBEAT_SECONDS = 30.0
GRACE_STEPS = 3  # extra turns after the step budget, all forced to the terminal tool
MAX_FORCED_CUT_OFFS = GRACE_STEPS + 1  # cut-off answers in a row when asked to submit: as many tries as a run gets


async def _with_heartbeat(awaitable, on_wait):
    """Await something, calling ``on_wait(seconds)`` every HEARTBEAT_SECONDS while it is outstanding."""
    task = asyncio.ensure_future(awaitable)
    waited = 0.0
    while True:
        done, _ = await asyncio.wait({task}, timeout=HEARTBEAT_SECONDS)
        if done:
            return task.result()
        waited += HEARTBEAT_SECONDS
        on_wait(waited)


async def _chat_with_heartbeat(llm, messages, tools, tool_choice, log, tag: str, step: int, reasoning=None):
    """Run the blocking model call in a thread; say so periodically while it is outstanding."""
    return await _with_heartbeat(
        asyncio.to_thread(llm.chat, messages, tools, tool_choice, reasoning),
        lambda waited: log(
            f"{tag}[step {step}] still waiting for {llm.models[0]} ({waited:.0f}s; request timeout {llm.timeout:.0f}s)"
        ),
    )


ELIDED = "[tool result elided to fit the context window; call the tool again if you still need it]"
ELIDED_FOR_GOOD = "[tool result elided to fit the context window]"  # when no tool will be run again


def compact_history(messages: list[dict[str, Any]], keep_recent: int = 4) -> int:
    """Replace the content of older tool results with a short note; returns how many were elided. With
    ``keep_recent=0`` everything goes and the run is asked to submit next, so the note no longer invites
    the model to call the tool again: a lookup would not be run."""
    notes = (ELIDED, ELIDED_FOR_GOOD)
    tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool" and m.get("content") not in notes]
    victims = tool_idx[: max(0, len(tool_idx) - keep_recent)] if keep_recent else tool_idx
    for i in victims:
        messages[i]["content"] = ELIDED if keep_recent else ELIDED_FOR_GOOD
    if not keep_recent:
        for m in messages:
            if m.get("role") == "tool" and m.get("content") == ELIDED:
                m["content"] = ELIDED_FOR_GOOD
    return len(victims)


def review_terminal(changed_lines: dict[str, set[int]]) -> TerminalTool:
    return TerminalTool(
        SUBMIT_REVIEW_TOOL,
        lambda args: parse_review(args, changed_lines),
        lambda r: f"review submitted: {r.verdict}, {len(r.findings)} finding(s)",
    )


async def run_review(
    llm: OpenRouter,
    bridge: ToolBridge,
    system: str,
    opening: str,
    changed_lines: dict[str, set[int]],
    *,
    max_steps: int = 40,
    max_context_tokens: int = 160_000,
    max_rejections: int = 2,
    max_seconds: float = 0,
    extra_tools: list[LocalTool] | None = None,
    log=print,
    quiet: bool = False,
) -> RunResult:
    """The single-pass review: investigate with the tools, then submit_review."""
    res = await run_agent(
        llm,
        bridge,
        system,
        opening,
        review_terminal(changed_lines),
        max_steps=max_steps,
        max_context_tokens=max_context_tokens,
        max_rejections=max_rejections,
        max_seconds=max_seconds,
        extra_tools=extra_tools,
        log=log,
        quiet=quiet,
    )
    return RunResult(
        res.output,
        res.steps,
        res.usage,
        res.models_used,
        res.forced,
        res.errors,
        res.transcript,
        res.seconds,
        res.tool_results,
    )


def _short(args: dict[str, Any], limit: int = 80) -> str:
    s = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return s if len(s) <= limit else s[: limit - 3] + "..."
