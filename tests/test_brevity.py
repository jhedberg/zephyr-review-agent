# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""Posted comments are short: one paragraph, one fix line, code when useful, no evidence or internals."""

from __future__ import annotations

from zephyr_reviewer.review.render import finding_text, inline_comments
from zephyr_reviewer.review.schema import Finding, Review


def _f(**kw) -> Finding:
    base = dict(
        file="a.c",
        line=3,
        severity="medium",
        category="bug",
        title="T",
        problem="The lock is leaked on error",
        inline=True,
    )
    base.update(kw)
    return Finding(**base)


def test_finding_text_folds_trigger_and_impact_without_repeating():
    assert finding_text(_f()) == "The lock is leaked on error."
    assert finding_text(_f(trigger="a transfer fails", impact="the bus wedges")) == (
        "The lock is leaked on error. Happens when a transfer fails. The bus wedges."
    )
    assert finding_text(_f(trigger="When CONFIG_NET_VLAN is enabled")) == (
        "The lock is leaked on error. When CONFIG_NET_VLAN is enabled."
    )
    # Clauses the problem already contains are not repeated.
    assert finding_text(
        _f(
            problem="The lock is leaked when a transfer fails, so the bus wedges",
            trigger="a transfer fails",
            impact="the bus wedges",
        )
    ) == ("The lock is leaked when a transfer fails, so the bus wedges.")


def test_a_trigger_keeps_the_spelling_of_an_acronym_or_identifier():
    assert finding_text(_f(trigger="Compiling with the check enabled")).endswith(
        "Happens when compiling with the check enabled."
    )
    assert finding_text(_f(trigger="A transfer fails")).endswith("Happens when a transfer fails.")
    assert finding_text(_f(trigger="Non-blocking mode is enabled")).endswith(
        "Happens when non-blocking mode is enabled."
    )
    for t in ("MISRA C:2012 Rule 8.9 is checked", "CONFIG_SMP is enabled", "K_NO_WAIT is passed", "`foo()` fails"):
        assert finding_text(_f(trigger=t)).endswith(f"Happens when {t}.")


def test_inline_comment_shape():
    f = _f(
        recommendation="Give the semaphore before returning",
        suggested_code="\tk_sem_give(&l);",
        evidence="read_file a.c",
        notes=["internal"],
    )
    body = inline_comments(Review("s", "comment", "high", [f]), "t")[0]["body"].splitlines()
    assert body[0].startswith("<!-- t:finding:") and body[1] == "**T** · 🟡 medium"
    assert body[3] == "The lock is leaked on error."
    assert body[5] == "**Fix:** Give the semaphore before returning."
    assert body[7:10] == ["```suggestion", "\tk_sem_give(&l);", "```"]
    assert body[-1] == "<sub>AI review, advisory; verify before acting on it.</sub>"
    joined = "\n".join(body)
    assert "read_file" not in joined and "internal" not in joined and "Problem" not in joined


def test_code_with_an_elided_part_is_shown_not_offered_as_a_suggestion():
    def body(code: str, file: str = "a.c") -> str:
        return inline_comments(Review("s", "comment", "high", [_f(suggested_code=code, file=file)]), "t")[0]["body"]

    two_sites = "expected = (uint16_t)((buf != NULL) ? cmd(buf)->opcode : 0U);\n...\nop->err = (uint8_t)err;"
    for code in (two_sites, "\tfoo();\n\t/* ... */\n\tbar();", "foo();\n// ...\nbar();", "foo();\n  …\nbar();"):
        assert "```suggestion" not in body(code) and f"```c\n{code}\n```" in body(code)
    assert "```python\nfoo()\n# ...\nbar()\n```" in body("foo()\n# ...\nbar()", "a.py")
    # Dots inside a line are code or a comment, not an elision.
    for code in ("\tva_start(ap, fmt); /* fmt, ... */", "#define LOG(...) printk(__VA_ARGS__)"):
        assert f"```suggestion\n{code}\n```" in body(code)
    # A line of "..." is a statement in Python and ends a document in YAML.
    for code, file in (("def probe(dev):\n    ...", "a.py"), ('doc = """\n...\n"""', "a.py"), ("a: 1\n...", "a.yaml")):
        assert f"```suggestion\n{code}\n```" in body(code, file)


def test_every_stage_is_told_what_suggested_code_replaces():
    from zephyr_reviewer.review.pipeline import _FINDING_PROPS
    from zephyr_reviewer.review.schema import SUBMIT_REVIEW_TOOL, SUGGESTED_CODE

    single = SUBMIT_REVIEW_TOOL["function"]["parameters"]["properties"]["findings"]["items"]["properties"]
    assert single["suggested_code"]["description"] == _FINDING_PROPS["suggested_code"]["description"] == SUGGESTED_CODE
    assert "nothing else" in SUGGESTED_CODE and "no elided parts" in SUGGESTED_CODE
