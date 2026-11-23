# SPDX-FileCopyrightText: Copyright (c) 2026 Silicon Laboratories Inc.
# SPDX-License-Identifier: Apache-2.0
"""What the posted assessment may say about the kind of defect a change fixes.

A fix for a security problem can be worded neutrally on purpose: a project's disclosure rules may ask that a
public commit and pull request do not reveal the sensitive nature of what is being fixed. A review that then
sums the change up as "prevents a use-after-free" says it for the author, in public and for good. So the
summary and the assessments name a class of memory-safety or security defect only when the pull request's own
text does, or when one of the review's findings does. That is checked here, on the text, and does not rest on
a model following an instruction.

Findings are not touched: a finding is about a problem the review found, and says what it has to say. A
summary that repeats a finding's class says nothing the finding does not."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

# One pattern per class, loose enough to meet the ways a class gets named. A class counts as stated by the
# pull request when any wording of the same class occurs in its title, description or commit messages.
_CLASSES = {
    "use after free": r"use[- ]after[- ]free|\buaf\b|double[- ]free|dangling (?:pointer|reference)"
    r"|freed (?:memory|buffer|object)",
    # One class: what one text calls a buffer overflow another calls an out-of-bounds write.
    "out of bounds": r"overflow|overrun|over-?read|underflow|\boverwrit\w+ (?:past|beyond)"
    r"|out[- ]of[- ]bounds?|\boob\b|(?:past|beyond) the (?:end|bounds?) of",
    "memory corruption": r"memory[- ](?:corruption|safety)|corrupt\w* (?:memory|the (?:heap|stack))",
    "information leak": r"(?:information|info|data)[- ](?:leak|disclosure)"
    r"|leak\w* (?:kernel|stack|heap|uninitiali[sz]ed) (?:memory|data)",
    "privilege": r"privilege (?:escalation|elevation)|escalat\w+ (?:of )?privilege",
    "code execution": r"code execution|arbitrary (?:read|write|code)",
    "denial of service": r"denial[- ]of[- ]service|(?-i:\bDoS\b)",
    "vulnerability": r"vulnerabilit\w+|exploit\w*|\bcve-\d"
    r"|security (?:issue|flaw|bug|fix|hole|impact|risk|implication|problem|defect)",
}
_PATTERNS = {name: re.compile(pattern, re.I) for name, pattern in _CLASSES.items()}
_SENTENCES = re.compile(r"(?<=[.!?])\s+")
ASSESSMENTS = ("test_assessment", "commit_message_assessment", "discussion_assessment")


def named_classes(text: str) -> set[str]:
    """The classes of memory-safety or security defect that ``text`` names."""
    return {name for name, pattern in _PATTERNS.items() if pattern.search(text or "")}


def stated_text(title: str, body: str, commits: Iterable[Any] = ()) -> str:
    """What the author wrote about the change: title, description and commit messages."""
    parts = [title or "", body or ""]
    parts += [f"{getattr(c, 'subject', '')}\n{getattr(c, 'body', '')}" for c in commits]
    return "\n".join(parts)


def keep_to_what_is_stated(text: str, stated: str) -> tuple[str, int]:
    """``text`` without the sentences that name a class ``stated`` does not, and how many those were."""
    allowed = named_classes(stated)
    kept = []
    dropped = 0
    for sentence in _SENTENCES.split((text or "").strip()):
        if named_classes(sentence) - allowed:
            dropped += 1
        elif sentence:
            kept.append(sentence)
    return " ".join(kept), dropped


FINDING_TEXT = ("title", "problem", "trigger", "impact", "recommendation")


def keep_assessment_neutral(review: Any, stated: str, log: Callable[[str], None] = lambda m: None) -> int:
    """Apply ``keep_to_what_is_stated`` to the summary and the assessments of ``review``, with the review's own
    findings counted as stated; returns the number of sentences withheld. The log line does not repeat them: a
    workflow log is as public as the comment."""
    found = "\n".join(str(getattr(f, name, "") or "") for f in review.findings for name in FINDING_TEXT)
    stated = f"{stated}\n{found}"
    summary, dropped = keep_to_what_is_stated(review.summary, stated)
    if dropped:
        review.summary = summary or (
            "Automated review completed; see findings." if review.findings else "No issues found."
        )
    for field in ASSESSMENTS:
        text, n = keep_to_what_is_stated(getattr(review, field, "") or "", stated)
        if n:
            setattr(review, field, text)
            dropped += n
    if dropped:
        log(
            f"{dropped} sentence(s) of the summary or the assessments withheld: they name a kind of defect that "
            "neither the pull request's title, description and commit messages nor a finding name"
        )
    return dropped
