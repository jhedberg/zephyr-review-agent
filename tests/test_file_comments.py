# SPDX-FileCopyrightText: Copyright (c) 2026 Silicon Laboratories Inc.
# SPDX-License-Identifier: Apache-2.0

"""A finding without a line it can sit on, in a file the pull request changes, is posted on that file."""

from __future__ import annotations

import copy
import json
import re

import httpx

from tests.test_agent import GOOD_REVIEW
from zephyr_reviewer.review import publish as pub
from zephyr_reviewer.review.agent import RunResult
from zephyr_reviewer.review.config import ReviewConfig
from zephyr_reviewer.review.discussion import Discussion
from zephyr_reviewer.review.followup import as_claims, dismissed_findings, open_findings, parse_posted
from zephyr_reviewer.review.github_api import GitHub, PullRequest
from zephyr_reviewer.review.openrouter import Usage
from zephyr_reviewer.review.render import finding_key, inline_comments, inline_marker, preview_inline
from zephyr_reviewer.review.schema import Finding, Review, parse_review

FILE = "drivers/i2c/i2c_foo.c"
CHANGED = {FILE: {17, 18, 19}, "drivers/i2c/Kconfig": {20}}
TAG = "zephyr-reviewer"
ON_FILE = "/repos/o/r/pulls/7/comments"


def _run():
    review, _ = parse_review(GOOD_REVIEW, CHANGED)
    pr = PullRequest(7, "t", "", "a", "main", "b" * 40, "pr", "h" * 40, "o/r", False)
    return RunResult(review, 2, Usage(), ["m"], False), pr


def _render(exclude_inline: bool) -> str:
    return "summary-without" if exclude_inline else "summary-with-all"


def _gh(
    seen: list,
    existing: list | None = None,
    reject: tuple[str, ...] = (),
    threads: list | None = None,
    reactions: dict[int, list] | None = None,
) -> GitHub:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        path = request.url.path
        seen.append((request.method, path, body))
        m = re.fullmatch(r"/repos/o/r/pulls/comments/(\d+)/reactions", path)
        if request.method == "GET" and m:
            return httpx.Response(200, json=(reactions or {}).get(int(m.group(1)), []))
        if path == "/graphql":
            conn = {"pageInfo": {"hasNextPage": False, "endCursor": None}, "nodes": threads or []}
            return httpx.Response(200, json={"data": {"repository": {"pullRequest": {"reviewThreads": conn}}}})
        if request.method == "GET":
            return httpx.Response(200, json=(existing or []) if path == ON_FILE else [])
        if any(path.endswith(r) for r in reject):
            return httpx.Response(422, json={"message": "Unprocessable"})
        if path.endswith("/reviews"):
            return httpx.Response(200, json={"html_url": "https://github.com/o/r/pull/7#pullrequestreview-1"})
        if path == ON_FILE:
            return httpx.Response(201, json={"html_url": "https://github.com/o/r/pull/7#discussion_r5"})
        return httpx.Response(201, json={"html_url": "https://github.com/o/r/pull/7#issuecomment-9"})

    return GitHub("tok", "o/r", client=httpx.Client(transport=httpx.MockTransport(handler)))


def _summaries(seen: list) -> list[str]:
    return [b["body"] for m, p, b in seen if m in ("POST", "PATCH") and "/issues/" in p]


def test_a_finding_in_a_changed_file_without_a_changed_line_is_marked_for_its_file():
    review, _ = parse_review(GOOD_REVIEW, CHANGED)
    assert [(f.title, f.inline, f.on_file) for f in review.findings] == [
        ("Early return leaks the bus lock", True, False),
        ("select of I2C_BUS_RECOVERY without a definition", True, False),
        ("Unrelated file cited", False, False),  # the pull request does not change that file
        ("Context line cited", False, True),  # it changes this one, but not line 1
    ]
    raw = copy.deepcopy(GOOD_REVIEW)
    raw["findings"][3]["line"] = 0  # no line at all: about the file as a whole
    review, _ = parse_review(raw, CHANGED)
    assert review.findings[3].on_file and not review.findings[3].inline
    assert Review.from_dict(review.to_dict()).findings[3].on_file  # kept in review.json for `post`


def test_the_comment_on_a_file_names_no_line_and_offers_no_suggestion_block():
    f = Finding(
        file=FILE,
        line=40,
        severity="medium",
        category="bug",
        title="Recovery is never tested",
        problem="no test calls foo_recover()",
        recommendation="add a test that holds SDA low",
        suggested_code="ret = foo_recover(dev);",
        on_file=True,
    )
    (c,) = inline_comments(Review("s", "comment", "high", [f]), TAG)
    assert c == {"path": FILE, "subject_type": "file", "body": c["body"]}  # no line, no side
    assert c["body"].startswith(inline_marker(TAG, finding_key(f)))
    assert "```suggestion" not in c["body"] and "```c\nret = foo_recover(dev);\n```" in c["body"]
    title, text, severity, fix = parse_posted(c["body"])
    assert (title, severity, fix) == ("Recovery is never tested", "medium", "Add a test that holds SDA low.")
    assert text.endswith("This is about line 40, which the pull request does not change.")
    f.line = 0
    (c,) = inline_comments(Review("s", "comment", "high", [f]), TAG)
    assert "This is about line" not in c["body"]
    assert f"--- [1] {FILE} (the file as a whole) ---" in preview_inline([c])


def test_publish_posts_it_on_the_file_after_the_inline_review():
    run, pr = _run()
    seen: list = []
    logs: list[str] = []
    out = pub.publish(ReviewConfig(comment_tag=TAG), _gh(seen), pr, run, _render, log=logs.append)
    posts = [(p, b) for m, p, b in seen if m == "POST"]
    assert [p for p, _ in posts] == ["/repos/o/r/issues/7/comments", "/repos/o/r/pulls/7/reviews", ON_FILE]
    assert _summaries(seen) == ["summary-without"]
    review = posts[1][1]
    assert len(review["comments"]) == 2 and not any("subject_type" in c for c in review["comments"])
    assert review["body"] == "2 inline note(s) from the AI review."
    on_file = posts[2][1]
    assert on_file == {"commit_id": "h" * 40, "path": FILE, "subject_type": "file", "body": on_file["body"]}
    assert "**Context line cited**" in on_file["body"]
    assert out["inline_count"] == 3 and out["posted"] == "true"
    assert any("1 comment(s) on a file as a whole" in m for m in logs)


def test_a_rejected_comment_on_a_file_is_folded_into_the_summary():
    run, pr = _run()
    seen: list = []
    logs: list[str] = []
    out = pub.publish(
        ReviewConfig(comment_tag=TAG), _gh(seen, reject=("/pulls/7/comments",)), pr, run, _render, logs.append
    )
    first, second = _summaries(seen)
    assert first == "summary-without" and second.startswith("summary-without\n\n### Findings that could not be posted")
    folded = second[len(first) :]
    assert f"`{FILE}`" in folded and "**Context line cited**" in folded and "<!--" not in folded
    assert out["inline_count"] == 2 and any("folding the finding into the summary comment" in m for m in logs)


def test_a_rejected_inline_review_leaves_every_finding_to_the_summary():
    run, pr = _run()
    seen: list = []
    out = pub.publish(ReviewConfig(comment_tag=TAG), _gh(seen, reject=("/reviews",)), pr, run, _render, lambda m: None)
    assert _summaries(seen) == ["summary-without", "summary-with-all"]
    assert not any(m == "POST" and p == ON_FILE for m, p, _ in seen) and out["inline_count"] == 0


def test_a_comment_on_a_file_is_not_posted_twice():
    run, pr = _run()
    marker = inline_marker(TAG, finding_key(run.review.findings[3]))
    existing = [{"id": 1, "path": FILE, "subject_type": "file", "body": marker + "\n**Context line cited** · low"}]
    seen: list = []
    out = pub.publish(ReviewConfig(comment_tag=TAG), _gh(seen, existing), pr, run, _render, lambda m: None)
    assert not any(m == "POST" and p == ON_FILE for m, p, _ in seen) and out["inline_count"] == 2


def test_its_thread_is_re_checked_and_can_be_set_aside_like_one_on_a_line():
    run, _ = _run()
    body = next(c for c in inline_comments(run.review, TAG) if c.get("subject_type") == "file")["body"]
    raw = {
        "id": 5,
        "body": body,
        "path": FILE,
        "subject_type": "file",
        "line": 1,  # what GitHub reports for a comment on a file as a whole, with position 1
        "original_line": 1,
        "position": 1,
        "side": "RIGHT",
        "in_reply_to_id": None,
        "user": {"login": "zephyrbot", "type": "Bot"},
        "html_url": "https://github.com/o/r/pull/7#discussion_r5",
    }
    thread = {
        "id": "T5",
        "isResolved": False,
        "isOutdated": False,
        "path": FILE,
        "line": 1,
        "resolvedBy": None,
        "comments": {"nodes": [{"databaseId": 5}]},
    }
    (found,) = open_findings(_gh([], [raw], threads=[thread]), 7, TAG)
    assert (found.path, found.line, found.title, found.outdated) == (FILE, 0, "Context line cited", False)
    (claim,) = as_claims([found], 1, "a" * 40)
    assert (claim.location, claim.extra["thread_id"], claim.extra["comment_id"]) == (FILE, "T5", 5)
    # A thumbs down from a person sets it aside: not re-checked, and known as dismissed.
    down = {5: [{"content": "-1", "user": {"login": "carol", "type": "User"}}]}
    assert open_findings(_gh([], [raw], threads=[thread], reactions=down), 7, TAG) == []
    (gone,) = dismissed_findings(_gh([], [raw], threads=[thread], reactions=down), 7, TAG)
    assert (gone.path, gone.line, gone.title, gone.by) == (FILE, 0, "Context line cited", "carol")
    # The models are not told that it is on line 1.
    shown = Discussion.fetch(_gh([], [raw], threads=[thread]), 7, TAG).inline[0]
    assert (shown.path, shown.line, shown.outdated) == (FILE, None, False)
