# SPDX-FileCopyrightText: Copyright (c) 2026 Silicon Laboratories Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from tests.test_agent import GOOD_REVIEW
from zephyr_reviewer.intel.repo import Commit
from zephyr_reviewer.review.disclosure import (
    keep_assessment_neutral,
    keep_to_what_is_stated,
    named_classes,
    stated_text,
)
from zephyr_reviewer.review.schema import parse_review

NEUTRAL = stated_text(
    "drivers: foo: stop the transfer from the work item",
    "Moves stopping a transfer to the driver's work item.",
    [Commit("a" * 40, "A", "drivers: foo: stop the transfer from the work item", "The interrupt handler...")],
)


def test_the_classes_a_text_names():
    assert named_classes("Closes a use-after-free in foo_stop().") == {"use after free"}
    assert named_classes("The length check keeps foo_read() from a buffer overflow.") == {"out of bounds"}
    assert named_classes("Guards table accesses against writes past the end of the table.") == {"out of bounds"}
    assert named_classes("This is a security fix for CVE-2026-1234; exploitable remotely.") == {"vulnerability"}
    assert named_classes("a DoS by a peer") == {"denial of service"} and named_classes("dos line endings") == set()
    assert named_classes("leaks uninitialized memory to the peer") == {"information leak"}
    # Ordinary review vocabulary is none of them.
    ordinary = ("The lock is leaked on the error path.", "A race between the two work items.", "", "Returns -EINVAL.")
    ordinary += ("A memory leak when the allocation fails.", "The top area is left vulnerable to modification.")
    for text in ordinary:
        assert named_classes(text) == set(), text


def test_a_sentence_naming_a_class_the_author_did_not_is_withheld():
    said = "The change is correct and closes a use-after-free in foo_stop(). Tests are missing."
    assert keep_to_what_is_stated(said, NEUTRAL) == ("Tests are missing.", 1)
    # The author's own words may be repeated, in whatever form the class is named.
    stated = stated_text("drivers: foo: fix a UAF in foo_stop()", "", [])
    assert keep_to_what_is_stated(said, stated) == (said, 0)
    # Another class than the one the author named is still not for the review to add.
    both = "Prevents the use-after-free. Also closes an information leak."
    assert keep_to_what_is_stated(both, stated) == ("Prevents the use-after-free.", 1)
    assert keep_to_what_is_stated("", NEUTRAL) == ("", 0)


def test_the_summary_and_the_assessments_are_kept_to_what_the_pull_request_states():
    def review(**over):
        r, errs = parse_review({**GOOD_REVIEW, **over}, {"drivers/i2c/i2c_foo.c": {17, 18, 19}})
        assert r is not None, errs
        return r

    logged: list[str] = []
    r = review(
        summary="Fixes a heap overflow in the receive path.",
        commit_message_assessment="The message hides that this is a security fix. The subject is too long.",
        test_assessment="No test covers the error path.",
    )
    title = r.findings[0].title
    assert keep_assessment_neutral(r, NEUTRAL, logged.append) == 2
    # Nothing is left of the summary: the stock one for a review with findings. The findings are as they were.
    assert r.summary == "Automated review completed; see findings." and r.findings[0].title == title
    assert r.commit_message_assessment == "The subject is too long."
    assert r.test_assessment == "No test covers the error path."
    # The log is public too: it says that something was withheld, not what.
    assert len(logged) == 1 and logged[0].startswith("2 sentence(s) of the summary or the assessments withheld")
    assert not named_classes(logged[0])

    # What a finding is about may be said in the summary: the finding says it anyway.
    finding = {**GOOD_REVIEW["findings"][0], "title": "Out-of-bounds write past the end of buf", "problem": "p"}
    r = review(summary="The new copy can overflow buf. It also fixes a use-after-free.", findings=[finding])
    assert keep_assessment_neutral(r, NEUTRAL) == 1 and r.summary == "The new copy can overflow buf."
    r = review(summary="Prevents an out-of-bounds write.", findings=[])
    assert keep_assessment_neutral(r, NEUTRAL) == 1 and r.summary == "No issues found."
    r = review()
    assert keep_assessment_neutral(r, NEUTRAL, logged.append) == 0 and r.summary == GOOD_REVIEW["summary"]
    assert len(logged) == 1
