"""Deterministic TimeWarp verifiers: string_match, number_match and list_match.

Ported from the `StringMatchEvaluator`, `NumberMatchEvaluator` and
`ListMatchEvaluator` classes in `src/browsergym/timewarp/evaluators.py` of the
TimeWarp repository at commit 4978e690ef2ad33d81a4a0c043da3f80588902a0. Each
function takes the agent's free-text answer and the task's
`eval.reference_answers` object and returns whether the answer passes. A
malformed spec raises ValueError rather than scoring 0, as upstream does.
"""

from collections.abc import Callable, Mapping
from typing import Any

from timewarp.normalization import (
    as_entry_list,
    contains_entry,
    equals_entry,
    extract_numbers,
    find_entry,
    numbers_match,
    scope_text,
    to_decimal,
)


def string_match(answer: str, references: Mapping[str, Any]) -> bool:
    """Match a free-text answer against string references.

    Keys of `references`:
        exact_match: String or list; the normalized answer must equal one of them.
        must_include: Entries that must all appear, matched on word boundaries.
        must_exclude: Entries that must not appear.
        scope: "full" (default) or "first_sentence".

    Entries may hold `" |OR| "` alternatives, and entries written as `^...$`
    are regexes over the normalized text.
    """
    text = scope_text(answer, references.get("scope", "full"))

    exact = as_entry_list(references.get("exact_match"), "exact_match")
    includes = as_entry_list(references.get("must_include"), "must_include")
    excludes = as_entry_list(references.get("must_exclude"), "must_exclude")

    if not (exact or includes or excludes):
        raise ValueError(
            "string_match requires at least one of 'exact_match', "
            "'must_include' or 'must_exclude' in reference_answers"
        )

    if exact and not any(equals_entry(text, entry) for entry in exact):
        return False
    if not all(contains_entry(text, entry) for entry in includes):
        return False
    return not any(contains_entry(text, entry) for entry in excludes)


def number_match(answer: str, references: Mapping[str, Any]) -> bool:
    """Check that every required number appears in the answer, in any format.

    The spec lives at `references["number_match"]`, e.g.
    `{"value": 7000000, "rel_tolerance": 0.1}` or
    `{"values": [5, 1.8], "abs_tolerance": 0.01}`. Comparison is exact unless a
    tolerance is given; `values` entries may carry their own tolerances.
    """
    spec = references.get("number_match")
    if not isinstance(spec, Mapping):
        raise ValueError(
            "number_match requires a 'number_match' object in reference_answers"
        )

    text = scope_text(answer, spec.get("scope", "full"))
    default_rel = spec.get("rel_tolerance")
    default_abs = spec.get("abs_tolerance")

    if "values" in spec:
        requirements = spec["values"]
        if not isinstance(requirements, list) or not requirements:
            raise ValueError("number_match 'values' must be a non-empty list")
    elif "value" in spec:
        requirements = [spec["value"]]
    else:
        raise ValueError("number_match requires a 'value' or 'values' key")

    candidates = extract_numbers(text)
    if not candidates:
        return False

    for requirement in requirements:
        if isinstance(requirement, Mapping):
            expected = to_decimal(requirement["value"])
            rel = requirement.get("rel_tolerance", default_rel)
            abs_ = requirement.get("abs_tolerance", default_abs)
        else:
            expected = to_decimal(requirement)
            rel, abs_ = default_rel, default_abs

        if not any(numbers_match(c, expected, rel, abs_) for c in candidates):
            return False
    return True


def list_match(answer: str, references: Mapping[str, Any]) -> bool:
    """Check that the answer enumerates every item of a reference list.

    The spec lives at `references["list_match"]`, e.g.
    `{"items": [["koalas"], ["kangaroos |OR| kangaroo"]], "ordered": false,
    "forbidden": []}`. Each item is a list of interchangeable spellings (a bare
    string is accepted). With `ordered: true` the items' first occurrences must
    follow the given order. `forbidden` entries must not appear.
    """
    spec = references.get("list_match")
    if not isinstance(spec, Mapping):
        raise ValueError(
            "list_match requires a 'list_match' object in reference_answers"
        )

    items = spec.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("list_match requires a non-empty 'items' list")

    text = scope_text(answer, spec.get("scope", "full"))
    ordered = bool(spec.get("ordered", False))

    offsets = []
    for item in items:
        alternatives = as_entry_list(item, "list_match.items entry")
        if not alternatives:
            raise ValueError("list_match items must not be empty")
        matches = [
            offset
            for alternative in alternatives
            for offset in (find_entry(text, alternative),)
            if offset is not None
        ]
        if not matches:
            return False
        offsets.append(min(matches))

    if ordered and any(a >= b for a, b in zip(offsets, offsets[1:])):
        return False

    forbidden = as_entry_list(spec.get("forbidden"), "list_match.forbidden")
    return not any(contains_entry(text, entry) for entry in forbidden)


DETERMINISTIC_VERIFIERS: dict[str, Callable[[str, Mapping[str, Any]], bool]] = {
    "string_match": string_match,
    "number_match": number_match,
    "list_match": list_match,
}
