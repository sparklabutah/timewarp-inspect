"""Deterministic verifier tests.

The cases are taken from the upstream `src/tests/timewarp/test_evaluators.py`
(TimeWarp commit 4978e690ef2ad33d81a4a0c043da3f80588902a0) so that this port
keeps the upstream scoring behaviour.
"""

from decimal import Decimal
from typing import Any

import pytest

from timewarp.normalization import (
    extract_numbers,
    find_entry,
    first_sentence,
    normalize_text,
)
from timewarp.verifiers import list_match, number_match, string_match


def number_spec(scope: str = "full", **spec: Any) -> dict[str, Any]:
    return {"number_match": {"scope": scope, **spec}}


def list_spec(**spec: Any) -> dict[str, Any]:
    return {"list_match": spec}


class TestNormalizeText:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  Biology  ", "biology"),
            ('"Biology."', "biology"),
            ("**Biology**", "biology"),
            ("`Biology`", "biology"),
            ("Beyoncé", "beyonce"),
            ("Quest Lumaflex™ Band", "quest lumaflex band"),
            ("multi\n  line\ttext", "multi line text"),
            ("“smart quotes”", "smart quotes"),
            ("$9.99", "$9.99"),
            ("45%", "45%"),
            (None, ""),
        ],
    )
    def test_canonicalizes(self, raw: str | None, expected: str) -> None:
        assert normalize_text(raw) == expected

    def test_preserves_internal_punctuation(self) -> None:
        assert normalize_text("The U.S. government") == "the u.s. government"


class TestFirstSentence:
    def test_splits_on_terminator(self) -> None:
        assert first_sentence("Yes. It is mentioned in Biology.") == "Yes"

    def test_does_not_split_decimals(self) -> None:
        assert (
            first_sentence("The price is 9.99 dollars") == "The price is 9.99 dollars"
        )

    def test_splits_on_newline(self) -> None:
        assert first_sentence("No\nExplanation follows") == "No"

    def test_returns_whole_string_when_unsplittable(self) -> None:
        assert first_sentence("Biology") == "Biology"

    @pytest.mark.parametrize(
        "raw",
        [
            "**Yes.** There is no other article.",
            '"Yes." There is no other article.',
            "*Yes.* There is no other article.",
            "`Yes.` There is no other article.",
        ],
    )
    def test_splits_through_closing_markdown(self, raw: str) -> None:
        assert normalize_text(first_sentence(raw)) == "yes"


class TestWordBoundaryContainment:
    @pytest.mark.parametrize(
        ("answer", "reference", "matches"),
        [
            ("The answer is 10", "10", True),
            ("The answer is 100", "10", False),
            ("It happened in 2010", "10", False),
            ("There are 10, not 11", "10", True),
            ("The article is in the north wing", "no", False),
            ("No, it is not listed", "no", True),
            ("It is not listed", "no", False),
            ("Send an e-mail", "e mail", True),
            ("Send an e mail", "e-mail", True),
            ("Hong Kong's population", "hong kong", True),
            ("**Biology** is the answer", "biology", True),
        ],
    )
    def test_boundaries(self, answer: str, reference: str, matches: bool) -> None:
        assert (find_entry(answer, reference) is not None) is matches


class TestExtractNumbers:
    @pytest.mark.parametrize(
        ("text", "expected_subset"),
        [
            ("$1,234.56", ["1234.56"]),
            ("It costs 9.99 dollars", ["9.99"]),
            ("Over 57.7 million people", ["57700000.0"]),
            ("around 7 million", ["7000000"]),
            ("7,000,000", ["7000000"]),
            ("Thirteen articles", ["13"]),
            ("twenty one", ["21"]),
            ("one hundred two", ["102"]),
            ("The 13th of May", ["13"]),
            ("45%", ["45"]),
            ("-5 degrees", ["-5"]),
            ("no numbers here", []),
        ],
    )
    def test_extraction(self, text: str, expected_subset: list[str]) -> None:
        found = extract_numbers(text)
        for expected in expected_subset:
            assert any(value == Decimal(expected) for value in found), (text, found)
        if not expected_subset:
            assert found == []

    def test_ambiguous_suffix_yields_both_readings(self) -> None:
        found = extract_numbers("a 5m wingspan")
        assert Decimal(5) in found and Decimal(5_000_000) in found

    def test_metres_is_not_a_magnitude_suffix(self) -> None:
        found = extract_numbers("5 metres long")
        assert Decimal(5) in found and Decimal(5_000_000) not in found


class TestStringMatch:
    def test_exact_match_tolerates_formatting(self) -> None:
        assert string_match("  **Biology.**  ", {"exact_match": "Biology"})
        assert not string_match("Chemistry", {"exact_match": "Biology"})

    def test_exact_match_rejects_verbose_answers(self) -> None:
        assert not string_match("The answer is Biology", {"exact_match": "Biology"})

    def test_exact_match_list_is_any_of(self) -> None:
        refs = {"exact_match": ["Biology", "Biological science"]}
        assert string_match("biological science", refs)

    def test_must_include_is_containment(self) -> None:
        refs = {"must_include": ["biology"]}
        assert string_match("Biophysics is mentioned in the Biology article.", refs)
        assert not string_match("It appears only in Physics.", refs)

    def test_must_include_requires_all_entries(self) -> None:
        refs = {"must_include": ["biology", "physics"]}
        assert string_match("Both Biology and Physics mention it.", refs)
        assert not string_match("Only Biology mentions it.", refs)

    def test_or_alternatives(self) -> None:
        refs = {"must_include": ["kangaroos |OR| kangaroo"]}
        assert string_match("I found one kangaroo.", refs)
        assert string_match("I found kangaroos.", refs)
        assert not string_match("I found wombats.", refs)

    def test_or_separator_tolerates_missing_spaces(self) -> None:
        refs = {"must_include": ["kangaroos|OR|kangaroo"]}
        assert string_match("I found kangaroos.", refs)
        assert not string_match("I found wombats.", refs)

    def test_must_exclude(self) -> None:
        refs = {"must_include": ["biology"], "must_exclude": ["both", "neither"]}
        assert string_match("It is mentioned in Biology only.", refs)
        assert not string_match(
            "It is mentioned in both articles, Biology and Physics.", refs
        )

    def test_yes_no_with_first_sentence_scope(self) -> None:
        refs = {
            "must_include": ["yes"],
            "must_exclude": ["no"],
            "scope": "first_sentence",
        }
        assert string_match("Yes. There is no other article covering it.", refs)
        assert string_match("**Yes.** There is no other article covering it.", refs)
        assert not string_match("No. The article does not exist.", refs)
        assert string_match("Yes, although it is not in the related pages list.", refs)
        unscoped = {"must_include": ["yes"], "must_exclude": ["no"]}
        assert not string_match("Yes. There is no other article covering it.", unscoped)

    def test_regex_entry(self) -> None:
        refs = {"must_include": ["^yes.*$"], "scope": "first_sentence"}
        assert string_match("Yes, both articles mention it. Details follow.", refs)
        assert not string_match("No, neither does.", refs)

    def test_numeric_lookalike_is_rejected(self) -> None:
        refs = {"must_include": ["10"]}
        assert not string_match("There are 100 articles.", refs)
        assert string_match("There are 10 articles.", refs)

    def test_empty_spec_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one of"):
            string_match("anything", {})


class TestNumberMatch:
    def test_formatting_independence(self) -> None:
        refs = number_spec(value=7000000)
        for answer in [
            "7,000,000",
            "7000000",
            "7 million",
            "The difference is 7 million people.",
        ]:
            assert number_match(answer, refs), answer
        assert not number_match("70 million", refs)

    def test_relative_tolerance_for_hedged_golds(self) -> None:
        refs = number_spec(value=7000000, rel_tolerance=0.1)
        assert number_match("around 7.2 million", refs)
        assert not number_match("about 9 million", refs)

    def test_currency(self) -> None:
        refs = number_spec(value=9.99, abs_tolerance=0.01)
        assert number_match("It costs $9.99", refs)
        assert number_match("9.99 dollars", refs)
        assert not number_match("It costs $99.90", refs)

    def test_spelled_out_numbers(self) -> None:
        assert number_match("Thirteen articles are listed.", number_spec(value=13))
        assert number_match("13 articles are listed.", number_spec(value=13))

    def test_exact_by_default(self) -> None:
        assert not number_match("It was founded in 2011.", number_spec(value=2010))

    def test_multiple_required_values(self) -> None:
        refs = number_spec(values=[5, 1.8])
        assert number_match("The wingspan is 5 metres and the height 1.8 metres.", refs)
        assert not number_match("The wingspan is 5 metres.", refs)

    def test_no_numbers_in_answer(self) -> None:
        assert not number_match("I could not find it.", number_spec(value=13))

    def test_missing_value_raises(self) -> None:
        with pytest.raises(ValueError, match="'value' or 'values'"):
            number_match("13", number_spec())

    def test_missing_spec_raises(self) -> None:
        with pytest.raises(ValueError, match="'number_match' object"):
            number_match("13", {})


class TestListMatch:
    ANIMALS = [
        ["koalas |OR| koala"],
        ["kangaroos |OR| kangaroo"],
        ["wombats |OR| wombat"],
    ]

    def test_unordered_requires_every_item(self) -> None:
        refs = list_spec(items=self.ANIMALS)
        assert list_match("Wombats, koalas and kangaroos live there.", refs)
        assert not list_match("Koalas and kangaroos live there.", refs)

    def test_extra_items_are_tolerated(self) -> None:
        refs = list_spec(items=self.ANIMALS)
        assert list_match("Koalas, kangaroos, wombats, numbats and platypus.", refs)

    def test_ordered_enforces_sequence(self) -> None:
        refs = list_spec(
            items=[["periodic table"], ["atomic number"], ["proton"]], ordered=True
        )
        assert list_match("Periodic table, then atomic number, then proton.", refs)
        assert not list_match("Proton, then atomic number, then periodic table.", refs)

    def test_unordered_ignores_sequence(self) -> None:
        refs = list_spec(items=[["periodic table"], ["atomic number"]], ordered=False)
        assert list_match("Atomic number comes from the periodic table.", refs)

    def test_forbidden_entries(self) -> None:
        refs = list_spec(items=[["koalas"]], forbidden=["dingoes"])
        assert list_match("Koalas.", refs)
        assert not list_match("Koalas and dingoes.", refs)

    def test_plain_string_items(self) -> None:
        assert list_match("Koalas and wombats.", list_spec(items=["koalas", "wombats"]))

    def test_empty_items_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty 'items'"):
            list_match("anything", list_spec(items=[]))
