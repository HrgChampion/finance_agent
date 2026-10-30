"""Tests for the evaluation harness's scoring logic (no agents are run)."""
import pytest

import evaluate as ev

INF = float("inf")


def test_numbers_parses_money_percent_and_skips_glued_tokens():
    assert ev.numbers("FCF $98,767.0 million, -77% downside, FY2025, Q3, P/E 40.01.") == [98767.0, -77.0, 40.01]


@pytest.mark.parametrize("x, expected", [(2025.0, False), (3.0, False), (1.0, False), (12.0, True), (0.89, True)])
def test_checkable_skips_years_and_scores(x, expected):
    assert ev.checkable(x) is expected


def test_numbers_with_decimals():
    assert ev.numbers("35% and 0.04 and 1,575.64", with_decimals=True) == [(35.0, 0), (0.04, 2), (1575.64, 2)]


def test_grounded_allows_rounding_and_unit_changes():
    facts = {98767.0, 5092.0, 0.89, 77.0, 35.4}
    assert ev.grounded(98.8, facts, 1)        # $98.8B vs 98,767 USD millions
    assert ev.grounded(5.09, facts, 2)        # $5.09 trillion vs 5,092 billion
    assert ev.grounded(0.89, facts, 2)
    assert ev.grounded(-77, facts, 0)         # "-77%" vs "77% downside"
    assert ev.grounded(35, facts, 0)          # "35%" written from 35.4
    assert not ev.grounded(35.0, facts, 1)    # "35.0" claims more precision than 35.4 allows
    assert not ev.grounded(262, facts, 0)     # an implied price the model computed itself
    assert not ev.grounded(0.04, {0.0001}, 2) # precision-based tolerance, not a fixed one


def test_grounding_share_and_ungrounded_context():
    answer = "Net margin is 26.9% in 2025.\nImplied price ~$262 per share.\n*(Average Score: 2.75)*"
    share, checked, missing = ev.grounding(answer, ["Net margin: 26.9%"], derived=[2.75])
    assert checked == 3                       # 2025 is a year; the average is a derived number
    assert share == pytest.approx(2 / 3)
    assert missing == ["262 in: Implied price ~$262 per share."]


def test_grounding_ignores_list_markers():
    assert ev.grounding("6. Bottom line: margin 26.9%", ["26.9"]) == (1.0, 1, [])


def test_score_grounding_counts_the_rating_average_as_derived():
    result = {"answer": BRIEF + "\nThe average score of 2.67 implies HOLD.", "tool_texts": []}
    ev.score_grounding(result)
    assert result["ungrounded"] == []         # (4 + 1 + 3) / 3 = 2.67, Sentiment N/A excluded


def test_grounding_with_no_numbers():
    assert ev.grounding("Outside my scope.", ["anything 1.5"]) == (None, 0, [])


BRIEF = """| Dimension | Assessment | Score (1-5) |
|---|---|---|
| Financials | Strong | 4 |
| Valuation | Overvalued | 1 |
| Risk | Medium | 3 |
| **Sentiment** | Unavailable | **N/A** |

1. **HOLD**: balanced."""


def test_parse_brief():
    rating, scores = ev.parse_brief(BRIEF)
    assert rating == "HOLD"
    assert scores == {"Financials": 4.0, "Valuation": 1.0, "Risk": 3.0, "Sentiment": None}


@pytest.mark.parametrize("scores, expected", [
    ({"a": 4, "b": 1, "c": 3, "d": None}, "HOLD"),     # average 2.67
    ({"a": 4, "b": 3, "c": 3.5}, "HOLD"),              # average 3.5
    ({"a": 4, "b": 4, "c": 4, "d": 3}, "BUY"),         # 3.75 is BUY
    ({"a": 2, "b": 2, "c": 2.75}, "SELL"),             # 2.25 is SELL
    ({"a": None}, None),
])
def test_rule_rating(scores, expected):
    assert ev.rule_rating(scores) == expected


def events(*calls):
    """(tool, start, end) tuples -> recorder events from the orchestrator."""
    out = []
    for tool, start, end in calls:
        out.append({"agent": "OrchestratorAgent", "kind": "start", "tool": tool, "t": start, "text": ""})
        out.append({"agent": "OrchestratorAgent", "kind": "end", "tool": tool, "t": end, "text": ""})
    return sorted(out, key=lambda e: e["t"])


GOOD_BRIEF = events(("FinancialDataAgent", 0, 5), ("NewsAndSentimentAgent", 0.1, 9), ("RiskAgent", 0.1, 4),
                    ("ValuationAgent", 10, 18), ("ReportWriterAgent", 20, 23))


def test_workflow_ok():
    assert ev.check_workflow(ev.spans(GOOD_BRIEF)) == []


def test_workflow_sequential_step_1_is_flagged():
    sequential = events(("FinancialDataAgent", 0, 5), ("NewsAndSentimentAgent", 6, 9), ("RiskAgent", 10, 12),
                        ("ValuationAgent", 13, 18), ("ReportWriterAgent", 20, 23))
    assert ev.check_workflow(ev.spans(sequential)) == ["step 1 specialists did not run in parallel"]


def test_workflow_missing_specialist_is_flagged():
    sp = ev.spans(GOOD_BRIEF[:-2])                       # no ReportWriterAgent
    assert "instead of all five" in ev.check_workflow(sp)[0]


def test_routing():
    case = {"required": ["FinancialDataAgent", "ValuationAgent"], "allowed": ["FinancialDataAgent", "ValuationAgent"]}
    assert ev.check_routing(ev.spans(events(("FinancialDataAgent", 0, 5), ("ValuationAgent", 6, 9))), case) == []

    early = ev.check_routing(ev.spans(events(("FinancialDataAgent", 0, 5), ("ValuationAgent", 1, 9))), case)
    assert early == ["ValuationAgent started before FinancialDataAgent finished"]

    extra = ev.check_routing(ev.spans(events(("FinancialDataAgent", 0, 5), ("RiskAgent", 0, 3))), case)
    assert extra == ["missing ['ValuationAgent']", "unnecessary ['RiskAgent']"]


def test_routing_out_of_scope_and_repeats():
    assert ev.check_routing({}, {"allowed": []}) == []
    repeated = ev.spans(events(("FinancialDataAgent", 0, 2), ("FinancialDataAgent", 3, 5)))
    assert ev.check_routing(repeated, {"allowed": ["FinancialDataAgent"]}) == ["called more than once: ['FinancialDataAgent']"]


def test_spans_pairs_unfinished_calls():
    sp = ev.spans(events(("RiskAgent", 0, 3))[:1])       # started, never finished
    assert sp == {"RiskAgent": [[0, INF]]}
