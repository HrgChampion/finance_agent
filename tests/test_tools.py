"""Offline tests for tools.py. The yfinance and FinBERT layer is replaced with fakes,
so these run in seconds with no network and no LLM calls."""
import pandas as pd
import pytest

import tools

YEARS = pd.to_datetime(["2025-09-30", "2024-09-30", "2023-09-30"])   # latest first, like yfinance


def frame(rows):
    """{'Total Revenue': [latest, ..., oldest]} -> DataFrame shaped like a yfinance statement."""
    return pd.DataFrame(rows, index=YEARS).T


@pytest.fixture
def company(monkeypatch):
    """A fake, healthy USD company. Tests tweak `info` or the statements to build edge cases."""
    data = {
        "info": {
            "currency": "USD", "financialCurrency": "USD", "sharesOutstanding": 100e6,
            "currentPrice": 50.0, "marketCap": 5e9, "totalDebt": 200e6, "totalCash": 100e6,
            "beta": 1.0, "trailingPE": 25.0,
        },
        "income": frame({
            "Total Revenue": [121e6, 110e6, 100e6],          # exactly 10% CAGR
            "Gross Profit": [60.5e6, 55e6, 50e6],
            "Operating Income": [24.2e6, 22e6, 20e6],
            "Net Income": [12.1e6, 11e6, 10e6],
        }),
        "balance": frame({
            "Stockholders Equity": [100e6, 90e6, 80e6],
            "Total Debt": [50e6, 50e6, 50e6],
            "Current Assets": [80e6, 70e6, 60e6],
            "Current Liabilities": [40e6, 35e6, 30e6],
        }),
        "cash_flow": frame({"Free Cash Flow": [10e6, 9e6, 8e6]}),
    }
    monkeypatch.setattr(tools, "_info", lambda t: data["info"])
    monkeypatch.setattr(tools, "_income_statement", lambda t: data["income"])
    monkeypatch.setattr(tools, "_balance_sheet", lambda t: data["balance"])
    monkeypatch.setattr(tools, "_cash_flow_statement", lambda t: data["cash_flow"])
    monkeypatch.setattr(tools, "_risk_free_rate", lambda: 0.04)
    return data


# --- small helpers ---------------------------------------------------------------------------

def test_ratio_handles_missing_and_zero():
    assert tools._ratio(10, 4) == 2.5
    assert tools._ratio(None, 4) is None
    assert tools._ratio(10, None) is None
    assert tools._ratio(10, 0) is None


def test_pct_and_num_formatting():
    assert tools._pct(0.469) == "46.9%"
    assert tools._pct(None) == "N/A"
    assert tools._num(1.3379) == "1.34"
    assert tools._num(None) == "N/A"


def test_format_uses_currency_millions_and_skips_missing_rows():
    df = frame({"Total Revenue": [121e6, 110e6, 100e6]})
    text = tools._format(df, ["Total Revenue", "Gross Profit"], "INR")
    assert text.startswith("INR millions")
    assert "Gross Profit" not in text
    assert "2025" in text and "121.0" in text


# --- DCF -------------------------------------------------------------------------------------

def test_dcf_value_flat_perpetuity():
    # zero growth, zero terminal growth: value = FCF / r
    assert tools._dcf_value(100, 0, 0.10, 0, 0, 0, 1)["per_share"] == pytest.approx(1000)


def test_dcf_value_adjusts_for_net_debt_and_shares():
    assert tools._dcf_value(100, 0, 0.10, 0, debt=200, cash=50, shares=2)["per_share"] == pytest.approx(425)


def test_dcf_value_moves_the_right_way():
    base = tools._dcf_value(100, 0.05, 0.10, 0.02, 0, 0, 1)["per_share"]
    assert tools._dcf_value(100, 0.08, 0.10, 0.02, 0, 0, 1)["per_share"] > base   # more growth
    assert tools._dcf_value(100, 0.05, 0.12, 0.02, 0, 0, 1)["per_share"] < base   # higher discount rate


def test_run_dcf_model_defaults(company):
    out = tools.run_dcf_model("abc")
    assert out.startswith("DCF for ABC (5-year, USD)")
    assert "Growth yrs 1-5: 10.0% (default: 2-yr revenue CAGR 10.0%" in out
    assert "Discount rate: 9.5% (default CAPM" in out            # 4% + 1.0 x 5.5%
    assert "(cash flow statement, FY2025)" in out
    assert "Sensitivity" in out
    assert "CAUTION" not in out


def test_run_dcf_model_clamps_negative_growth(company):
    company["income"] = frame({"Total Revenue": [80e6, 90e6, 100e6]})
    assert "Growth yrs 1-5: 0.0%" in tools.run_dcf_model("ABC")


def test_run_dcf_model_override_is_labelled(company):
    out = tools.run_dcf_model("ABC", growth_rate=0.03, discount_rate=0.08)
    assert "Growth yrs 1-5: 3.0% (override)" in out
    assert "Discount rate: 8.0% (override)" in out


def test_run_dcf_model_refuses_negative_fcf(company):
    company["cash_flow"] = frame({"Free Cash Flow": [-5e6, 9e6, 8e6]})
    assert tools.run_dcf_model("BANK").startswith("ERROR: DCF not meaningful")


def test_run_dcf_model_falls_back_to_yahoo_fcf(company):
    company["cash_flow"] = frame({"Operating Cash Flow": [20e6, 19e6, 18e6]})
    company["info"]["freeCashflow"] = 10e6
    assert "(Yahoo levered FCF, fallback)" in tools.run_dcf_model("ABC")


def test_run_dcf_model_refuses_currency_mismatch(company):
    company["info"]["financialCurrency"] = "TWD"                  # e.g. an ADR
    assert tools.run_dcf_model("TSM").startswith("ERROR: DCF not comparable")


def test_run_dcf_model_rejects_discount_rate_below_terminal_growth(company):
    out = tools.run_dcf_model("ABC", discount_rate=0.02, terminal_growth=0.025)
    assert out.startswith("ERROR: discount rate")


def test_run_dcf_model_flags_us_rate_for_other_currencies(company):
    company["info"].update(currency="INR", financialCurrency="INR")
    out = tools.run_dcf_model("RELIANCE.NS")
    assert "INR" in out
    fair_value_line = next(line for line in out.splitlines() if line.startswith("Fair value"))
    assert "CAUTION: discount rate uses the US risk-free rate" in fair_value_line


# --- statements and ratios -------------------------------------------------------------------

def test_statement_tools_label_the_reporting_currency(company):
    company["info"]["financialCurrency"] = "INR"
    assert tools.get_income_statement("ABC").startswith("INR millions")
    assert tools.get_balance_sheet("ABC").startswith("INR millions")
    assert tools.get_cash_flow_statement("ABC").startswith("INR millions")


def test_compute_key_ratios(company):
    out = tools.compute_key_ratios("ABC")
    assert "fiscal year 2025" in out
    assert "Gross margin: 50.0%" in out
    assert "Operating margin: 20.0%" in out
    assert "Net margin: 10.0%" in out
    assert "ROE: 12.1%" in out
    assert "Debt/Equity: 0.50" in out
    assert "Current ratio: 2.00" in out
    assert "Price: USD 50.00" in out


def test_compute_key_ratios_bank_has_no_current_ratio(company):
    company["balance"] = company["balance"].drop(index=["Current Assets", "Current Liabilities"])
    assert "Current ratio: N/A" in tools.compute_key_ratios("JPM")


def test_compute_key_ratios_empty_statements(company):
    company["income"] = pd.DataFrame()
    assert tools.compute_key_ratios("ABC").startswith("ERROR")


# --- peer multiples --------------------------------------------------------------------------

def test_peer_multiples_premium_and_skipped_peers(monkeypatch):
    infos = {
        "ABC": {"marketCap": 5e9, "trailingPE": 30.0, "currency": "USD"},
        "P1": {"marketCap": 4e9, "trailingPE": 10.0, "currency": "USD"},
        "P2": {"marketCap": 6e9, "trailingPE": 20.0, "currency": "USD"},
        "P3": {"marketCap": 3e9, "trailingPE": 30.0, "currency": "USD"},
        "FAKE": {},
    }
    monkeypatch.setattr(tools, "_info", lambda t: infos[t])
    out = tools.get_peer_multiples("ABC", ["p1", "P2", "P3", "FAKE", "ABC"])
    assert "P/E: ABC 30.0 vs median 20.0 -> 50% premium" in out   # median of peers only
    assert "Skipped (no data / invalid ticker): FAKE" in out
    assert "Fwd P/E: N/A" in out


def test_peer_multiples_needs_a_valid_peer(monkeypatch):
    monkeypatch.setattr(tools, "_info", lambda t: {"marketCap": 5e9} if t == "ABC" else {})
    assert tools.get_peer_multiples("ABC", ["FAKE"]).startswith("ERROR: no valid peers")


# --- search, news and sentiment --------------------------------------------------------------

class FakeSearch:
    quotes, news = [], []

    def __init__(self, *args, **kwargs):
        pass


def test_find_ticker_keeps_only_stocks(monkeypatch):
    FakeSearch.quotes = [
        {"symbol": "AAPL", "quoteType": "EQUITY", "longname": "Apple Inc.", "exchDisp": "NASDAQ"},
        {"symbol": "AAPL240119C", "quoteType": "OPTION"},
        {"symbol": "APLE", "quoteType": "EQUITY", "shortname": "Apple Hospitality", "exchange": "NYQ"},
    ]
    monkeypatch.setattr(tools.yf, "Search", FakeSearch)
    out = tools.find_ticker("Apple")
    assert "AAPL | Apple Inc. | NASDAQ" in out
    assert "APLE | Apple Hospitality | NYQ" in out
    assert "OPTION" not in out and "AAPL240119C" not in out


def test_find_ticker_no_match(monkeypatch):
    FakeSearch.quotes = []
    monkeypatch.setattr(tools.yf, "Search", FakeSearch)
    assert tools.find_ticker("zzqqxx").startswith("ERROR")


def test_news_filters_unrelated_and_undated_and_sorts_by_focus(monkeypatch):
    day = 1793318400                                               # 2026-10-30 UTC
    FakeSearch.news = [
        {"title": "Broad market story", "relatedTickers": ["ABC", "X", "Y"], "providerPublishTime": day},
        {"title": "Unrelated story", "relatedTickers": ["XYZ"], "providerPublishTime": day},
        {"title": "Undated story", "relatedTickers": ["ABC"]},
        {"title": "ABC-only story", "relatedTickers": ["ABC"], "providerPublishTime": day - 86400},
    ]
    monkeypatch.setattr(tools.yf, "Search", FakeSearch)
    items = tools._news.__wrapped__("ABC")                         # bypass the cache
    assert [i["title"] for i in items] == ["ABC-only story", "Broad market story"]
    assert items[0]["date"] == "2026-10-29"


def fake_finbert(labels):
    """A stand-in FinBERT pipeline returning one confident label per headline."""
    def pipeline(titles):
        return [[{"label": l, "score": 0.9 if l == label else 0.05} for l in ("positive", "neutral", "negative")]
                for label in labels]
    return lambda: pipeline


@pytest.mark.parametrize("labels, overall", [
    (["positive", "positive", "neutral"], "Positive"),
    (["negative", "negative", "neutral"], "Negative"),
    (["neutral", "neutral", "neutral"], "Neutral"),
    (["positive", "positive", "negative", "negative"], "Mixed"),
])
def test_analyze_sentiment_overall_label(monkeypatch, labels, overall):
    items = tuple({"date": "2026-10-30", "title": f"headline {i}"} for i in range(len(labels)))
    monkeypatch.setattr(tools, "_news", lambda t: items)
    monkeypatch.setattr(tools, "_finbert", fake_finbert(labels))
    assert f"Overall: {overall}" in tools.analyze_sentiment("ABC")


def test_analyze_sentiment_without_news(monkeypatch):
    monkeypatch.setattr(tools, "_news", lambda t: ())
    assert tools.analyze_sentiment("ABC").startswith("ERROR")
