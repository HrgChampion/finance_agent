from strands import tool
import truststore
from functools import lru_cache
import yfinance as yf
import pandas as pd
import ssl, certifi
from pathlib import Path
from datetime import datetime, timezone
from curl_cffi import requests as curl_requests
import logging
import warnings

# yfinance's own code triggers pandas inplace deprecation warnings; not ours to fix
warnings.filterwarnings("ignore", message=".*inplace keyword.*")
# yfinance prints e.g. "HTTP Error 404 ... Quote not found" itself; our tools already report errors
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

def _build_ca_bundle() -> str:
    """certifi + Windows trusted roots (incl. corporate proxy CA), for yfinance's curl_cffi."""
    pem = Path(__file__).with_name("ca_bundle.pem")
    with pem.open("w") as f:
        f.write(Path(certifi.where()).read_text())
        for store in ("ROOT", "CA"):
            for cert, enc, _ in ssl.enum_certificates(store):
                if enc == "x509_asn":
                    f.write(ssl.DER_cert_to_PEM_cert(cert))
    return str(pem)

SESSION = curl_requests.Session(impersonate="chrome", verify=_build_ca_bundle())

truststore.inject_into_ssl()

BALANCE_ROWS =[
"Total Assets","Total Liabilities Net Minority Interest","Stockholders Equity",
"Total Debt","Cash And Cash Equivalents","Current Assets","Current Liabilities"
]

INCOME_ROWS = ["Total Revenue", "Gross Profit", "Operating Income", "Net Income", "EBITDA"]
CASH_FLOW_ROWS = ["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow"]


@lru_cache(maxsize=32)
def _balance_sheet(ticker:str) -> pd.DataFrame:
    """Cached raw fetch. Shared by get_balance_sheet and compute_key_ratios."""
    return yf.Ticker(ticker,session=SESSION).balance_sheet

def _format(df:pd.DataFrame,rows:list[str],currency:str="USD") ->str:
    """Keep existing rows,last 4 years,in millions of the reporting currency, as text."""
    rows= [r for r in rows if r in df.index]
    out = (df.loc[rows].iloc[:,:4]/1e6).round(0)
    out.columns =[c.strftime("%Y") for c in out.columns]
    return f"{currency} millions\n" + out.to_string()

@lru_cache(maxsize=32)
def _income_statement(ticker:str) -> pd.DataFrame:
    """Get the income statement """
    return yf.Ticker(ticker,session=SESSION).income_stmt

@lru_cache(maxsize=32)
def _cash_flow_statement(ticker:str) ->pd.DataFrame:
    """Get the cash flow statemnt of the ticker"""
    return yf.Ticker(ticker,session=SESSION).cashflow

@lru_cache
def _info(ticker) ->dict:
    """Get the information of the ticker"""
    return yf.Ticker(ticker,session=SESSION).info

def _currency(ticker: str, kind: str = "financial") -> str:
    """Currency of the financial statements ('financial') or of the share price ('trading'), e.g. 'INR'."""
    info = _info(ticker)
    if kind == "financial":
        return info.get("financialCurrency") or info.get("currency") or "USD"
    return info.get("currency") or "USD"

def _ratio(a,b):
    """a / b, or None if either is missing or b is zero."""
    if a is None or b is None or b==0 :
        return None
    return a/b

def _pct(x):
    """0.469 -> '46.9%', None -> 'N/A'."""
    return "N/A" if x is None else f"{x:.1%}"

def _num(x):
    """1.3379 -> '1.34', None -> 'N/A'."""
    return "N/A" if x is None else f"{x:.2f}"

@lru_cache
def _finbert():
    from transformers import pipeline
    return pipeline("text-classification", model="ProsusAI/finbert", top_k=None)


@lru_cache(maxsize=32)
def _news(ticker: str) -> tuple:
    """Cached news fetch. Shared by search_recent_news, analyze_sentiment (and RiskAgent)."""
    raw = yf.Search(ticker, news_count=15, session=SESSION).news
    items = []
    for n in raw:
        related = n.get("relatedTickers") or []
        if ticker not in related or "providerPublishTime" not in n:
            continue                                  # skip unrelated / undated articles
        items.append({
            "date": datetime.fromtimestamp(n["providerPublishTime"], tz=timezone.utc).strftime("%Y-%m-%d"),
            "publisher": n.get("publisher", "Unknown"),
            "title": n.get("title", "").strip(),
            "focus": len(related),                    # 1 = article is about this company only
        })
    items.sort(key=lambda i: i["date"], reverse=True) # newest first...
    items.sort(key=lambda i: i["focus"])              # ...then most focused first (stable sort keeps date order)
    return tuple(items[:8])


@tool
def search_recent_news(ticker: str) -> str:
    """Get recent news headlines (date, publisher, title) for a stock ticker, e.g. 'AAPL', most company-specific first."""
    ticker = ticker.strip().upper()
    try:
        items = _news(ticker)
        if not items:
            return f"ERROR: no recent news found for {ticker}"
        lines = [f"Recent news for {ticker} (most company-specific first):"]
        for i in items:
            tickers = "ticker" if i["focus"] == 1 else "tickers"
            lines.append(f"{i['date']} | {i['publisher']} | {i['title']} | focus: {i['focus']} {tickers}")
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: could not fetch news for {ticker}: {e}"


@tool
def find_ticker(company: str) -> str:
    """Find the stock ticker symbol for a company name, e.g. 'Apple' -> AAPL.
    Returns up to 5 matching listed stocks (symbol | name | exchange), best match first."""
    try:
        quotes = yf.Search(company, max_results=10, news_count=0, session=SESSION).quotes
        stocks = [q for q in quotes if q.get("quoteType") == "EQUITY"][:5]
        if not stocks:
            return f"ERROR: no listed stock found for '{company}'"
        lines = [f"Stocks matching '{company}' (best match first):"]
        for q in stocks:
            name = q.get("longname") or q.get("shortname") or "?"
            lines.append(f"{q['symbol']} | {name} | {q.get('exchDisp') or q.get('exchange', '?')}")
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: could not search for '{company}': {e}"



@tool
def get_income_statement(ticker:str) ->str:
    """Get the annual income statement (last 4 years, in millions of the reporting currency) for a stock ticker, e.g. 'AAPL'."""
    ticker = ticker.strip().upper()
    try:
        df = _income_statement(ticker)
        if df.empty:
            return f"Error: no income statement data found for {ticker}"
        return _format(df,INCOME_ROWS,_currency(ticker))
    except Exception as e:
        return f"Error:could not fetch income statement for {ticker} :{e}"
        
@tool
def get_balance_sheet(ticker: str) -> str:
    """Get the annual balance sheet (last 4 years, in millions of the reporting currency) for a stock ticker, e.g. 'AAPL'."""
    ticker = ticker.strip().upper()
    try:
        df = _balance_sheet(ticker)
        if df.empty:
            return f"Error: no balance sheet data found for {ticker}"
        return _format(df,BALANCE_ROWS,_currency(ticker))
    except Exception as e:
        return f"Error:could not fetch balance sheet for {ticker} :{e}"
    

@tool
def get_cash_flow_statement(ticker : str) -> str:
    """Get the annual cashflow statement (last 4 years, in millions of the reporting currency) for a stock ticker, e.g. 'AAPL'."""
    ticker = ticker.strip().upper()
    try:
        df =_cash_flow_statement(ticker)
        if df.empty:
            return f"ERROR: no cash flow data found for {ticker}"
        return _format(df,CASH_FLOW_ROWS,_currency(ticker))
    except Exception as e:
        return f"ERROR: could not fetch cash flow for {ticker} : {e}"
    

@tool
def compute_key_ratios(ticker: str) -> str:
    """Compute key financial ratios for the latest fiscal year of a stock ticker, e.g. 'AAPL':
    gross/operating/net margin, ROE, debt/equity, current ratio, plus price, trailing P/E and market cap."""
    ticker = ticker.strip().upper()
    try:
        inc_df = _income_statement(ticker)
        bal_df = _balance_sheet(ticker)
        info = _info(ticker)
        if inc_df.empty or bal_df.empty:
            return f"ERROR: no financial statement data found for {ticker}"

        inc = inc_df.iloc[:, 0]                      # latest fiscal year
        bal = bal_df.iloc[:, 0]
        year = inc_df.columns[0].strftime("%Y")
        revenue = inc.get("Total Revenue")
        equity = bal.get("Stockholders Equity")
        price = info.get("currentPrice")
        mcap = info.get("marketCap")
        ccy = _currency(ticker, "trading")

        lines = [
            f"Key ratios for {ticker} (fiscal year {year})",
            f"Gross margin: {_pct(_ratio(inc.get('Gross Profit'), revenue))}",
            f"Operating margin: {_pct(_ratio(inc.get('Operating Income'), revenue))}",
            f"Net margin: {_pct(_ratio(inc.get('Net Income'), revenue))}",
            f"ROE: {_pct(_ratio(inc.get('Net Income'), equity))}",
            f"Debt/Equity: {_num(_ratio(bal.get('Total Debt'), equity))}",
            f"Current ratio: {_num(_ratio(bal.get('Current Assets'), bal.get('Current Liabilities')))}",
            f"Price: {'N/A' if price is None else f'{ccy} {price:,.2f}'}",
            f"P/E (trailing): {_num(info.get('trailingPE'))}",
            f"Market cap: {'N/A' if mcap is None else f'{ccy} {mcap / 1e9:,.0f} billion'}",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: could not compute ratios for {ticker}: {e}"


@tool
def analyze_sentiment(ticker: str) -> str:
    """Score recent news headlines for a stock ticker, e.g. 'AAPL', with FinBERT (finance-tuned model).
    Returns each headline's score from -1.0 (negative) to +1.0 (positive) and the overall sentiment."""
    ticker = ticker.strip().upper()
    try:
        items = _news(ticker)
        if not items:
            return f"ERROR: no recent news found for {ticker} to analyze"
        results = _finbert()([i["title"] for i in items])

        lines = [f"FinBERT sentiment for {ticker} (score = P(positive) - P(negative)):"]
        scores = []
        counts = {"positive": 0, "neutral": 0, "negative": 0}
        for item, result in zip(items, results):
            probs = {r["label"]: r["score"] for r in result}
            score = probs["positive"] - probs["negative"]
            label = max(probs, key=probs.get)
            scores.append(score)
            counts[label] += 1
            lines.append(f"{item['date']} | {score:+.2f} {label} | {item['title']}")

        avg = sum(scores) / len(scores)
        if counts["positive"] >= 2 and counts["negative"] >= 2:
            overall = "Mixed"
        elif avg > 0.15:
            overall = "Positive"
        elif avg < -0.15:
            overall = "Negative"
        else:
            overall = "Neutral"
        lines.append(
            f"Overall: {overall} (avg {avg:+.2f}) | "
            f"{counts['positive']} positive / {counts['neutral']} neutral / {counts['negative']} negative"
        )
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: could not analyze sentiment for {ticker}: {e}"


MULTIPLES = {
    "trailingPE": "P/E",
    "forwardPE": "Fwd P/E",
    "enterpriseToEbitda": "EV/EBITDA",
    "priceToSalesTrailing12Months": "P/S",
}


@tool
def get_peer_multiples(ticker: str, peers: list[str]) -> str:
    """Compare valuation multiples (P/E, forward P/E, EV/EBITDA, P/S) of a stock against peer tickers you choose,
    e.g. ticker='AAPL', peers=['MSFT', 'GOOGL']. Returns a table and the premium/discount vs. the peer median."""
    ticker = ticker.strip().upper()
    symbols = [ticker] + [p.strip().upper() for p in peers if p.strip().upper() != ticker][:6]
    try:
        rows, skipped = {}, []
        for sym in symbols:
            info = _info(sym)
            if not info.get("marketCap"):
                skipped.append(sym)                      # invalid ticker or no data: skip, don't crash
                continue
            row = {"Company": (info.get("shortName") or sym)[:22],
                   "Ccy": info.get("currency") or "?",
                   "Mkt cap B": info["marketCap"] / 1e9}
            for key, label in MULTIPLES.items():
                row[label] = info.get(key)
            rows[sym] = row
        if ticker not in rows:
            return f"ERROR: no valuation data found for {ticker}"
        if len(rows) < 2:
            return f"ERROR: no valid peers found for {ticker} (skipped: {skipped})"

        df = pd.DataFrame.from_dict(rows, orient="index")
        median = df.drop(index=ticker).median(numeric_only=True)   # peers only, not the target
        df.loc["PEER MEDIAN"] = median
        df.loc["PEER MEDIAN", ["Company", "Ccy"]] = "-"

        lines = [f"Valuation multiples: {ticker} vs peers", df.round(1).to_string(), "",
                 "Premium / discount vs peer median:"]
        for label in MULTIPLES.values():
            own, med = df.loc[ticker, label], median.get(label)
            if pd.isna(own) or pd.isna(med) or med == 0:
                lines.append(f"{label}: N/A")
            else:
                diff = own / med - 1
                word = "premium" if diff >= 0 else "discount"
                lines.append(f"{label}: {ticker} {own:.1f} vs median {med:.1f} -> {abs(diff):.0%} {word}")
        if skipped:
            lines.append(f"Skipped (no data / invalid ticker): {', '.join(skipped)}")
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: could not fetch peer multiples for {ticker}: {e}"

EQUITY_RISK_PREMIUM = 0.055
DCF_YEARS = 5

@lru_cache(maxsize=1)
def _risk_free_rate() ->float:
    """10 year US Treasury yield as a decimal (^TNX), fallback 4.5%"""
    try:
        hist = yf.Ticker("^TNX", session=SESSION).history(period="5d")
        return float(hist["Close"].iloc[-1]) / 100
    except Exception:
        return 0.045

def _dcf_value(fcf0, g, r, tg, debt, cash, shares, years=DCF_YEARS) -> dict:
    """Pure DCF maths. Returns fair value per share and its components."""
    pv_fcf = sum(fcf0 * (1 + g) ** t / (1 + r) ** t for t in range(1, years + 1))
    fcf_n = fcf0 * (1 + g) ** years
    pv_tv = fcf_n * (1 + tg) / (r - tg) / (1 + r) ** years
    ev = pv_fcf + pv_tv
    return {"pv_fcf": pv_fcf, "pv_tv": pv_tv, "ev": ev,
            "per_share": (ev - debt + cash) / shares}


@tool
def run_dcf_model(ticker: str, growth_rate: float | None = None,
                  discount_rate: float | None = None, terminal_growth: float = 0.025) -> str:
    """Run a 5-year discounted cash flow (DCF) valuation for a stock ticker, e.g. 'AAPL'.
    Optional assumptions as decimals (0.06 = 6%): growth_rate for years 1-5, discount_rate, terminal_growth.
    Omit them to use defaults (revenue CAGR growth, CAPM discount rate, 2.5% terminal).
    Returns fair value per share vs current price, all assumptions, and a sensitivity table."""
    ticker = ticker.strip().upper()
    try:
        info = _info(ticker)
        shares = info.get("sharesOutstanding")
        price = info.get("currentPrice")
        debt = info.get("totalDebt") or 0
        cash = info.get("totalCash") or 0
        ccy = _currency(ticker)
        if ccy != _currency(ticker, "trading"):
            return (f"ERROR: DCF not comparable for {ticker}: statements are in {ccy} but the shares trade in "
                    f"{_currency(ticker, 'trading')} (e.g. an ADR). Value it on peer multiples instead.")

        fcf0, fcf_src = None, ""
        cf = _cash_flow_statement(ticker)
        if "Free Cash Flow" in cf.index:
            fcf_hist = cf.loc["Free Cash Flow"].dropna()
            if not fcf_hist.empty:
                fcf0 = float(fcf_hist.iloc[0])
                fcf_src = f"(cash flow statement, FY{fcf_hist.index[0]:%Y})"
        if not fcf0:
            fcf0 = info.get("freeCashflow")
            fcf_src = "(Yahoo levered FCF, fallback)"
        if not fcf0 or fcf0 <= 0:
            return (f"ERROR: DCF not meaningful for {ticker}: latest free cash flow is negative or unavailable "
                    f"(typical for banks and insurers). Value it on peer multiples instead.")

        # growth default: historical revenue CAGR, clamped to 0-15%
        g_note = "(override)"
        if growth_rate is None:
            rev = _income_statement(ticker).loc["Total Revenue"].dropna()
            n = len(rev) - 1
            cagr = (rev.iloc[0] / rev.iloc[-1]) ** (1 / n) - 1 if n > 0 and rev.iloc[-1] > 0 else 0.05
            growth_rate = min(max(cagr, 0.0), 0.15)
            g_note = f"(default: {n}-yr revenue CAGR {cagr:.1%}, clamped 0-15%)"

        # discount rate default: CAPM cost of equity
        r_note = "(override)"
        if discount_rate is None:
            rf = _risk_free_rate()
            beta = info.get("beta") or 1.0
            discount_rate = rf + beta * EQUITY_RISK_PREMIUM
            r_note = f"(default CAPM: rf {rf:.2%} + beta {beta:.2f} x ERP {EQUITY_RISK_PREMIUM:.1%})"
            if ccy != "USD":
                r_note += f" - NOTE: rf is the US 10-yr yield; {ccy} rates may differ, override if justified"

        if discount_rate <= terminal_growth:
            return (f"ERROR: discount rate ({discount_rate:.1%}) must be above "
                    f"terminal growth ({terminal_growth:.1%})")

        base = _dcf_value(fcf0, growth_rate, discount_rate, terminal_growth, debt, cash, shares)
        fair = base["per_share"]

        lines = [
            f"DCF for {ticker} ({DCF_YEARS}-year, {ccy})",
            f"Starting FCF: {ccy} {fcf0 / 1e9:,.1f}B {fcf_src} | Debt {ccy} {debt / 1e9:,.1f}B | "
            f"Cash {ccy} {cash / 1e9:,.1f}B | Shares {shares / 1e9:,.2f}B",
            f"Growth yrs 1-{DCF_YEARS}: {growth_rate:.1%} {g_note}",
            f"Discount rate: {discount_rate:.1%} {r_note}",
            f"Terminal growth: {terminal_growth:.1%}",
            f"PV of {DCF_YEARS}-yr cash flows: {ccy} {base['pv_fcf'] / 1e9:,.0f}B | "
            f"PV of terminal value: {ccy} {base['pv_tv'] / 1e9:,.0f}B ({base['pv_tv'] / base['ev']:.0%} of EV)",
        ]
        # on the fair-value line itself, so the caveat travels with the number
        caveat = " (CAUTION: discount rate uses the US risk-free rate)" if ccy != "USD" and r_note.startswith("(default") else ""
        if price:
            move = fair / price - 1
            lines.append(f"Fair value per share: {ccy} {fair:,.2f} vs price {ccy} {price:,.2f} -> "
                         f"{abs(move):.0%} {'upside' if move >= 0 else 'downside'}{caveat}")
        else:
            lines.append(f"Fair value per share: {ccy} {fair:,.2f} (current price unavailable){caveat}")

        # sensitivity: discount rate +/-1%, terminal growth +/-0.5%
        tgs = [terminal_growth - 0.005, terminal_growth, terminal_growth + 0.005]
        rs = [discount_rate - 0.01, discount_rate, discount_rate + 0.01]
        lines.append(f"Sensitivity (fair value per share, {ccy}):")
        lines.append("            " + "".join(f"tg {t:>4.1%}   " for t in tgs))
        for r in rs:
            cells = ""
            for t in tgs:
                cell = f"{_dcf_value(fcf0, growth_rate, r, t, debt, cash, shares)['per_share']:,.0f}" if r > t else "n/a"
                cells += f"{cell:>9}  "
            lines.append(f"r {r:>6.1%}  {cells}")
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: could not run DCF for {ticker}: {e}"




if __name__ == "__main__":
    print(get_income_statement("AAPL"))
    print(get_balance_sheet("AAPL"))
    print(get_cash_flow_statement("AAPL"))
    print(compute_key_ratios("AAPL"))
    print(search_recent_news("AAPL"))
    print(analyze_sentiment("AAPL"))
    print(get_peer_multiples("AAPL", ["MSFT", "GOOGL", "AMZN", "META", "FAKEXX"]))
    print(run_dcf_model("AAPL"))
    print(run_dcf_model("AAPL", growth_rate=0.08, discount_rate=0.09))  