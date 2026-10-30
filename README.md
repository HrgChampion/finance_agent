# Finance Research Agent

A multi-agent system that writes an investment brief for a listed company. An orchestrator delegates
to five specialist agents, each with its own tools and real market data, and returns a rated brief
(BUY / HOLD / SELL) in which every number traces back to a tool output.

Built with [Strands Agents](https://strandsagents.com), LiteLLM (Gemini + Groq) and yfinance.

## Architecture

```
                         OrchestratorAgent  (director of research)
                                   │
   ① gather, in parallel           │
   ┌───────────────────────┬───────┴───────────────┬──────────────────────┐
   ▼                       ▼                       ▼                      │
FinancialDataAgent   NewsAndSentimentAgent      RiskAgent                 │
(junior analyst)     (news desk)                (the sceptic)             │
   │ financial report                                                     │
   ▼                                                                      │
② ValuationAgent  (DCF + peer multiples, uses the financial report)       │
   │                                                                      │
   ▼                                                                      │
③ ReportWriterAgent  (rating table → brief, no tools) ◄── all 4 reports ──┘
```

| Agent | Model | Tools |
|---|---|---|
| OrchestratorAgent | Gemini 3.5 Flash-Lite | `find_ticker`, plus the five agents below via `Agent.as_tool()` |
| FinancialDataAgent | Gemini 3.5 Flash-Lite | `get_income_statement`, `get_balance_sheet`*, `get_cash_flow_statement`, `compute_key_ratios`* |
| NewsAndSentimentAgent | Gemini 3.5 Flash-Lite | `search_recent_news`*, `analyze_sentiment` (FinBERT) |
| RiskAgent | Gemini 3.5 Flash-Lite | `get_balance_sheet`*, `compute_key_ratios`*, `search_recent_news`* |
| ValuationAgent | gpt-oss-120b on Groq | `run_dcf_model`, `get_peer_multiples` |
| ReportWriterAgent | Gemini 3.5 Flash-Lite | none (synthesis only) |

`*` shared tools: results are cached with `lru_cache`, so a second agent asking for the same data
costs no extra network call. Gemini falls back to Groq if it is unavailable.

### Design choices

- **Tools do the maths, agents do the reasoning.** Ratios, DCF, CAGR and peer premiums are computed
  in Python; agents are told never to invent numbers and to write "N/A" when data is missing.
- **Two-layer tools.** A cached helper fetches raw data; a `@tool` wrapper normalises the ticker,
  catches errors (returning an `ERROR: ...` string instead of raising) and returns compact text.
- **Deterministic rating.** The ReportWriter scores four dimensions 1-5 and derives the rating from
  their average (≥ 3.75 BUY, ≤ 2.25 SELL, otherwise HOLD), so the rating is explainable.

## Setup

Requires Python 3.12+ (developed on 3.14). Dependency versions are pinned to the ones this project
was verified with.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project folder:

```
GEMINI_API_KEY=...
GROQ_API_KEY=...
```

Both have free tiers. The first run downloads the FinBERT model (~440 MB) from Hugging Face.

Behind a corporate proxy that intercepts TLS, nothing extra is needed: `truststore` makes Python use
the Windows certificate store, and `tools.py` builds `ca_bundle.pem` for yfinance's `curl_cffi`.

## Usage

Ask anything about listed companies, by ticker or by name, in any market Yahoo Finance covers:

```
python bank_agent.py AAPL                                      # a bare ticker: full investment brief
python bank_agent.py "What is Apple's debt to equity ratio?"   # one question
python bank_agent.py "Compare the margins of Microsoft and Google"
python bank_agent.py "Is Reliance Industries overvalued?"
python bank_agent.py                                           # interactive session
```

The orchestrator plans each question: it resolves company names with `find_ticker`, calls only the
specialists the question needs (in parallel where it can), and answers from their reports. Questions
asking for a brief, a rating or "should I buy" run the full workflow and return a rated brief. In the
interactive session the conversation is kept, so follow-ups such as *"and what are its risks?"* work.
Questions outside listed companies are declined.

Each answer is saved to `briefs/`:

| File | Contents |
|---|---|
| `<date>_<time>_<question>.md` | the question and the answer |
| `<date>_<time>_<question>.metrics.md` | per-agent model, tokens, cycles, time and tool calls for that question |
| `<date>_<time>_session.log` | timestamped log of every agent, tool call and model call, including fallbacks |

A full brief takes about 40 seconds and uses roughly 35-40k tokens; narrower questions use far less. About half of those are the
orchestrator's, because every specialist report passes through its context. Step ① runs in parallel,
so its time is that of the slowest of the three agents (usually NewsAndSentimentAgent, because FinBERT
runs on the CPU).

## Tests

```
python -m pytest
```

The tests cover the tools and the non-LLM parts of the agent code: DCF maths and its guards (negative
free cash flow, currency mismatch), ratios, peer premiums, currency labels, news filtering, sentiment
labels, retry rules, the run summary, specialist history isolation, and the evaluation harness's own
scoring logic. yfinance and FinBERT are replaced with fakes, so the suite
runs offline, spends no tokens and finishes in under 30 seconds.

To try the tools against live Yahoo Finance data, without any LLM: `python tools.py`.

## Evaluation

`python evaluate.py` runs every case in [evals/cases.json](evals/cases.json) several times and scores
the runs with deterministic checks (no LLM judge). The latest report is in
[evals/REPORT.md](evals/REPORT.md).

| Check | How it is measured |
|---|---|
| Grounding | Every number in the answer is matched against the numbers in that run's tool outputs (0.5% rounding and unit changes allowed) |
| Rating consistency | The same ticker gets the same BUY / HOLD / SELL in every run |
| Rating rule | The stated rating matches the average-score rule |
| Workflow | Briefs call all five specialists, step ① in parallel, then valuation, then the writer (from hook timings) |
| Routing | Questions call the specialists they need, no others, and off-topic questions call none |

The grounding check was validated with a negative control: the same answer with its numbers shifted by
7% scores 18% instead of 100%.

**Results** (3 tickers x 3 brief runs, 5 questions x 3 runs, 2026-10-30):

| Metric | Result |
|---|---|
| Numbers traceable to a tool output | 97% (302 / 312) |
| Rating follows the score rule | 9 / 9 briefs |
| Same rating in every run | 2 / 3 tickers (RELIANCE.NS: HOLD, BUY, HOLD; its valuation score ranged 3-5) |
| Workflow followed | 7 / 9 briefs |
| Questions routed correctly | 9 / 9 completed runs |
| Mean cost | full brief 37 s / ~45k tokens; question 10 s / ~10k tokens |

What the evaluation found:

- **A real bug.** The first evaluation showed a ValuationAgent answering without calling any tool. Strands'
  `as_tool()` snapshots a specialist's conversation when the wrapper is created, so a second orchestrator
  in the same process inherited the previous run's history. `orchestrator_agent()` now clears it (with a
  regression test); tokens per brief fell from 86-120k to ~45k.
- **Prompted parallelism is not guaranteed.** In 2 of 9 briefs the orchestrator called RiskAgent a turn
  after the other two step ① specialists. Only a code-enforced workflow (e.g. a Strands Graph) would
  guarantee it.
- **The 10 untraceable numbers:** 5 were artifacts of the checker itself (a list number, rating averages),
  since fixed; 5 were values the model computed itself, such as an implied price from the peer P/E.
- **Not measured in this run:** the *Nvidia* and *weather* cases hit the Gemini free-tier quota after
  ~45 minutes of evaluation. Rerun them with `python evaluate.py --match "Nvidia|weather"`.

Raw runs, including every tool output, are saved to `evals/results/` (not committed), so a report can be
rebuilt without model calls: `python evaluate.py --rescore evals/results/<stamp>.json`.

## Project layout

| File | Purpose |
|---|---|
| `bank_agent.py` | agent definitions, orchestrator, CLI and run summary |
| `tools.py` | yfinance data tools, FinBERT sentiment, DCF and peer multiples |
| `observability.py` | Strands hooks and a LiteLLM callback for the live run log |
| `evaluate.py` | evaluation harness: grounding, consistency, workflow and routing checks |
| `evals/` | evaluation cases and the latest report |
| `tests/` | offline pytest suite |

## Known limitations

- **Not investment advice.** This is a learning project; ratings come from simple rules and LLM
  judgement.
- **DCF growth comes from a short revenue history.** The default growth rate is the revenue CAGR over
  the available years (usually 3-4), clamped to 0-15%. A starting year at a cyclical peak drags it
  down, which is why the ValuationAgent runs a second DCF when the latest year's growth differs by
  more than 3 points.
- **The discount rate is the CAPM cost of equity**, not a full WACC (no debt weighting), and the
  risk-free rate is always the US 10-year yield. For non-US companies the tool flags this, and the
  ValuationAgent may override it.
- **Currencies come from the data.** Statements are shown in the company's reporting currency and
  prices in the trading currency. When the two differ (ADRs such as TSM), the DCF is refused because
  a per-share value would mix currencies.
- **One call per specialist at a time.** Strands does not allow the same agent to run twice at once,
  so several companies are passed to a specialist together ("AAPL, MSFT") rather than in parallel calls.
- **Cash excludes short-term investments**, which overstates net debt for cash-rich companies such
  as Apple.
- **Built for non-financial companies.** Banks and insurers have no gross profit, current ratio or
  meaningful free cash flow. Those figures show as N/A, `run_dcf_model` refuses to run on negative free
  cash flow, and the valuation falls back to peer multiples alone.
- **Ratings and peers vary between runs.** LLM judgement drifts: the same data can score differently,
  and the ValuationAgent may pick a different peer set each time (it chooses peers itself).
- **News comes from Yahoo Finance search** and covers only the last few days of headlines. Relevance is
  imperfect: for JPM it also returns stories about *JPMorgan analysts' calls on other companies*, which
  are not news about JPM itself.
- **Free-tier rate limits are real.** Gemini allows 15 requests per minute and Groq 8,000 tokens per
  minute, so several questions in quick succession hit them. Each agent retries rate limits, overloads
  and network errors with backoff (4s, 8s, 16s...), and each model falls back to the other; the run
  log shows every failure and fallback in one line. The cross-model fallback only covers errors before a
  response starts streaming: a quota error mid-stream is retried on the same model, so once a daily
  quota is used up, runs fail until it resets.
