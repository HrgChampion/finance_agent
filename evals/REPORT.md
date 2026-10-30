# Evaluation report

2026-10-30 18:32 | 3 runs per case | cases: `evals/cases.json`

## Summary

| Metric | Result |
|---|---|
| Runs completed | 75% (18/24) |
| Numbers traceable to a tool output | 97% (302/312) |
| Briefs: same rating in every run | 67% (2/3) of tickers |
| Briefs: rating follows the score rule | 100% (9/9) |
| Briefs: workflow followed | 78% (7/9) |
| Briefs: mean time / tokens | 37s / 44,917 |
| Questions: routed correctly | 100% (9/9) |
| Questions: mean time / tokens | 10s / 9,977 |

## Briefs

| Ticker | Ratings | Score range Financials / Valuation / Risk / Sentiment | Rule ok | Workflow ok | Grounded | Time | Tokens |
|---|---|---|---|---|---|---|---|
| AAPL | HOLD, HOLD, HOLD | 3-4 / 1 / 3 / 4 | 3/3 | 1/3 | 99% | 52s | 47,772 |
| JPM | HOLD, HOLD, HOLD | 3-4 / 2 / 3 / 3 | 3/3 | 3/3 | 90% | 27s | 40,784 |
| RELIANCE.NS | HOLD, BUY, HOLD | 4 / 3-5 / 3* / 3* | 3/3 | 3/3 | 99% | 30s | 46,196 |

`*` a run scored this dimension N/A (data unavailable).

## Questions

| Question | Routed ok | Specialists called (most common) | Grounded | Time | Tokens |
|---|---|---|---|---|---|
| What is Apple's debt to equity ratio? | 3/3 | FinancialDataAgent | 100% | 9s | 8,733 |
| Is there any negative news about Tesla? | 3/3 | NewsAndSentimentAgent, RiskAgent | 100% | 9s | 10,327 |
| Compare the operating margins of Microsoft and Google | 3/3 | FinancialDataAgent | 100% | 11s | 10,871 |

## Problems

- **Write an investment brief for AAPL** (run 2): step 1 specialists did not run in parallel
- **Write an investment brief for AAPL** (run 3): step 1 specialists did not run in parallel
- **Is Nvidia overvalued?** (run 1): EventLoopException: litellm.MidStreamFallbackError: litellm.RateLimitError: litellm.RateLimitError: vertex_ai_betaException - b'{\n  "error": {\n    "code": 429,\n    "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to:; missing ['ValuationAgent']; unnecessary ['NewsAndSentimentAgent']
- **Is Nvidia overvalued?** (run 2): MidStreamFallbackError: litellm.MidStreamFallbackError: litellm.RateLimitError: litellm.RateLimitError: vertex_ai_betaException - b'{\n  "error": {\n    "code": 429,\n    "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head; missing ['FinancialDataAgent', 'ValuationAgent']
- **Is Nvidia overvalued?** (run 3): MidStreamFallbackError: litellm.MidStreamFallbackError: litellm.RateLimitError: litellm.RateLimitError: vertex_ai_betaException - b'{\n  "error": {\n    "code": 429,\n    "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head; missing ['FinancialDataAgent', 'ValuationAgent']
- **What's the weather in Delhi today?** (run 1): MidStreamFallbackError: litellm.MidStreamFallbackError: litellm.RateLimitError: litellm.RateLimitError: vertex_ai_betaException - b'{\n  "error": {\n    "code": 429,\n    "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head
- **What's the weather in Delhi today?** (run 2): MidStreamFallbackError: litellm.MidStreamFallbackError: litellm.RateLimitError: litellm.RateLimitError: vertex_ai_betaException - b'{\n  "error": {\n    "code": 429,\n    "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head
- **What's the weather in Delhi today?** (run 3): MidStreamFallbackError: litellm.MidStreamFallbackError: litellm.RateLimitError: litellm.RateLimitError: vertex_ai_betaException - b'{\n  "error": {\n    "code": 429,\n    "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head

## Numbers not traceable to a tool output

Usually values the model computed itself (e.g. an implied price) or rounded heavily; occasionally a real error. Review them by hand.

- **Write an investment brief for AAPL** (run 2):
  - 2.75 in: **HOLD**: The average score of 2.75 points to a HOLD rating, reflecting strong underlying financial performance and posi
- **Write an investment brief for JPM** (run 1):
  - -15 in: 2. Valuation: DCF fair value vs. current price: DCF returned “DCF not meaningful” due to strongly negative free-cash-flo
  - 6 in: 6. Bottom line: JPMorgan Chase demonstrates solid operational revenue expansion and strong profitability, but negative f
- **Write an investment brief for JPM** (run 2):
  - -23.6 in: | Valuation     | Overvalued; trading at USD 343.58 against an implied peer-median P/E fair value of approx. USD 262 (-2
  - -23.6 in: 2. **Valuation**: DCF fair value is not meaningful for banks due to negative or unavailable free cash flow (DCF engine r
  - 24 in: 2. **Valuation**: DCF fair value is not meaningful for banks due to negative or unavailable free cash flow (DCF engine r
  - 6 in: 6. **Bottom line**: JPMorgan demonstrates strong fundamental profitability and revenue expansion, but its rich valuation
- **Write an investment brief for JPM** (run 3):
  - 219 in: 2. **Valuation**: DCF fair value is not meaningful due to negative reported free cash flow (-$147,782.0 million in 2025)
  - 6 in: 6. **Bottom line**: JPM demonstrates robust revenue growth and strong profitability metrics, but its valuation sits at a
- **Write an investment brief for RELIANCE.NS** (run 2):
  - 3.75 in: *(Average Score: 3.75)*

## Method

- **Grounding:** every number in the final answer (except years, list numbers and 1-5 scores) is matched against the numbers in that run's tool outputs, allowing 0.5% rounding and unit changes (millions/billions). Approximate: a number can match a fact by coincidence, so treat it as an upper bound.
- **Workflow / routing:** from the orchestrator's tool-call timings, recorded with Strands hooks.
- **Rating rule:** the average of the scored dimensions (N/A excluded): >= 3.75 BUY, <= 2.25 SELL, else HOLD.
