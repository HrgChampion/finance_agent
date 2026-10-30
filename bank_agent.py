from strands import Agent ,tool, ModelRetryStrategy
import litellm
from strands.models.litellm import LiteLLMModel
from strands.telemetry.metrics import EventLoopMetrics
from tools import get_income_statement,get_balance_sheet,get_cash_flow_statement,get_peer_multiples,compute_key_ratios,search_recent_news,analyze_sentiment,run_dcf_model,find_ticker,_info
from observability import setup_logging, AgentLogger
from dotenv import load_dotenv
import argparse
from datetime import date, datetime
from pathlib import Path
import re
import sys
import time
import truststore
truststore.inject_into_ssl()


load_dotenv()
flash_lite = LiteLLMModel(model_id="gemini/gemini-3.5-flash-lite", client_args={"num_retries": 3,"fallbacks":["groq/openai/gpt-oss-120b"]})
groq_llama = LiteLLMModel(model_id="groq/openai/gpt-oss-120b", client_args={"num_retries": 3,"fallbacks":["gemini/gemini-3.5-flash-lite"]})

class TransientErrorRetry(ModelRetryStrategy):
    """Strands only retries its own throttling exception. Also retry LiteLLM's rate limits (429),
    overloads (503, including mid-stream failures that LiteLLM's fallbacks cannot catch) and network
    errors, with backoff 4s, 8s, 16s, 32s, 64s."""

    def is_retryable(self, exception: Exception) -> bool:
        return super().is_retryable(exception) or isinstance(exception, (
            litellm.RateLimitError,
            litellm.ServiceUnavailableError,
            litellm.InternalServerError,
            litellm.APIConnectionError,
        ))


AGENT_LOGGER = AgentLogger()
OUTPUT_DIR = Path("briefs")


financial_data_agent= Agent(
    name="FinancialDataAgent",
    description = "Fetches and summarises a company's financial statements (income statement, balance sheet, "
            "cash flow) and key ratios. Input: one or more ticker symbols, e.g. 'AAPL' or 'AAPL, MSFT'.",
    system_prompt="""You are a junior financial analyst. Your job is to build the spreadsheet: accurate numbers, no opinions.

                    For each ticker you are given (there may be several; cover every one, in the same format):
                    1. Retrieve the income statement, balance sheet and cash flow statement using your tools.
                    2. Compute the key ratios using your tools.
                    3. Report the last 3-4 fiscal years.

                    Output format:
                    - Section headings: Income Statement, Balance Sheet, Cash Flow, Key Ratios
                    - Under each: a compact table of the key line items by year (revenue, net income, total debt, cash,
                    free cash flow, margins, ROE, debt/equity, current ratio)
                    - One line at the end: "Trends:" with the 2-3 most notable year-over-year changes

                    Rules:
                    - Use ONLY numbers returned by your tools. Never estimate or invent figures.
                    - If a figure is unavailable, write "N/A". Do not guess.
                    - State the currency and units exactly as the tools report them (e.g. USD millions, INR millions).
                    - No investment opinions, ratings or recommendations. That is not your role.""",
    model= flash_lite,
    tools=[get_income_statement,get_balance_sheet,get_cash_flow_statement,compute_key_ratios],
    callback_handler=None,
    hooks=[AGENT_LOGGER],
    retry_strategy=TransientErrorRetry()
)

news_sentiment_agent =  Agent(
    name="NewsAndSentimentAgent",
    description = "Finds recent news about a company and scores the market sentiment. "
            "Input: one or more ticker symbols, e.g. 'AAPL' or 'AAPL, MSFT'.",
    system_prompt="""You are the news desk. Your job is to report what the market is hearing about a company, and how it feels about it.

                    For each ticker you are given (there may be several; cover every one, in the same format):
                    1. Search for recent news using your tools.
                    2. Analyse the sentiment of the headlines using your tools.

                    Output format:
                    - Top headlines: 5-8 bullet points, each with date, source and a one-line summary
                    - Sentiment per headline: Positive / Neutral / Negative
                    - Overall sentiment: Positive / Neutral / Negative / Mixed, with a score from -1.0 to +1.0
                    - Key themes: 2-3 recurring topics (e.g. earnings, regulation, product launches)

                    Rules:
                    - Every headline must have a date. Ignore undated items.
                    - Report only news returned by your tools. Never invent headlines or events.
                    - Separate facts (what happened) from sentiment (how it is perceived).
                    - If little news is found, say so plainly rather than padding.""",
    model = flash_lite,
    tools=[search_recent_news,analyze_sentiment],
    callback_handler=None,
    hooks=[AGENT_LOGGER],
    retry_strategy=TransientErrorRetry()
)

risk_agent = Agent(
    name = "RiskAgent",
    description = "Identifies and ranks the downside risks of investing in a company: leverage, liquidity "
            "and negative news. Input: one or more ticker symbols, e.g. 'AAPL' or 'AAPL, MSFT'.",
    system_prompt="""You are the risk manager, the sceptic in the room. Your job is to find what could go wrong. Assume the bull case is already covered by others.

                    For each ticker you are given (there may be several; cover every one, in the same format):
                    1. Examine the balance sheet and key ratios using your tools.
                    2. Search recent news for red flags using your tools.

                    Assess:
                    - Leverage: debt/equity, total debt vs. cash
                    - Liquidity: current ratio, quick ratio, ability to cover short-term obligations
                    - Negative news: lawsuits, regulation, management changes, guidance cuts, scandals
                    - Concentration: dependence on one product, customer or market (if evident)

                    Output format:
                    - Risk table: Risk | Evidence (with the actual number or headline) | Severity (High / Medium / Low)
                    - Ranked from most to least severe
                    - Overall risk level: High / Medium / Low, with a one-sentence justification

                    Rules:
                    - Every risk must cite evidence from your tools: a number or a dated headline.
                    - Never invent numbers or events. If data is missing, flag it as a risk in itself ("limited data").
                    - Be sceptical but fair. Do not manufacture risks without evidence.""",
    model = flash_lite,
    tools=[get_balance_sheet,compute_key_ratios,search_recent_news],
    callback_handler=None,
    hooks=[AGENT_LOGGER],
    retry_strategy=TransientErrorRetry()
)

valuation_agent = Agent(
    name="ValuationAgent",
    description = "Estimates a company's fair value per share using a DCF model and peer multiples. "
            "Input: one or more ticker symbols AND the full report from FinancialDataAgent.",
    system_prompt="""You are a valuation specialist. Your job is to answer one question: what is this company worth, and how does that compare to its current price?

                    You will receive one or more tickers and a financial data report. Use that report as your context. Do not re-fetch the statements.

                    For each ticker you are given (there may be several; cover every one, in the same format):
                    1. Run a DCF model using your tools.
                    2. Get peer valuation multiples using your tools.
                    3. Compare the company's multiples to its peers.

                    Output format:
                    - DCF: fair value per share, with the key assumptions (growth rate, discount rate, terminal growth).
                    Call the discount rate "cost of equity (CAPM)", not WACC.
                    - Peer comparison table: Company | P/E | Fwd P/E | EV/EBITDA | P/S
                    - Current share price vs. fair value: upside/downside as a %
                    - Valuation verdict: Undervalued / Fairly valued / Overvalued, with a one-sentence reason

                    Rules:
                    - Always state your assumptions. A valuation without assumptions is meaningless.
                    - Use only numbers from your tools or the financial report you were given. Never invent figures.
                    - If the DCF and the multiples disagree, say so and explain why rather than hiding it.
                    
                    - Call run_dcf_model with DEFAULT assumptions first (pass only the ticker).
                    - Only override an assumption if a specific number in the financial report justifies it,
                    and quote that number. Never use "typical" values from memory.
                    - Report the sensitivity range, not just the single fair value.

                    - If the latest year-over-year revenue growth in the financial report differs from the DCF's
                    default growth by more than 3 percentage points, run a SECOND DCF using that latest growth
                    rate, quote the number that justifies it, and report both fair values side by side.
                    - If run_dcf_model returns an ERROR saying the DCF is not meaningful, do not retry it with other
                    assumptions. Say so plainly and base the verdict on the peer multiples alone.


                    For each company, choose 4-5 peers yourself and pass them to get_peer_multiples. Pick them by
                    industry first (same business model: e.g. never banks as peers for an energy or tech company),
                    then by size, then from the same market where possible (e.g. for AAPL: MSFT, GOOGL, AMZN, META).
                    - Always state the currency of every price and fair value (e.g. USD, INR).
                    - If a tool output contains a NOTE or caveat (e.g. the risk-free rate does not match the
                    company's market), repeat it in your report next to the number it affects.
                    """,
    model = groq_llama,
    tools=[get_peer_multiples,run_dcf_model],
    callback_handler=None,
    hooks=[AGENT_LOGGER],
    retry_strategy=TransientErrorRetry()
)

report_writer_agent = Agent(
    name="ReportWriterAgent",
    description = "Synthesises the four specialist reports into the final investment brief with a rating. "
            "Input: the FULL text of all four reports (financial data, news & sentiment, risk, valuation) "
            "for ONE company. For several companies, call it once per company, one after another.",
    system_prompt="""You are the managing director. You do not research. You read your team's four reports and make the call.

                    You will receive four reports: Financial Data, News & Sentiment, Risk, and Valuation.

                    First, build a rating table:
                    | Dimension     | Assessment | Score (1-5) |
                    | Financials    | ...        | ...         |
                    | Valuation     | ...        | ...         |
                    | Risk          | ...        | ...         |  (5 = lowest risk)
                    | Sentiment     | ...        | ...         |

                    Then write the investment brief:
                    1. Rating: BUY / HOLD / SELL, in bold, with a one-sentence rationale
                    2. Valuation: DCF fair value vs. current price, WITH the growth rate and discount rate used
                        and the sensitivity range; plus the P/E premium or discount vs. the peer median.
                    3. Investment thesis: 3 bullet points, the bull case
                    4. Key risks: the top 3 from the Risk report
                    5. Sentiment: one paragraph on market mood
                    6. Bottom line: two sentences

                    Rules:
                    - Use ONLY information in the four reports. Never invent numbers, news or facts.
                    - If the reports contradict each other, name the contradiction and explain how you weighed it.
                    - If a report is missing or says data was unavailable, say so in the brief and lower your confidence.
                    - Keep the brief under 500 words. Write for a busy portfolio manager.
                    - Quote numbers and ranges exactly as the reports state them. Do not round or paraphrase.
                    - Derive the rating from the average of the scored dimensions (exclude N/A):
                     average >= 3.75 -> BUY, average <= 2.25 -> SELL, otherwise HOLD.
                    - Show the average under the rating table. The rationale must be consistent with the rating.
                    - Every risk title and thesis bullet must be directly supported by its evidence. Do not generalise beyond it.


                    
                    If a report is unavailable, score that dimension N/A and exclude it, rather than giving it a low score.
                    """,
    model = flash_lite,
    tools=[],
    callback_handler=None,
    hooks=[AGENT_LOGGER],
    retry_strategy=TransientErrorRetry()
)

SPECIALISTS = [financial_data_agent, news_sentiment_agent, risk_agent, valuation_agent, report_writer_agent]


def reset_metrics(agents):
    """Start fresh counters, so each question's summary covers only that question."""
    for a in agents:
        a.event_loop_metrics = EventLoopMetrics()


def run_summary(agents, wall_seconds):
    lines = [
        "| Agent | Model | Cycles | Tokens in | Tokens out | Time (s) | Tool calls |",
        "|---|---|---|---|---|---|---|",
    ]
    total_in = total_out = 0
    for a in agents:
        m = a.event_loop_metrics
        if not m.cycle_count:
            continue                                  # agent not used for this question
        usage = m.accumulated_usage
        tools = ", ".join(
            f"{name} x{t.call_count}" + (f" ({t.error_count} err)" if t.error_count else "")
            for name, t in m.tool_metrics.items()
        ) or "-"
        model_id = a.model.get_config().get("model_id", "?")
        lines.append(
            f"| {a.name} | {model_id} | {m.cycle_count} | {usage['inputTokens']:,} | "
            f"{usage['outputTokens']:,} | {sum(m.cycle_durations):.1f} | {tools} |"
        )
        total_in += usage["inputTokens"]
        total_out += usage["outputTokens"]
    lines.append(f"\nTotal tokens: {total_in:,} in / {total_out:,} out | Wall time: {wall_seconds:.1f}s")
    return "\n".join(lines)

def orchestrator_agent():

    prompt = f"""You are the director of research at an investment firm. Users ask you questions about listed
                companies and their stocks. You answer by delegating to your specialists, then either returning their
                brief or writing the answer yourself from their reports. Today is {date.today()}.

                Your team:
                - find_ticker: turns a company name into its ticker symbol
                - FinancialDataAgent: financial statements, key ratios, trends
                - NewsAndSentimentAgent: recent headlines and market sentiment
                - RiskAgent: downside risks (leverage, liquidity, negative news)
                - ValuationAgent: DCF fair value and peer multiples; needs the FinancialDataAgent report
                - ReportWriterAgent: full investment brief with a BUY / HOLD / SELL rating; needs all four reports

                HOW TO WORK
                1. Identify the companies. If the user names a company rather than a ticker, use find_ticker.
                   Prefer the primary listing (e.g. NASDAQ/NYSE for US companies, .NS for Indian ones) and say
                   which ticker you used. If the question refers to an earlier company ("what about its risks?"),
                   take it from the conversation.
                2. Call only the specialists the question needs. Examples:
                   - "What is Apple's debt to equity?"           -> FinancialDataAgent
                   - "Any bad news about Tesla?"                 -> NewsAndSentimentAgent and RiskAgent
                   - "Is MSFT overvalued?"                       -> FinancialDataAgent, then ValuationAgent
                   - "Compare the margins of AAPL and MSFT"      -> FinancialDataAgent with "AAPL, MSFT"
                   - "Write a brief on X" / "Should I buy X?" / "Rate X" -> the FULL WORKFLOW below
                3. Call independent specialists together, in a single response, so they run in parallel.
                   ValuationAgent always comes after FinancialDataAgent and receives its full report.
                4. Several companies: pass all tickers in ONE call to each specialist (e.g. "AAPL, MSFT").
                   Never call the same specialist twice at the same time.

                FULL WORKFLOW (briefs, ratings, buy / sell / hold questions)
                STEP 1 - FinancialDataAgent, NewsAndSentimentAgent and RiskAgent, in parallel.
                STEP 2 - ValuationAgent with the tickers AND the full FinancialDataAgent report.
                STEP 3 - ReportWriterAgent with the FULL text of all four reports, clearly labelled
                         "FINANCIAL DATA REPORT:", "NEWS & SENTIMENT REPORT:", "RISK REPORT:", "VALUATION REPORT:".
                         One call per company, one after another.
                STEP 4 - Return the ReportWriterAgent's brief(s) WORD FOR WORD, with no commentary.

                ANSWERING ALL OTHER QUESTIONS
                - Answer the question in the first sentence, then give the supporting numbers. If the evidence
                  points in different directions (e.g. DCF says undervalued, multiples say expensive), say so in
                  that first sentence instead of picking one side.
                  Use a table when comparing companies.
                - Use ONLY facts and numbers from your specialists' reports. Never invent or recall numbers.
                - Always state the currency (e.g. USD, INR) and the fiscal year of the numbers.
                - Pass on any caveat a specialist flags about its numbers (e.g. a DCF assumption that may not
                  fit the company's market), and let it temper how firmly you state the conclusion.
                - Do not give a BUY / HOLD / SELL rating unless you ran the full workflow.
                - Keep it short: a busy portfolio manager is reading.

                RULES
                - Pass reports in full when handing them to another specialist. Do not summarise them.
                - If a specialist or tool fails, continue with what you have and say what is missing.
                - Scope: listed companies, their financials, news, risks and valuation. For anything else
                  (personal tax, crypto price predictions, unrelated topics), reply in one sentence that it is
                  outside your scope, without calling any specialist."""

    agent = Agent(
        name ="OrchestratorAgent",
        system_prompt=prompt,
        model=flash_lite,
        tools=[
            find_ticker,
            financial_data_agent.as_tool(),
            news_sentiment_agent.as_tool(),
            risk_agent.as_tool(),
            valuation_agent.as_tool(),
            report_writer_agent.as_tool()
        ],
        callback_handler=None,
        hooks=[AGENT_LOGGER],
        retry_strategy=TransientErrorRetry()
    )

    return agent

def is_ticker(text):
    """True if text is a single valid ticker symbol, e.g. 'AAPL' or 'RELIANCE.NS'."""
    if not re.fullmatch(r"[A-Za-z0-9.\-^]{1,12}", text):
        return False
    try:
        return bool(_info(text.upper()).get("marketCap"))
    except Exception:
        return False


def answer(orchestrator, question):
    """Run one question through the agents; save and print the answer with its metrics."""
    agents = [orchestrator] + SPECIALISTS
    reset_metrics(agents)
    start = time.perf_counter()
    try:
        result = str(orchestrator(question))
    except Exception as e:
        print(f"\nSorry, that failed: {e}")
        return
    summary = run_summary(agents, time.perf_counter() - start)

    # save before printing, so a console problem can never lose the answer
    slug = re.sub(r"[^A-Za-z0-9]+", "_", question).strip("_")[:50]
    out = OUTPUT_DIR / f"{datetime.now():%Y-%m-%d_%H%M%S}_{slug}.md"
    out.write_text(f"**Question:** {question}\n\n{result}", encoding="utf-8")
    out.with_suffix(".metrics.md").write_text(summary, encoding="utf-8")

    print(f"\n{result}\n\n{summary}\n\nSaved to {out}")


def main():
    # Windows consoles default to cp1252, which cannot print characters like "≈" in the brief
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Ask the finance research agents anything about listed companies.",
        epilog='Examples:  python bank_agent.py AAPL   |   python bank_agent.py "Compare AAPL and MSFT margins"   '
               '|   python bank_agent.py   (interactive)')
    parser.add_argument("question", nargs="*", help="a question, or just a ticker for a full investment brief")
    question = " ".join(parser.parse_args().question).strip()

    OUTPUT_DIR.mkdir(exist_ok=True)
    setup_logging(OUTPUT_DIR / f"{datetime.now():%Y-%m-%d_%H%M%S}_session.log")
    orchestrator = orchestrator_agent()

    if question:
        if is_ticker(question):                      # a bare ticker keeps the original behaviour
            question = f"Write an investment brief for {question.upper()}"
        answer(orchestrator, question)
        return

    # interactive: one conversation, so follow-ups like "what about its risks?" work
    print("Finance research assistant. Ask about any listed company. Type 'exit' to quit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            answer(orchestrator, question)



if __name__ == "__main__":
    main()
