"""Offline tests for the non-LLM parts of bank_agent.py: retry rules, ticker detection,
and the run summary. No model is called."""
from types import SimpleNamespace

import litellm
import pytest
from litellm.exceptions import MidStreamFallbackError

import bank_agent


@pytest.fixture
def retry():
    return bank_agent.TransientErrorRetry()


def test_retries_rate_limits_and_overloads(retry):
    rate_limit = litellm.RateLimitError("429", llm_provider="gemini", model="m")
    assert retry.is_retryable(rate_limit)
    assert retry.is_retryable(litellm.ServiceUnavailableError("503", llm_provider="gemini", model="m"))
    # a 429 that arrives mid-stream, which LiteLLM's own fallbacks cannot catch
    assert retry.is_retryable(MidStreamFallbackError("m", model="m", llm_provider="gemini",
                                                     original_exception=rate_limit))


def test_does_not_retry_real_errors(retry):
    assert not retry.is_retryable(litellm.BadRequestError("400", model="m", llm_provider="gemini"))
    assert not retry.is_retryable(ValueError("bug"))


def test_every_agent_has_its_own_retry_strategy():
    agents = [bank_agent.orchestrator_agent()] + bank_agent.SPECIALISTS
    strategies = [a._retry_strategy for a in agents]
    assert all(isinstance(s, bank_agent.TransientErrorRetry) for s in strategies)
    assert len({id(s) for s in strategies}) == len(agents)        # retry state is per agent


def test_new_orchestrator_does_not_inherit_specialist_history():
    old_run = [{"role": "user", "content": [{"text": "AAPL"}]}]
    for specialist in bank_agent.SPECIALISTS:
        specialist.messages = list(old_run)              # as if a previous run had used them
    orchestrator = bank_agent.orchestrator_agent()
    assert all(s.messages == [] for s in bank_agent.SPECIALISTS)
    wrappers = [orchestrator.tool_registry.registry[s.name] for s in bank_agent.SPECIALISTS]
    assert all(w._initial_messages == [] for w in wrappers)   # the "fresh" state each call restores


@pytest.mark.parametrize("text",["Compare AAPL and MSFT", "what is apple's P/E?", ""])
def test_is_ticker_rejects_questions_without_a_lookup(monkeypatch, text):
    monkeypatch.setattr(bank_agent, "_info", lambda t: pytest.fail("should not look up a sentence"))
    assert not bank_agent.is_ticker(text)


def test_is_ticker_checks_the_symbol_exists(monkeypatch):
    known = {"AAPL": {"marketCap": 1}, "RELIANCE.NS": {"marketCap": 1}}
    monkeypatch.setattr(bank_agent, "_info", lambda t: known.get(t, {}))
    assert bank_agent.is_ticker("aapl")
    assert bank_agent.is_ticker("reliance.ns")
    assert not bank_agent.is_ticker("APPLE")


def fake_agent(name, cycles, tokens_in=0, tokens_out=0, tools=None):
    metrics = SimpleNamespace(
        cycle_count=cycles, cycle_durations=[1.5] * cycles, tool_metrics=tools or {},
        accumulated_usage={"inputTokens": tokens_in, "outputTokens": tokens_out},
    )
    model = SimpleNamespace(get_config=lambda: {"model_id": "gemini/test"})
    return SimpleNamespace(name=name, event_loop_metrics=metrics, model=model)


def test_run_summary_skips_unused_agents_and_totals_tokens():
    tool = SimpleNamespace(call_count=2, error_count=1)
    agents = [
        fake_agent("OrchestratorAgent", 2, 1000, 100, {"RiskAgent": tool}),
        fake_agent("RiskAgent", 1, 500, 50),
        fake_agent("ValuationAgent", 0),
    ]
    summary = bank_agent.run_summary(agents, wall_seconds=12.34)
    assert "| OrchestratorAgent | gemini/test | 2 | 1,000 | 100 | 3.0 | RiskAgent x2 (1 err) |" in summary
    assert "ValuationAgent" not in summary
    assert "Total tokens: 1,500 in / 150 out | Wall time: 12.3s" in summary
