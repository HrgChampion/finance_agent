import logging

import litellm
from litellm.integrations.custom_logger import CustomLogger
from strands.hooks import (
    AfterInvocationEvent,
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
)

log = logging.getLogger("finance_agent")


def setup_logging(log_file, console=True):
    """Log to a file (and the console), each line timestamped to the millisecond."""
    fmt = logging.Formatter("%(asctime)s.%(msecs)03d | %(message)s", datefmt="%H:%M:%S")
    log.setLevel(logging.INFO)
    log.propagate = False
    handlers = [logging.FileHandler(log_file, mode="w", encoding="utf-8")]
    if console:
        handlers.append(logging.StreamHandler())
    for handler in handlers:
        handler.setFormatter(fmt)
        log.addHandler(handler)
    litellm.callbacks = [ModelLogger()]
    # LiteLLM prints full tracebacks even for failures it then recovers from (retry / fallback);
    # ModelLogger already records each failure in one line
    litellm.suppress_debug_info = True
    logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)


def _short(value, limit=60):
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


class AgentLogger(HookProvider):
    """Strands hooks: one log line per agent start/end, tool call and failed model call."""

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(BeforeInvocationEvent, self.agent_start)
        registry.add_callback(AfterInvocationEvent, self.agent_end)
        registry.add_callback(BeforeToolCallEvent, self.tool_start)
        registry.add_callback(AfterToolCallEvent, self.tool_end)
        registry.add_callback(AfterModelCallEvent, self.model_end)

    def agent_start(self, event: BeforeInvocationEvent) -> None:
        log.info(f"[{event.agent.name}] started")

    def agent_end(self, event: AfterInvocationEvent) -> None:
        log.info(f"[{event.agent.name}] finished")

    def tool_start(self, event: BeforeToolCallEvent) -> None:
        log.info(f"[{event.agent.name}] -> {event.tool_use['name']}({_short(event.tool_use['input'])})")

    def tool_end(self, event: AfterToolCallEvent) -> None:
        # our tools return "ERROR: ..." strings instead of raising, so check the text too
        text = "".join(block.get("text", "") for block in event.result.get("content", []))
        failed = event.exception or event.result.get("status") == "error" or text.startswith("ERROR")
        status = f"FAILED: {_short(event.exception or text)}" if failed else "ok"
        log.info(f"[{event.agent.name}] <- {event.tool_use['name']} {status} ({event.duration or 0:.1f}s)")

    def model_end(self, event: AfterModelCallEvent) -> None:
        if event.exception:
            log.info(f"[{event.agent.name}] model call FAILED: {_short(event.exception, 120)}")


class ModelLogger(CustomLogger):
    """LiteLLM callback: shows which model actually answered, so a fallback is visible."""

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        usage = getattr(response_obj, "usage", None)
        tokens = f"{usage.prompt_tokens:,} in / {usage.completion_tokens:,} out" if usage else "tokens n/a"
        seconds = (end_time - start_time).total_seconds()
        log.info(f"    model {kwargs.get('model')} ok | {tokens} | {seconds:.1f}s")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        log.info(f"    model {kwargs.get('model')} FAILED: {_short(kwargs.get('exception'), 120)}")
