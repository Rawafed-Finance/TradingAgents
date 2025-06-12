# TradingAgents/graph/signal_processing.py

from tradingagents.llm_interface import BaseLLM


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm: BaseLLM):
        """Initialize with an LLM for processing."""
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """
        Process a full trading signal to extract the core decision.

        Args:
            full_signal: Complete trading signal text

        Returns:
            Extracted decision (BUY, SELL, or HOLD)
        """
        messages = [
            (
                "system",
                "You are an efficient assistant designed to analyze paragraphs or financial reports provided by a group of analysts. Your task is to extract the investment decision: SELL, BUY, or HOLD. Provide only the extracted decision (SELL, BUY, or HOLD) as your output, without adding any additional text or information.",
            ),
            ("human", full_signal),
        ]

        llm = self.quick_thinking_llm
        # If local LLM, flatten messages to a single string
        if hasattr(llm, "llm"):  # crude check for LocalLLM
            prompt_text = "\n".join([m[1] for m in messages])
            result = llm.chat(prompt_text)
            return result if isinstance(result, str) else result.content
        else:
            return llm.chat(messages).content
