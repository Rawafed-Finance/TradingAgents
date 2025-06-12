from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
import os


def create_market_analyst(llm, toolkit):

    def truncate_prompt(prompt_text, max_tokens=512):
        max_chars = max_tokens * 4
        return prompt_text[:max_chars]

    def truncate_prompt_to_tokens(llm, prompt_text, max_tokens=None):
        # Use the model's context length if available, default to 8192
        if max_tokens is None:
            max_tokens = getattr(llm.llm, 'context_length', 8192)
        tokens = llm.llm.tokenize(bytes(prompt_text, "utf-8"), add_bos=True)
        if len(tokens) <= max_tokens:
            return prompt_text
        for cut in range(len(prompt_text), 0, -100):
            candidate = prompt_text[:cut]
            tokens = llm.llm.tokenize(bytes(candidate, "utf-8"), add_bos=True)
            if len(tokens) <= max_tokens:
                return candidate
        return prompt_text[:max_tokens * 4]

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [
                toolkit.get_YFin_data_online,
                toolkit.get_stockstats_indicators_report_online,
            ]
        else:
            tools = [
                toolkit.get_YFin_data,
                toolkit.get_stockstats_indicators_report,
            ]

        system_message = (
            """You are a trading assistant tasked with analyzing financial markets. Your role is to select the **most relevant indicators** for a given market condition or trading strategy from the following list. The goal is to choose up to **8 indicators** that provide complementary insights without redundancy. Categories and each category's indicators are:

Moving Averages:
- close_50_sma: 50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.
- close_200_sma: 200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.
- close_10_ema: 10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.

MACD Related:
- macd: MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.
- macds: MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.
- macdh: MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.

Momentum Indicators:
- rsi: RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.

Volatility Indicators:
- boll: Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.
- boll_ub: Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.
- boll_lb: Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.
- atr: ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.

Volume-Based Indicators:
- vwma: VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses.

- Select indicators that provide diverse and complementary information. Avoid redundancy (e.g., do not select both rsi and stochrsi). Also briefly explain why they are suitable for the given market context. When you tool call, please use the exact name of the indicators provided above as they are defined parameters, otherwise your call will fail. Please make sure to call get_YFin_data first to retrieve the CSV that is needed to generate indicators. Write a very detailed and nuanced report of the trends you observe. Do not simply state the trends are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions."""
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
        )

        # --- Enhanced Local LLM Path: Simulate Tool Calling ---
        if toolkit.config.get("llm_backend") == "local":
            # 1. Pre-fetch tool outputs
            # Always call get_YFin_data first (required for indicators)
            if toolkit.config["online_tools"]:
                yfin_data = toolkit.get_YFin_data_online.func(
                    symbol=ticker, start_date=current_date, end_date=current_date
                )
                indicators = ["close_50_sma", "macd", "rsi"]  # Only 3 most relevant
                indicator_outputs = {}
                for ind in indicators:
                    try:
                        indicator_outputs[ind] = toolkit.get_stockstats_indicators_report_online.func(
                            symbol=ticker, indicator=ind, curr_date=current_date, look_back_days=30
                        )
                    except Exception as e:
                        indicator_outputs[ind] = f"[Error fetching {ind}: {e}]"
            else:
                yfin_data = toolkit.get_YFin_data.func(
                    symbol=ticker, start_date=current_date, end_date=current_date
                )
                indicators = ["close_50_sma", "macd", "rsi"]
                indicator_outputs = {}
                for ind in indicators:
                    try:
                        indicator_outputs[ind] = toolkit.get_stockstats_indicators_report.func(
                            symbol=ticker, indicator=ind, curr_date=current_date, look_back_days=30
                        )
                    except Exception as e:
                        indicator_outputs[ind] = f"[Error fetching {ind}: {e}]"

            # 2. Summarize tool outputs (first 10 lines for yfin, 5 for indicators)
            def summarize_output(text, n):
                lines = text.splitlines()
                return "\n".join(lines[:n]) + ("\n..." if len(lines) > n else "")
            yfin_data_summary = summarize_output(str(yfin_data), 10)
            indicator_outputs_summary = {k: summarize_output(str(v), 5) for k, v in indicator_outputs.items()}

            # 3. Shorten system instructions
            short_system_message = (
                f"You are a market analyst. Use the provided data and indicators to write a concise, actionable market report for {ticker} on {current_date}. Focus on insights and recommendations. Do not repeat instructions."
            )
            # 4. Build a structured prompt
            prompt_sections = [
                "### SYSTEM INSTRUCTIONS\n" + short_system_message,
                f"\n### TOOL OUTPUT: get_YFin_data (summary)\n{yfin_data_summary}",
            ]
            for ind in indicators:
                prompt_sections.append(f"\n### TOOL OUTPUT: get_stockstats_indicators_report ({ind}, summary)\n{indicator_outputs_summary[ind]}")
            # Add user messages (if any)
            if state.get("messages"):
                user_msgs = state["messages"]
                try:
                    user_msgs_str = "\n".join([
                        m.content if hasattr(m, "content") else str(m) for m in user_msgs
                    ])
                except Exception:
                    user_msgs_str = str(user_msgs)
                prompt_sections.append("\n### USER MESSAGES\n" + user_msgs_str)
            # 5. Add explicit synthesis instruction
            prompt_sections.append(f"\n### TASK\nBased on the above data, write a concise, actionable market report for {ticker} on {current_date}. Do not repeat instructions. Focus on insights and recommendations.")
            prompt_text = "\n".join(prompt_sections)
            # Always truncate to model's context window before sending to LLM
            max_ctx = getattr(llm.llm, 'n_ctx', None)
            if callable(max_ctx):
                max_ctx = max_ctx()
            if max_ctx is None:
                max_ctx = getattr(llm.llm, 'context_length', 4096)
            if callable(max_ctx):
                max_ctx = max_ctx()
            max_ctx = int(max_ctx)
            prompt_text = truncate_prompt_to_tokens(llm, prompt_text, max_tokens=max_ctx)
            result = llm.chat(prompt_text)
            # Save report and metadata
            output_dir = os.path.join("output_reports", ticker)
            os.makedirs(output_dir, exist_ok=True)
            meta = {
                "symbol": ticker,
                "date": current_date,
                "model": getattr(llm.llm, 'model_path', str(llm.llm)),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "llm_backend": toolkit.config.get("llm_backend"),
                "prompt_truncated": len(prompt_text),
            }
            report_data = {
                "meta": meta,
                "report": result,
            }
            out_path = os.path.join(output_dir, f"market_report_{current_date}.json")
            with open(out_path, "w") as f:
                json.dump(report_data, f, indent=2)
            return {
                "messages": [result],
                "market_report": result,
            }

        # --- OpenAI/Function Calling Path (unchanged) ---
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. The company we want to look at is {ticker}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        if hasattr(llm, "bind_tools"):
            chain = prompt | llm.bind_tools(tools)
            result = chain.invoke(state["messages"])
        else:
            # LocalLLM fallback: just use chat
            prompt_text = prompt.format(messages=state["messages"])
            if toolkit.config.get("llm_backend") == "local":
                prompt_text = truncate_prompt_to_tokens(llm, prompt_text)
            result = llm.chat(prompt_text)
            # If using LocalLLM, result is a string, not an object with .content
            if toolkit.config.get("llm_backend") == "local":
                return {
                    "messages": [result],
                    "market_report": result,
                }

        return {
            "messages": [result],
            "market_report": result.content,
        }

    return market_analyst_node

# Add .gitignore for output_reports
if not os.path.exists(".gitignore"):
    with open(".gitignore", "w") as f:
        f.write("output_reports/\n")
else:
    with open(".gitignore", "r+") as f:
        lines = f.readlines()
        if "output_reports/\n" not in lines:
            f.write("output_reports/\n")
