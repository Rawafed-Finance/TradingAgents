from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_news_analyst(llm, toolkit):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [toolkit.get_global_news_openai, toolkit.get_google_news]
        else:
            tools = [
                toolkit.get_finnhub_news,
                toolkit.get_reddit_news,
                toolkit.get_google_news,
            ]

        system_message = (
            "You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Look at news from EODHD, and finnhub to be comprehensive. Do not simply state the trends are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions."
            + """ Make sure to append a Makrdown table at the end of the report to organize key points in the report, organized and easy to read."""
        )

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
                    "For your reference, the current date is {current_date}. We are looking at the company {ticker}",
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
                result = llm.chat(prompt_text)
                return {
                    "messages": [result],
                    "news_report": result,
                }
            result = llm.chat(prompt_text)

        return {
            "messages": [result],
            "news_report": result.content,
        }

    return news_analyst_node

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
