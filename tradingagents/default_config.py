import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "data_dir": "/Users/yluo/Documents/Code/ScAI/FR1-data",
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings
    "deep_think_llm": "o4-mini",
    "quick_think_llm": "gpt-4o-mini",
    "llm_backend": "local",  # 'openai' or 'local'
    "local_model_path": "/Users/ahmed/Desktop/rawafed/TradingAgents/.models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",  # Set this if using local LLM
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Tool settings
    "online_tools": True,
}
