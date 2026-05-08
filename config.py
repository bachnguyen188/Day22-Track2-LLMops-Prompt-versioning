import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# LangSmith Config
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true").lower() == "true"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "Day22-Lab-RAG")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# OpenAI Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Models
EVAL_MODEL = os.getenv("EVAL_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

def check_config():
    print("=" * 60)
    print("Checking Configuration...")
    print(f"OK: LangSmith Project : {LANGCHAIN_PROJECT}")
    print(f"OK: OpenAI Endpoint   : {OPENAI_BASE_URL}")
    print(f"OK: Evaluation Model  : {EVAL_MODEL}")
    print(f"OK: Embedding Model   : {EMBEDDING_MODEL}")
    
    if not OPENAI_API_KEY:
        print("ERROR: MISSING: OPENAI_API_KEY")
    if not LANGCHAIN_API_KEY:
        print("ERROR: MISSING: LANGCHAIN_API_KEY")
    print("=" * 60)

if __name__ == "__main__":
    check_config()
