import os
import hashlib
from dotenv import load_dotenv
from config import LANGCHAIN_API_KEY, LANGCHAIN_PROJECT, OPENAI_API_KEY, OPENAI_BASE_URL
from langsmith import Client, traceable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── 1. Environment setup ────────────────────────────────────────────────────
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT or "Day22-Lab-RAG"

# ── 2. LangSmith Client setup ───────────────────────────────────────────────
client = Client(api_key=LANGCHAIN_API_KEY)

# ── 3. Define two prompt versions ──────────────────────────────────────────
# V1: Concise and direct
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", "You are a concise assistant. Answer the question using ONLY the context: {context}"),
    ("human", "{question}"),
])

# V2: Detailed and structured
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", "You are a professional AI tutor. Provide a detailed, structured explanation based on the context. If the context is insufficient, explain what's missing.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

# ── 4. Push prompts to Hub (Helper) ─────────────────────────────────────────
def push_prompts():
    print("📤 Pushing prompts to LangSmith Prompt Hub...")
    try:
        # Note: In a real scenario, you'd use your username/handle prefix
        client.push_prompt("rag-prompt-v1", object=PROMPT_V1)
        client.push_prompt("rag-prompt-v2", object=PROMPT_V2)
        print("✅ Prompts pushed successfully.")
    except Exception as e:
        print(f"⚠️ Warning during push: {e}")

# ── 5. Deterministic A/B Routing ────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Route 50/50 based on the hash of the request_id.
    """
    h = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return "rag-prompt-v1" if h % 2 == 0 else "rag-prompt-v2"

# ── 6. Main Execution ───────────────────────────────────────────────────────
@traceable(name="ab-routing-query")
def run_ab_test(question: str, request_id: str):
    # a) Determine which version to use
    version_name = get_prompt_version(request_id)
    
    # b) Pull from Hub
    print(f"   [Routing] ID: {request_id} -> Using {version_name}")
    prompt = client.pull_prompt(version_name)
    
    # c) Build temporary chain and run
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    # Dummy context for this demo task
    context = "Machine learning is a subset of AI. RAG stands for Retrieval-Augmented Generation."
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context})
    return version_name, answer

def main():
    print("=" * 60)
    print("Step 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    # First, push the prompts
    push_prompts()

    questions = [
        "What is machine learning?",
        "What does RAG stand for?",
        "Explain deep learning.",
        "What is a vector database?",
    ]

    for i, q in enumerate(questions):
        request_id = f"user-req-{i}"
        version, ans = run_ab_test(q, request_id)
        print(f"Q: {q}")
        print(f"V: {version}")
        print(f"A: {ans[:80]}...\n")

if __name__ == "__main__":
    main()
