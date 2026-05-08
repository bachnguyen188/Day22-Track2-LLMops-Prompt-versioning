import os
import json
import numpy as np
from dotenv import load_dotenv
from config import OPENAI_API_KEY, OPENAI_BASE_URL, EMBEDDING_MODEL
from qa_pairs import QA_PAIRS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

# RAGAS Imports (Version 0.4.x style)
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

# ── 1. Setup ────────────────────────────────────────────────────────────────
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# ── 2. RAG Logic (Simplified from Step 1) ──────────────────────────────────
def get_retriever():
    text = Path("data/knowledge_base.txt").read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# ── 3. Evaluation Function ──────────────────────────────────────────────────
def run_evaluation(version_name, prompt_template):
    print(f"📊 Evaluating {version_name}...")
    retriever = get_retriever()
    
    samples = []
    for pair in QA_PAIRS:
        question = pair["question"]
        reference = pair["reference"]
        
        # 1. Retrieve
        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        
        # 2. Generate
        chain = prompt_template | llm | StrOutputParser()
        response = chain.invoke({"question": question, "context": "\n\n".join(contexts)})
        
        # 3. Create RAGAS sample
        sample = SingleTurnSample(
            user_input=question,
            response=response,
            retrieved_contexts=contexts,
            reference=reference
        )
        samples.append(sample)
    
    dataset = EvaluationDataset(samples=samples)
    
    # Run RAGAS
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    result = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)
    
    # Compute means (RAGAS 0.4 returns lists)
    summary = {
        "faithfulness": float(np.mean(result["faithfulness"])),
        "answer_relevancy": float(np.mean(result["answer_relevancy"])),
        "context_recall": float(np.mean(result["context_recall"])),
        "context_precision": float(np.mean(result["context_precision"])),
    }
    return summary

def main():
    print("=" * 60)
    print("Step 3: RAGAS Evaluation")
    print("=" * 60)

    # Define Prompts
    PROMPT_V1 = ChatPromptTemplate.from_messages([
        ("system", "Concise answer using context:\n{context}"),
        ("human", "{question}"),
    ])
    
    PROMPT_V2 = ChatPromptTemplate.from_messages([
        ("system", "Provide a detailed explanation based on the context:\n{context}"),
        ("human", "{question}"),
    ])

    results = {}
    results["V1"] = run_evaluation("Prompt V1", PROMPT_V1)
    results["V2"] = run_evaluation("Prompt V2", PROMPT_V2)

    # Print Comparison Table
    print("\n" + "=" * 40)
    print(f"{'Metric':<20} | {'V1':<8} | {'V2':<8}")
    print("-" * 40)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        v1_score = results["V1"][metric]
        v2_score = results["V2"][metric]
        print(f"{metric:<20} | {v1_score:.4f} | {v2_score:.4f}")
    print("=" * 40)

    # Save to file
    with open("data/ragas_report.json", "w") as f:
        json.dump(results, f, indent=4)
    print("✅ Results saved to data/ragas_report.json")

    # Check target
    max_faith = max(results["V1"]["faithfulness"], results["V2"]["faithfulness"])
    if max_faith >= 0.8:
        print(f"✅ Target met! Max Faithfulness: {max_faith:.4f}")
    else:
        print(f"⚠️ Target not met. Max Faithfulness: {max_faith:.4f} (Goal: 0.8)")

if __name__ == "__main__":
    main()
