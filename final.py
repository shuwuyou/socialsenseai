import os
import sys
import json
import torch
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone.vectorstores import Pinecone as LangchainPinecone
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Import LangGraph from the installed package (ensure it's installed: pip install langgraph)
from langgraph.graph import Graph

# ====================================================
# Configuration & Environment Setup
# ====================================================
PINECONE_API_KEY = "YOUR-API-KEY"
PINECONE_INDEX_NAME = "text-embedding-index"
OPENAI_API_KEY = "YOUR-API-KEY"

FINETUNED_MODEL_PATH = "t5_lora_summarization"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# ====================================================
# Pinecone and Vectorstore Initialization (Assignment 1)
# ====================================================
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    # "all-MiniLM-L6-v2" produces embeddings of dimension 384.
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east1-gcp")
    )

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = LangchainPinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings, text_key="text")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ====================================================
# Load Fine-Tuned T5 Summarization Model (Assignment 3)
# ====================================================
tokenizer_t5 = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH, use_fast=True)
fine_tuned_model = AutoModelForSeq2SeqLM.from_pretrained(FINETUNED_MODEL_PATH).cuda()
fine_tuned_model.eval()

def generate_t5_summary(text: str) -> str:
    """Generate a summary using the fine-tuned T5 model."""
    inputs = tokenizer_t5(
        f"summarize: {text}",
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(fine_tuned_model.device)
    with torch.no_grad():
        output_ids = fine_tuned_model.generate(**inputs, max_new_tokens=100)
    return tokenizer_t5.decode(output_ids[0], skip_special_tokens=True)

def generate_chat_summary(text: str) -> str:
    """Generate a summary using ChatOpenAI as the base summarizer."""
    prompt = f"Summarize the following text concisely: {text}"
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    response = llm.invoke(prompt)
    return response.content.strip()

# ====================================================
# Define Agent Nodes (Multi-Agent System using LangGraph, Assignment 4)
# ====================================================
def retrieval_node(state: dict) -> dict:
    """Retrieval agent: fetches relevant documents using the Pinecone vectorstore."""
    query = state.get("query", "")
    retrieved = retriever.invoke(query)
    state["retrieved_docs"] = [doc.page_content for doc in retrieved]
    return state

def answer_generation_node(state: dict) -> dict:
    """Answer generation agent: uses ChatOpenAI to produce the final answer."""
    prompt = f"""
You are an AI assistant. Based on the following summarized information and the user's query, generate an accurate and informative answer.

User Query: {state.get("query", "")}
Summary: {state.get("summarized_text", "")}
"""
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    response = llm.invoke(prompt)
    state["answer"] = response.content.strip()
    return state

# ====================================================
# Workflow Functions for Different Approaches
# ====================================================

# 1. Base LLM (No RAG)
def run_base_llm(query: str) -> str:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    response = llm.invoke(query)
    return response.content.strip()

# 2. Basic RAG (Retriever + LLM)
def run_basic_rag(query: str) -> str:
    retrieved = retriever.invoke(query)
    docs = [doc.page_content for doc in retrieved]
    prompt = f"Answer the following query based on the provided documents.\nQuery: {query}\nDocuments:\n" + "\n".join(docs)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    response = llm.invoke(prompt)
    return response.content.strip()

# 3. Advanced Agentic RAG with Base Summarization (using ChatOpenAI summarization)
def build_agent_graph_base() -> Graph:
    graph = Graph()
    graph.add_node("Retrieval", retrieval_node)
    # Summarization node using ChatOpenAI as summarizer
    graph.add_node("Summarization", lambda state: {**state, "summarized_text": generate_chat_summary(" ".join(state.get("retrieved_docs", [])))} )
    graph.add_node("AnswerGeneration", answer_generation_node)
    graph.add_edge("Retrieval", "Summarization")
    graph.add_edge("Summarization", "AnswerGeneration")
    graph.set_entry_point("Retrieval")
    graph.set_finish_point("AnswerGeneration")
    return graph

def run_agentic_rag_base(query: str) -> str:
    initial_state = {"query": query, "retrieved_docs": [], "summarized_text": "", "answer": ""}
    graph = build_agent_graph_base()
    compiled_graph = graph.compile() if hasattr(graph, "compile") else graph
    final_state = compiled_graph.invoke(initial_state)
    return final_state["answer"]

# 4. Advanced Agentic RAG with Fine-Tuned Summarization (using LoRA fine-tuned T5)
def build_agent_graph_finetuned() -> Graph:
    graph = Graph()
    graph.add_node("Retrieval", retrieval_node)
    # Summarization node using the fine-tuned T5 model
    graph.add_node("Summarization", lambda state: {**state, "summarized_text": generate_t5_summary(" ".join(state.get("retrieved_docs", [])))} )
    graph.add_node("AnswerGeneration", answer_generation_node)
    graph.add_edge("Retrieval", "Summarization")
    graph.add_edge("Summarization", "AnswerGeneration")
    graph.set_entry_point("Retrieval")
    graph.set_finish_point("AnswerGeneration")
    return graph

def run_agentic_rag_finetuned(query: str) -> str:
    initial_state = {"query": query, "retrieved_docs": [], "summarized_text": "", "answer": ""}
    graph = build_agent_graph_finetuned()
    compiled_graph = graph.compile() if hasattr(graph, "compile") else graph
    final_state = compiled_graph.invoke(initial_state)
    return final_state["answer"]

# ====================================================
# Evaluation Function (for multiple sample queries)
# ====================================================
def evaluate_system(queries):
    results = {}
    for q in queries:
        results[q] = {
            "Base LLM": run_base_llm(q),
            "Basic RAG": run_basic_rag(q),
            "Agentic RAG (Base Summarization)": run_agentic_rag_base(q),
            "Agentic RAG (Fine-Tuned Summarization)": run_agentic_rag_finetuned(q)
        }
    return results

# ====================================================
# Front-End: Interactive Mode
# ====================================================
if __name__ == "__main__":
    # If command-line argument 'eval' is provided, run evaluation.
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        sample_queries = [
            "What are the benefits of renewable energy?",
            "How does climate change impact agriculture?",
            "Explain the significance of artificial intelligence in modern society.",
            "What are the key challenges in cybersecurity?",
            "How can urban planning improve quality of life?"
        ]
        eval_results = evaluate_system(sample_queries)
        for query, outputs in eval_results.items():
            print(f"\nQuery: {query}")
            for method, answer in outputs.items():
                print(f"{method}:\n{answer}\n")
    else:
        # Interactive mode: User enters a query; automatically generate four answers.
        user_query = input("Enter your query: ")
        print("\nGenerating answers using all four workflows...\n")
        base_llm_answer = run_base_llm(user_query)
        basic_rag_answer = run_basic_rag(user_query)
        agentic_rag_base_answer = run_agentic_rag_base(user_query)
        agentic_rag_finetuned_answer = run_agentic_rag_finetuned(user_query)
        
        print("======================================")
        print("Base LLM (No RAG):")
        print(base_llm_answer)
        print("======================================")
        print("Basic RAG (Retriever + LLM):")
        print(basic_rag_answer)
        print("======================================")
        print("Agentic RAG (Base Summarization):")
        print(agentic_rag_base_answer)
        print("======================================")
        print("Agentic RAG (Fine-Tuned Summarization):")
        print(agentic_rag_finetuned_answer)
