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
from flask import Flask, request, render_template_string

# Import LangGraph (ensure it's installed: pip install langgraph)
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
    """Generate a summary using ChatOpenAI as a base summarizer."""
    prompt = f"Summarize the following text concisely: {text}"
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    response = llm.invoke(prompt)
    return response.content.strip()

# ====================================================
# Define Agent Nodes (Multi-Agent System via LangGraph, Assignment 4)
# ====================================================
def retrieval_node(state: dict) -> dict:
    """Retrieval agent: retrieves relevant documents using the Pinecone vectorstore."""
    query = state.get("query", "")
    retrieved = retriever.invoke(query)
    state["retrieved_docs"] = [doc.page_content for doc in retrieved]
    return state

def answer_generation_node(state: dict) -> dict:
    """Answer generation agent: uses ChatOpenAI to produce the final answer based on the query and summary."""
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

# 1. Base LLM (No RAG): Directly query ChatOpenAI.
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
# Flask Web App Front-End
# ====================================================
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Advanced Multi-Agent RAG System</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        input[type="text"] { width: 80%; padding: 8px; margin: 10px 0; }
        input[type="submit"] { padding: 8px 16px; }
        .answer { border: 1px solid #ccc; padding: 10px; margin: 10px 0; }
        h2 { color: #2c3e50; }
    </style>
</head>
<body>
    <h1>Advanced Multi-Agent RAG System</h1>
    <form method="POST">
        <label for="query">Enter your query:</label><br>
        <input type="text" id="query" name="query" value="{{ query | default('') }}" required><br>
        <input type="submit" value="Submit">
    </form>
    {% if answers %}
    <h2>Results</h2>
    <div class="answer">
        <h3>Base LLM (No RAG):</h3>
        <p>{{ answers["Base LLM"] }}</p>
    </div>
    <div class="answer">
        <h3>Basic RAG (Retriever + LLM):</h3>
        <p>{{ answers["Basic RAG"] }}</p>
    </div>
    <div class="answer">
        <h3>Agentic RAG (Base Summarization):</h3>
        <p>{{ answers["Agentic RAG (Base Summarization)"] }}</p>
    </div>
    <div class="answer">
        <h3>Agentic RAG (Fine-Tuned Summarization):</h3>
        <p>{{ answers["Agentic RAG (Fine-Tuned Summarization)"] }}</p>
    </div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["query"]
        # Automatically generate answers from all four workflows.
        answers = {
            "Base LLM": run_base_llm(query),
            "Basic RAG": run_basic_rag(query),
            "Agentic RAG (Base Summarization)": run_agentic_rag_base(query),
            "Agentic RAG (Fine-Tuned Summarization)": run_agentic_rag_finetuned(query)
        }
        return render_template_string(HTML_TEMPLATE, query=query, answers=answers)
    return render_template_string(HTML_TEMPLATE, query="", answers=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
