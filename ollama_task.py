import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# ------------ LangChain ------------
from langchain.docstore.document import Document as LCDocument
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# ------------ LlamaIndex ------------
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
    Document as LIDocument,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# =========================================================
# Groq
# =========================================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
LLM_MODEL = "llama-3.1-8b-instant"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# =========================================================
# LlamaIndex: global settings (Groq via OpenAI-compatible endpoint)
# =========================================================
Settings.llm = OpenAILike(
    model=LLM_MODEL,
    api_base=GROQ_BASE_URL,
    api_key=GROQ_API_KEY,
    is_chat_model=True,
    temperature=0,
)
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
Settings.node_parser = SentenceSplitter(chunk_size=400, chunk_overlap=60)


def load_csv_docs(path, *, framework, content_cols=("title", "text")):
    df = pd.read_csv(path).fillna("")
    text_series = df[list(content_cols)].agg("\n".join, axis=1)

    meta_cols = [c for c in df.columns if c not in content_cols]
    metas = df[meta_cols].to_dict(orient="records")

    if framework == "langchain":
        return [
            LCDocument(page_content=t, metadata=m) for t, m in zip(text_series, metas)
        ]
    elif framework == "llamaindex":
        return [LIDocument(text=t, metadata=m) for t, m in zip(text_series, metas)]
    else:
        raise ValueError("framework must be 'langchain' or 'llamaindex'")


def LangChain(chunks, *, question: str):
    """
    LangChain: manual RAG (retrieve -> custom prompt -> chat).
    """
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if os.path.exists("chroma_db"):
        vectordb = Chroma(persist_directory="chroma_db", embedding_function=emb)
    else:
        vectordb = Chroma.from_documents(chunks, emb, persist_directory="chroma_db")
        vectordb.persist()

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    context_docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(d.page_content for d in context_docs)

    prompt = f"""Use ONLY the Context below to answer.
                If the answer isn't in the context, say you don't know.

                Question: {question}

                Context:
                {context}
                """

    llm = ChatGroq(model=LLM_MODEL, temperature=0, groq_api_key=GROQ_API_KEY)
    response = llm.invoke(prompt)

    print("\n=== LangChain (manual) ===")
    print(response.content)


def LangChain2(chunks, *, question: str):
    """
    LangChain: RetrievalQA (stuff).
    """
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if os.path.exists("chroma_db"):
        vectordb = Chroma(persist_directory="chroma_db", embedding_function=emb)
    else:
        vectordb = Chroma.from_documents(chunks, emb, persist_directory="chroma_db")
        vectordb.persist()

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(model=LLM_MODEL, temperature=0, groq_api_key=GROQ_API_KEY)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )
    result = qa.invoke({"query": question})

    print("\n=== LangChain (RetrievalQA) ===")
    print(result["result"])


def LlamaIndex(*, question):
    store_path = "llamaindex_storage"
    if os.path.exists(store_path):
        sc = StorageContext.from_defaults(persist_dir=store_path)
        index = load_index_from_storage(sc)
    else:
        data = load_csv_docs("historical_100_long.csv", framework="llamaindex")
        index = VectorStoreIndex.from_documents(data)
        index.storage_context.persist(persist_dir=store_path)

    engine = index.as_query_engine(similarity_top_k=4, response_mode="compact")
    response = engine.query(question)

    print("\n=== LlamaIndex ===")
    print(response)


if __name__ == "__main__":
    question = input("Insert your prompt: ")

    data = load_csv_docs("historical_100_long.csv", framework="langchain")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    # sanitize to avoid empty/whitespace embedding errors
    chunks = [
        d
        for d in splitter.split_documents(data)
        if d.page_content and d.page_content.strip()
    ]

    LangChain(chunks, question=question)
    LangChain2(chunks, question=question)
    LlamaIndex(question=question)
