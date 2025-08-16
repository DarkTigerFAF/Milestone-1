import os
import asyncio
import pandas as pd
from dotenv import load_dotenv
import logging

# ------------ LangChain ------------
from langchain.docstore.document import Document as LCDocument
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
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

# ------------ Semantic Kernel ------------
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.hugging_face import HuggingFaceTextEmbedding
from semantic_kernel.functions import kernel_function, KernelArguments

# =========================================================
# ENV + MODELS
# =========================================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.1-8b-instant"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# =========================================================
# LlamaIndex global settings
# =========================================================
Settings.llm = OpenAILike(
    model=LLM_MODEL,
    api_base="https://api.groq.com/openai/v1",  # Groq endpoint (OpenAI-compatible)
    api_key=GROQ_API_KEY,
    is_chat_model=True,
    temperature=0,
)
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
Settings.node_parser = SentenceSplitter(chunk_size=400, chunk_overlap=60)


# =========================================================
# CSV Loader
# =========================================================
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


# =========================================================
# LangChain + LlamaIndex pipelines
# =========================================================
def LangChain(chunks, *, question: str) -> str:
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = (
        Chroma.from_documents(chunks, emb, persist_directory="chroma_db")
        if not os.path.exists("chroma_db")
        else Chroma(persist_directory="chroma_db", embedding_function=emb)
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    context_docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in context_docs)

    prompt = f"""Use ONLY the Context below to answer.
    If the answer isn't in the context, say you don't know.

    Question: {question}

    Context:
    {context}
    """

    llm = ChatGroq(model=LLM_MODEL, temperature=0, groq_api_key=GROQ_API_KEY)
    return llm.invoke(prompt).content


def LangChain2(chunks, *, question: str) -> str:
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = (
        Chroma.from_documents(chunks, emb, persist_directory="chroma_db")
        if not os.path.exists("chroma_db")
        else Chroma(persist_directory="chroma_db", embedding_function=emb)
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = ChatGroq(model=LLM_MODEL, temperature=0, groq_api_key=GROQ_API_KEY)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa.invoke({"query": question})["result"]


def LlamaIndex(*, question: str) -> str:
    store_path = "llamaindex_storage"
    if os.path.exists(store_path):
        sc = StorageContext.from_defaults(persist_dir=store_path)
        index = load_index_from_storage(sc)
    else:
        data = load_csv_docs("historical_100_long.csv", framework="llamaindex")
        index = VectorStoreIndex.from_documents(data)
        index.storage_context.persist(persist_dir=store_path)

    engine = index.as_query_engine(similarity_top_k=4, response_mode="compact")
    return str(engine.query(question))


# =========================================================
# Wrap functions as SK Native Functions
# =========================================================
class RAGPlugin:
    def __init__(self, chunks):
        self.chunks = chunks

    @kernel_function(
        name="LangChainManual", description="Run LangChain manual RAG pipeline"
    )
    def langchain_manual(self, question: str) -> str:
        return LangChain(self.chunks, question=question)

    @kernel_function(
        name="LangChainQA", description="Run LangChain RetrievalQA pipeline"
    )
    def langchain_qa(self, question: str) -> str:
        return LangChain2(self.chunks, question=question)

    @kernel_function(name="LlamaIndexRAG", description="Run LlamaIndex RAG pipeline")
    def llamaindex_rag(self, question: str) -> str:
        return LlamaIndex(question=question)


# =========================================================
# MAIN (async-first for SK v1.x)
# =========================================================
async def main():
    question = input("Insert your prompt: ")

    # Preprocess data for LangChain
    data = load_csv_docs("historical_100_long.csv", framework="langchain")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = [d for d in splitter.split_documents(data) if d.page_content.strip()]

    kernel = Kernel()

    # Register Groq LLM (OpenAI-compatible)
    kernel.add_service(
        OpenAIChatCompletion(
            service_id="groq_llm",
            ai_model_id=LLM_MODEL,
            api_key=GROQ_API_KEY,
        )
    )

    # Register HuggingFace embeddings
    kernel.add_service(
        HuggingFaceTextEmbedding(service_id="hf_embed", ai_model_id=EMBED_MODEL)
    )

    # Register our RAG pipelines as a plugin
    rag_plugin = RAGPlugin(chunks)
    kernel.add_plugin(rag_plugin, "rag")

    print("\n=== Semantic Kernel: LangChain Manual ===")
    result1 = await kernel.invoke(
        kernel.plugins["rag"]["LangChainManual"], KernelArguments(question=question)
    )
    print(result1)

    print("\n=== Semantic Kernel: LangChain RetrievalQA ===")
    result2 = await kernel.invoke(
        kernel.plugins["rag"]["LangChainQA"], KernelArguments(question=question)
    )
    print(result2)

    print("\n=== Semantic Kernel: LlamaIndex ===")
    result3 = await kernel.invoke(
        kernel.plugins["rag"]["LlamaIndexRAG"], KernelArguments(question=question)
    )
    print(result3)


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    asyncio.run(main())
