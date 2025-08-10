import os
import pandas as pd
import ollama, json
from langchain.docstore.document import Document as LCDocument
from llama_index.core import Document as LIDocument
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
    Document as LIDocument,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.langchain import LangchainEmbedding
from pathlib import Path

Settings.llm = Ollama(model="llama3.1")
Settings.embed_model = LangchainEmbedding(OllamaEmbeddings(model="nomic-embed-text"))

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


def LangChain(chunks, *, question: str):
    """
    LangChain the manual augment way
    """

    emb = OllamaEmbeddings(model="nomic-embed-text")
    if os.path.exists("chroma_db"):
        vectordb = Chroma(persist_directory="chroma_db", embedding_function=emb)
    else:
        vectordb = Chroma.from_documents(chunks, emb, persist_directory="chroma_db")
        vectordb.persist()

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    context_docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(d.page_content for d in context_docs)

    prompt = f"""
                Use ONLY the Context below to answer.
                If the answer isn't in the context, say you don't know.

                Question: {question}

                Context:
                {context}
            """

    llm = ChatOllama(model="llama3.1", temperature=0)

    response = llm.invoke(prompt)

    print("\n=== LangChain (manual) ===")
    print(response.content)


def LangChain2(chunks, *, question: str):
    """
    LangChain with RetrievalQA.
    """

    emb = OllamaEmbeddings(model="nomic-embed-text")

    if os.path.exists("chroma_db"):
        vectordb = Chroma(persist_directory="chroma_db", embedding_function=emb)
    else:
        vectordb = Chroma.from_documents(chunks, emb, persist_directory="chroma_db")
        vectordb.persist()

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    llm = ChatOllama(model="llama3.1", temperature=0)
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
    chunks = splitter.split_documents(data)

    LangChain(chunks, question=question)
    LangChain2(chunks, question=question)
    LlamaIndex(question=question)
