import asyncio
import os
from dotenv import load_dotenv

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.hugging_face import HuggingFaceTextEmbedding
from sklearn.metrics.pairwise import cosine_similarity
import httpx


async def main():
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    LLM_MODEL = "llama-3.1-8b-instant"
    EMBED_MODEL = "BAAI/bge-small-en-v1.5"

    kernel = sk.Kernel()
    kernel.add_service(OpenAIChatCompletion("groq_chat", LLM_MODEL, GROQ_API_KEY))
    kernel.add_service(
        HuggingFaceTextEmbedding(
            service_id="hf_embed", ai_model_id=EMBED_MODEL, device=0
        )
    )

    embed_service = kernel.get_service("hf_embed")

    # Docs
    documents = {
        "doc1": "Python is a high-level programming language created by Guido van Rossum in 1991.",
        "doc2": "Semantic Kernel is a lightweight SDK to mix programming languages with AI services.",
        "doc3": "RAG combines information retrieval with text generation for grounded AI responses.",
        "doc4": "Machine learning lets systems learn from experience without explicit programming.",
    }

    # Embed all docs with kernel’s embedding service
    doc_texts = list(documents.values())
    doc_embeddings = [
        await embed_service.generate_embeddings(text) for text in doc_texts
    ]

    async def rag_query(question: str) -> str:
        print(f"\nQ: {question}")
        q_embed = await embed_service.generate_embeddings(question)

        # similarity
        sims = cosine_similarity([q_embed], doc_embeddings)[0]
        top_idx = sims.argsort()[-2:][::-1]
        context = "\n\n".join(doc_texts[i] for i in top_idx)

        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30.0,
            )
            return (
                r.json()["choices"][0]["message"]["content"]
                if r.status_code == 200
                else f"Error {r.status_code}"
            )

    # Demo
    for q in [
        "What is Python?",
        "Tell me about Semantic Kernel",
        "What does RAG stand for?",
        "How does ML work?",
    ]:
        print("A:", await rag_query(q))


if __name__ == "__main__":
    asyncio.run(main())
