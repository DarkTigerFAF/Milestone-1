import asyncio
import os
from dotenv import load_dotenv

from openai import AsyncOpenAI
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.hugging_face import HuggingFaceTextEmbedding
from sklearn.metrics.pairwise import cosine_similarity


async def main():
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    LLM_MODEL = "llama-3.1-8b-instant"
    EMBED_MODEL = "BAAI/bge-small-en-v1.5"

    kernel = sk.Kernel()

    groq_client = AsyncOpenAI(
        api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1"
    )

    kernel.add_service(
        OpenAIChatCompletion(
            service_id="chat",
            ai_model_id=LLM_MODEL,
            async_client=groq_client,
        )
    )

    kernel.add_service(
        HuggingFaceTextEmbedding(service_id="embed", ai_model_id=EMBED_MODEL, device=0)
    )

    embed_service = kernel.get_service("embed")

    documents = {
        "doc1": "Python is a high-level programming language created by Guido van Rossum in 1991.",
        "doc2": "Semantic Kernel is a lightweight SDK to mix programming languages with AI services.",
        "doc3": "RAG combines information retrieval with text generation for grounded AI responses.",
        "doc4": "Machine learning lets systems learn from experience without explicit programming.",
    }
    doc_texts = list(documents.values())

    doc_embeddings = [
        await embed_service.generate_embeddings(text) for text in doc_texts
    ]

    async def rag_query(question: str) -> str:
        print(f"\nQ: {question}")

        q_embed = await embed_service.generate_embeddings(question)

        sims = cosine_similarity([q_embed], doc_embeddings)[0]
        top_idx = sims.argsort()[-2:][::-1]
        context = "\n\n".join(doc_texts[i] for i in top_idx)

        prompt = """
        Context:
        {{$context}}

        Question: {{$question}}
        Answer:
        """

        result = await kernel.invoke_prompt(
            prompt=prompt, context=context, question=question, service_id="chat"
        )

        return str(result)

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
