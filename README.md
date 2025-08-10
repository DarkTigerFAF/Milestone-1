**RAG (Retrieval Augmented Generation)** is similar to giving the existing model additional documents and instructing it to respond to a prompt based on these documents.

**How it works:**
1. You provide the data  
2. It indexes this data into vectors (using an embedding model)  
3. (Optionally) the vectors are saved to disk (which is what I did)  
4. When a prompt is given, it embeds the prompt, calculates the vector similarity between my prompt embedding and the already given context (indexed documents)'s embeddings, retrieves the most relevant chunks, and the model answers using that context.
