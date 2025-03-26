from dotenv import load_dotenv

import os
from pinecone import Pinecone, ServerlessSpec

load_dotenv()  # Load environment variables from .env file

class EmbeddingGenerator:
    def __init__(self):
        # Load API key from environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

    def chunk_text_by_tokens(self, text, chunk_size, encoding_name="cl100k_base"):
        """
        Splits the text into chunks based on the number of tokens.
        """
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        return [encoding.decode(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]

    def generate_embeddings(self, chunks):
        """
        Generates embeddings for each chunk and returns a list of embeddings.
        """
        embeddings = []
        for chunk in chunks:
            response = openai.embeddings.create(
                input=chunk,
                model="text-embedding-ada-002"
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def process_text(self, text, chunk_size=1000):
        """
        Splits the text it into chunks, and generates the embeddings.
        """
        chunks = self.chunk_text_by_tokens(text, chunk_size)
        embeddings = self.generate_embeddings(chunks)
        return chunks, embeddings
    
class PineconeStore:
    def __init__(self, environment="us-east-1"):
        # Load API key from environment variables
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")

        # Create a Pinecone instance
        self.pc = Pinecone(api_key=pinecone_api_key)

        # Define the index name
        self.index_name = "pdf-vector-store"

        # Check if the index exists, if not create it
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )

    def save_vectors(self, vectors, metadata, chunks):
        # Get the index
        index = self.pc.Index(self.index_name)

        # Iterate over the embeddings and save each one with unique metadata
        for i, vector in enumerate(vectors):
            vector_id = f"{metadata['id']}_chunk_{i}"  # Unique ID for each chunk
            chunk_metadata = {
                "id": vector_id,
                "source": metadata["source"],
                "chunk": i,
                "text": chunks[i]  # Add the text of the chunk here
            }
            # Upsert each vector with its corresponding metadata
            index.upsert(vectors=[(vector_id, vector, chunk_metadata)])

if __name__ == '__main__':
    vector_store = PineconeStore()
    vector_store.save_vector(embedding, {"id": "doc_1", "source": "Minecraft.pdf"}, chunks)
    