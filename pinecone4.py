import logging
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import time
import fitz  # PyMuPDF

# Configurar el logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener la clave API desde las variables de entorno
api_key = os.getenv("PINECONE_API_KEY")

# Initialize a Pinecone client with your API key
pc = Pinecone(api_key=api_key)

# *********************************************************
# Target the index
index_name = "dense-index"
dense_index = pc.Index(index_name)

# *********************************************************
# View stats for the index
stats = dense_index.describe_index_stats()
logging.info(f"Index stats: {stats}")
time.sleep(10)

# *********************************************************
# Loop to handle user queries
try:
    while True:
        # Define the query
        query = input("Ingrese su consulta: ")

        # Generate embedding for the query
        query_embedding = pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=[query],
            parameters={
                "input_type": "passage"
            }
        )[0]['values']

        # Search the dense index
        results = dense_index.query(
            vector=query_embedding,
            top_k=10,
            namespace="example-namespace",
            include_metadata=True
        )

        # Print the results
        print("Search results:")
        for match in results["matches"]:
            print(f"id: {match['id']}, score: {round(match['score'], 2)}, text: {match['metadata']['chunk_text']}, category: {match['metadata']['category']}")
        time.sleep(10)

        # *********************************************************
        # Search the dense index and rerank results
        reranked_results = dense_index.query(
            vector=query_embedding,
            top_k=10,
            namespace="example-namespace",
            include_metadata=True,
            rerank={
                "model": "bge-reranker-v2-m3",
                "top_n": 10,
                "rank_fields": ["chunk_text"]
            }
        )

        # Print the reranked results
        print("Reranked results:")
        for match in reranked_results["matches"]:
            print(f"id: {match['id']}, score: {round(match['score'], 2)}, text: {match['metadata']['chunk_text']}, category: {match['metadata']['category']}")
        time.sleep(10)

except KeyboardInterrupt:
    print("\nSaliendo del programa...")

# *********************************************************
# Delete the index (optional)
# pc.delete_index(index_name)
# print(f"Index '{index_name}' deleted.")