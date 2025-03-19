# Import the Pinecone library
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener la clave API desde las variables de entorno
api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=api_key)

"""
index_name = "quickstart"
# Crear el índice y mostrar el resultado
result = pc.create_index(
    name=index_name,
    dimension=2, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)
print(f"Resultado de la creación del índice '{index_name}': {result}")
"""

# Crear un índice denso con incrustación integrada y mostrar el resultado
index_name = "dense-index"
if not pc.has_index(index_name):
    result = pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )
    print(f"Resultado de la creación del índice denso '{index_name}': {result}")
else:
    print(f"El índice '{index_name}' ya existe.")
