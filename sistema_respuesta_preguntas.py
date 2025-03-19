from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import time
import fitz  # PyMuPDF

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener la clave API desde las variables de entorno
api_key = os.getenv("PINECONE_API_KEY")

# Initialize a Pinecone client with your API key
pc = Pinecone(api_key=api_key)

# Crear un índice denso con incrustación integrada
index_name = "dense-index"

# *********************************************************
def extraer_texto_pdf(ruta_pdf):
    """Extrae el texto de un archivo PDF y lo devuelve como una cadena."""
    documento = fitz.open(ruta_pdf)
    texto = ""
    for pagina in documento:
        texto += pagina.get_text()
    return texto

def dividir_en_fragmentos(texto, tamano_fragmento=200):
    """Divide el texto en fragmentos de tamaño especificado."""
    palabras = texto.split()
    fragmentos = []
    for i in range(0, len(palabras), tamano_fragmento):
        fragmento_texto = " ".join(palabras[i:i + tamano_fragmento])
        fragmentos.append(fragmento_texto)
    return fragmentos

# *********************************************************
# Cargar los datos de los "documentos largos"
documentos_largos = [
    extraer_texto_pdf("Minecraft.pdf"),
    # ... otros documentos ...
    "Las fuentes de energía renovable incluyen la energía eólica, solar e hidroeléctrica."
]

# Dividir los documentos en secciones cortas llamadas fragmentos (chunks)
fragmentos = []
for i, documento in enumerate(documentos_largos):
    chunks = dividir_en_fragmentos(documento)
    for j, chunk in enumerate(chunks):
        fragmentos.append({
            "_id": f"frag{i+1}_{j+1}",
            "chunk_text": chunk,
            "category": "general"  # Puedes ajustar la categoría según sea necesario
        })

# Transformar los fragmentos en vectores numéricos (Embeddings)
texts = [fragmento["chunk_text"] for fragmento in fragmentos]
embeddings = pc.inference.embed(
    model="llama-text-embed-v2",
    inputs=texts,
    parameters={
        "input_type": "passage"
    }
)

# Guardar los embeddings en una base de datos vectorial (Pinecone)
records = []
for fragmento, embedding in zip(fragmentos, embeddings):
    records.append({
        "id": fragmento["_id"],
        "values": embedding['values'],
        "metadata": {"chunk_text": fragmento["chunk_text"], "category": fragmento["category"]}
    })

# *********************************************************
# Target the index
dense_index = pc.Index(index_name)

# Upsert the records into a namespace
dense_index.upsert(
    vectors=records,
    namespace="example-namespace"
)

# *********************************************************
# Wait for the upserted vectors to be indexed
time.sleep(10)

# View stats for the index
stats = dense_index.describe_index_stats()
print("Index stats:", stats)
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