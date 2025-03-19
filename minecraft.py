"""
pip install 
openai
langchain
tiktoken
pinecone-client
python-dotenv
"""
import os
from dotenv import load_dotenv, find_dotenv
import langchain
import pinecone
import tiktoken
import openai
import logging
import time
from pinecone import Pinecone, ServerlessSpec

def cargar_documento(archivo):
    import os
    nombre, extension = os.path.splitext(archivo) 
    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Cargando {archivo}...')
        loader = PyPDFLoader(archivo)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Cargando {archivo}...')
        loader = Docx2txtLoader(archivo)
    else:
        print('El formato del documento no está soportado!')
        return None

    data = loader.load()
    return data

# wikipedia
def desde_wikipedia(busqueda, lang='es', load_max_docs=3):
    from langchain.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query=busqueda, lang=lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data

def fragmentar(data, chunk_size=200):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=20)
    fragmentos = text_splitter.split_documents(data)
    return fragmentos

def costo_embedding(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total Tokens: {total_tokens}')
    print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0001:.5f}')

def borrar_indices(pc, index_name='todos'):
    if index_name == 'todos':
        indexes = pc.list_indexes().names()
        print('Borrando todos los índices ... ')
        for index in indexes:
            pc.delete_index(index)
        print('Listo!')
    else:
        print(f'Borrando el índice: {index_name} ...', end='')
        pc.delete_index(index_name)
        print('Listo')
    
def creando_vectores(pc, index_name, fragmentos):
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings()
    
    if index_name in pc.list_indexes().names():
        print(f'El índice {index_name} ya existe. Cargando los embeddings ... ', end='')
        vectores = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')
    else:
        print(f'Creando el índice {index_name} y los embeddings ...', end='')
        pc.create_index(
            name=index_name, 
            dimension=1536, 
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        vectores = Pinecone.from_documents(fragmentos, embeddings, index_name=index_name)
        print('Ok')
        
    return vectores

def consultas(vectores, pregunta):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

    retriever = vectores.as_retriever(search_type='similarity', search_kwargs={'k': 3})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    answer = chain.run(pregunta)
    return answer

def consulta_con_memoria(vectores, pregunta, memoria=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI
    
    llm = ChatOpenAI(temperature=1)
    retriever = vectores.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    
    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    respuesta = crc({'question': pregunta, 'chat_history': memoria})
    memoria.append((pregunta, respuesta['answer']))
    
    return respuesta, memoria

def main():
    # Cargar las variables de entorno desde el archivo .env
    load_dotenv()

    # Inicializar Pinecone
    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

    # Cargar el documento
    documento = "minecraft.pdf"
    contenido = cargar_documento(documento)
    fragmentos = fragmentar(contenido)
    print(f"El Número de fragmentos es de: {len(fragmentos)} fragmentos")
    costo_embedding(fragmentos)
    borrar_indices(pc, "todos")
    index_name = 'minecraft'
    vectores = creando_vectores(pc, index_name, fragmentos)

    memoria = []
    while True:
        pregunta = input("Realiza una pregunta escribe 'salir' para terminar: \n")
        if pregunta == "salir":
            print("Adios!!!")
            break
        else:
            respuesta, memoria = consulta_con_memoria(vectores, pregunta, memoria)
            print(respuesta['answer'])

if __name__ == "__main__":
    main()