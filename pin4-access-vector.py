from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQA

class PineconeRetriever:
    def __init__(self, pinecone_api_key, openai_api_key):
        load_dotenv()  # Load environment variables from .env file

        # Initialize Pinecone connection
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = "pdf-vector-store"
        self.index = self.pc.Index(self.index_name)

        # Initialize OpenAI model and embeddings
        self.embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = OpenAI(temperature=0, api_key=openai_api_key)

        # Create the Pinecone vector store
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embedding_model, text_key="text")
        self.retriever = self.vector_store.as_retriever()

        # Create the RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.retriever)

    def query(self, query_text):
        # Execute the QA chain with the input query
        response = self.qa_chain.invoke({"query": query_text})
        return response['result']


if __name__ == '__main__':
    # Replace with your actual API keys
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    retriever = PineconeRetriever(pinecone_api_key=pinecone_api_key, openai_api_key=openai_api_key)
    result = retriever.query("YOUR QUERY HERE")
    print(result)
    