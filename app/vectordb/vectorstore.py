'''Importing necessary libraries'''
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import requests
import pdfplumber
from io import BytesIO
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

load_dotenv()

class VectorStore:
    
    def __init__(self, pdf_url: str):
        """Initialize the VectorStore with the PDF URL."""
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.pdf_url = pdf_url  
        self.headers = {"User-Agent": "Mozilla/5.0"} 
        self.docs = None

    def pdf_to_docs(self) -> str:   
        """Converts the PDF URL to text."""
        loader = PyPDFLoader(self.pdf_url,headers=self.headers)
        self.docs = loader.load()
        return self.docs

    def text_splitter(self, docs:list ) -> list:
        """Splits the text into smaller chunks for better processing."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_documents(docs)
        print(f"Split into {len(chunks)} chunks.")
        return chunks

    def create_vector_db(self, chunks: list) -> Chroma:
        """Creates a vector database from the chunks."""
        try:
            vector_db = Chroma.from_documents(documents=chunks, embedding=self.embeddings,persist_directory="./chroma_db")
            retriever = vector_db.as_retriever(
                search_type="similarity"
            )
            return retriever
        except Exception as e:
            print(f"Error creating vector DB: {e}")
            return None
