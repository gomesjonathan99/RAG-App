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

    def pdf_to_text(self) -> str:
        """Converts the PDF URL to text."""
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(self.pdf_url, headers=headers, stream=True)
            if response.status_code == 200:
                pdf_stream = BytesIO(response.content)
                with pdfplumber.open(pdf_stream) as pdf:
                    pdf_str = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                print(pdf_str)
            # loader = PyPDFLoader(self.pdf_url)
            # pages = loader.load()  
            # pdf_str = ''.join([page.page_content for page in pages])
            return pdf_str
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return None

    def text_splitter(self, pdf_str: str) -> list:
        """Splits the text into smaller chunks for better processing."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        chunks = text_splitter.split_text(pdf_str)
        print(f"Split into {len(chunks)} chunks.")
        return list(set(chunks)) 

    def create_vector_db(self, chunks: list) -> Chroma:
        """Creates a vector database from the chunks."""
        try:
            vector_db = Chroma.from_texts(texts=chunks, embedding=self.embeddings, persist_directory="./chroma_db")
            vector_db_as_retriever = vector_db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 6,
                    "fetch_k": 15,
                    "lambda_mult": 0.9,
                }
            )
            return vector_db_as_retriever
        except Exception as e:
            print(f"Error creating vector DB: {e}")
            return None
