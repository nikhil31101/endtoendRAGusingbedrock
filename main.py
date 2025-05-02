import boto3
import streamlit as st
from langchain_aws import BedrockLLM
from langchain_aws import BedrockEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def get_documents():
    # Load PDF documents from the specified directory
    loader = PyPDFDirectoryLoader("docs")
    documents = loader.load()
    return documents


