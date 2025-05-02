import boto3
import streamlit as st
from langchain_aws import BedrockLLM, BedrockEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Define custom prompt template
custom_prompt = """You are a helpful assistant. Answer the question based on the provided context, but summarize with 250 words or less. If the answer is not in the context, say "I don't know."
\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"""

# Initialize Bedrock client
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# Get embeddings
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    region_name="us-east-1",
    client=bedrock_client
)

# Load and split documents
def get_documents():
    loader = PyPDFDirectoryLoader("data")  # Folder containing PDFs
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    docs = text_splitter.split_documents(documents)
    return docs

# Create vector store
def get_vectorstore(docs):
    vectorstore_faiss = FAISS.from_documents(docs, embeddings)
    return vectorstore_faiss

# Initialize LLM
def get_llm():
    llm = BedrockLLM(
        model_id="mistral.mixtral-8x7b-instruct-v0:1",  # Change to actual model ID if needed
        region_name="us-east-1",
        client=bedrock_client,
        temperature=0.2,
        max_tokens=512,
    )
    return llm

# Custom prompt template instance
prompt_template_instance = PromptTemplate(
    template=custom_prompt,
    input_variables=["context", "question"],
)

# Get LLM response
def get_llm_response(llm, vectorstore_faiss, question):
    retriever = vectorstore_faiss.as_retriever(search_kwargs={"k": 1})
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template_instance},
)
    answer = qa({"query": question})
    return answer

# Streamlit UI
def main():
    st.set_page_config(page_title="Document QA System", page_icon="📘")
    st.title("Document Question Answering System")
    st.header("End-to-End Bedrock RAG")

    question = st.text_input("Ask a question about the document:")
    
    with st.sidebar:
        st.title("Settings")

        if st.button("Load Documents"):
            with st.spinner("Processing..."):
                docs = get_documents()
            vectorstore = get_vectorstore(docs)
            vectorstore.save_local("faiss_index")  # Save the FAISS index
            st.success("Documents loaded and vector store created.")

        if st.button("send"):
            with st.spinner("Processing..."):
                vectorstore_faiss = FAISS.load_local(
                    "faiss_index",
                    embeddings,
                    allow_dangerous_deserialization=True  # Allow loading trusted pickle
                )
                llm = get_llm()
                st.write(get_llm_response(llm, vectorstore_faiss, question))


if __name__ == "__main__":
    main()
