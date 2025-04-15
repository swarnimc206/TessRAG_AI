import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
import requests
import os
from dotenv import load_dotenv
import google.generativeai as genai



# --- Configuration ---
def configure_gemini():
    """Handles Gemini API configuration with error handling"""
    try:
        # Use Streamlit Secrets instead of .env
        google_api_key = st.secrets["GOOGLE_API_KEY"]  # Key from Streamlit Secrets
        
        if not google_api_key:
            st.error("❌ GOOGLE_API_KEY not found in Streamlit Secrets")
            st.stop()
        
        genai.configure(api_key=google_api_key)
        return google_api_key
        
    except Exception as e:
        st.error(f"❌ Failed to configure Gemini: {str(e)}")
        st.stop()

google_api_key = configure_gemini()

# --- Document Processing Functions ---
def extract_text_from_pdf(pdf):
    text = ""
    try:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"PDF extraction error: {str(e)}")
        return ""

def extract_text_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "iframe"]):
            element.decompose()
            
        # Get main content
        main_content = soup.find("main") or soup.find("article") or soup
        return main_content.get_text(separator=" ", strip=True)
    except Exception as e:
        st.error(f"URL extraction error: {str(e)}")
        return ""

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    if not text.strip():
        return []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# --- RAG Core Functions ---
def create_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=google_api_key
        )
        return FAISS.from_texts(text_chunks, embedding=embeddings)
    except Exception as e:
        st.error(f"Vector store creation failed: {str(e)}")
        return None

def generate_questions_with_rag(vector_store, num_questions=5):
    prompt_template = """
    You are an expert question generator. Create {num_questions} high-quality questions based on the context.
    
    Guidelines:
    1. Questions should cover key concepts
    2. Mix factual and analytical questions
    3. Avoid yes/no questions
    4. Make questions clear and specific
    
    Context:
    {context}
    
    Generated Questions:
    """
    
    try:
        # Retrieve relevant chunks
        relevant_docs = vector_store.similarity_search("key concepts", k=3)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Initialize model
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            temperature=0.7,
            google_api_key=google_api_key
        )
        
        # Create chain
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "num_questions"]
        )
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        
        # Generate questions
        result = chain.invoke({
            "input_documents": relevant_docs,
            "context": context,
            "num_questions": num_questions
        })
        
        return result["output_text"]
    except Exception as e:
        st.error(f"Question generation failed: {str(e)}")
        return None

# --- Streamlit UI ---
def main():
    # st.set_page_config(
    #     page_title="RAG Question Generator",
    #     page_icon="❓",
    #     layout="centered"
    # )
    
    st.title("📝 RAG-Powered Question Generator")
    st.markdown("Generate quiz questions from documents using AI")
    
    # Input selection
    input_type = st.radio(
        "Select input type:",
        ("PDF", "Website URL", "Direct Text"),
        horizontal=True
    )
    
    # Document input
    text = ""
    if input_type == "PDF":
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_file:
            with st.spinner("Extracting text from PDF..."):
                text = extract_text_from_pdf(uploaded_file)
    elif input_type == "Website URL":
        url = st.text_input("Enter URL:")
        if url:
            with st.spinner("Extracting content from URL..."):
                text = extract_text_from_url(url)
    else:
        text = st.text_area("Paste text here:", height=200)
    
    # Question parameters
    col1, col2 = st.columns(2)
    with col1:
        num_questions = st.number_input(
            "Number of questions",
            min_value=1,
            max_value=20,
            value=5
        )
    with col2:
        question_level = st.selectbox(
            "Difficulty level",
            ["Basic", "Intermediate", "Advanced"]
        )
    
    # Generation button
    if st.button("Generate Questions", type="primary") and text:
        with st.spinner("Generating questions using RAG..."):
            # Process document
            chunks = split_text_into_chunks(text)
            
            if not chunks:
                st.error("No text content found. Try another document.")
                return
            
            # Create vector store
            vector_store = create_vector_store(chunks)
            
            if vector_store:
                # Generate questions
                questions = generate_questions_with_rag(
                    vector_store,
                    num_questions=num_questions
                )
                
                if questions:
                    st.subheader("Generated Questions")
                    st.text_area(
                        "Copy these questions:",
                        questions,
                        height=300
                    )
                    
                    # Download option
                    st.download_button(
                        label="Download Questions",
                        data=questions,
                        file_name="generated_questions.txt",
                        mime="text/plain"
                    )

if __name__ == "__main__":
    main()

def app():
    main()
