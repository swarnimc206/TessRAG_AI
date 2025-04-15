import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
import requests
import os
from dotenv import load_dotenv
import google.generativeai as genai
import time


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

# --- Knowledge Base Processing ---
class OncologyKnowledgeBase:
    def __init__(self):
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        
    def load_default_resources(self):
        """Loads default oncology resources"""
        default_urls = [
            "https://www.cancer.gov/about-cancer/treatment",
            "https://www.cancer.gov/types",
            "https://www.cancer.gov/research/areas",
            "https://www.cancer.gov/publications/pdq"
        ]
        
        all_text = ""
        for url in default_urls:
            all_text += self._extract_text_from_url(url) + "\n\n"
            
        self.process_text(all_text)
    
    def _extract_text_from_pdf(self, pdf_file):
        text = ""
        try:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            st.error(f"PDF extraction error: {str(e)}")
            return ""
    
    def _extract_text_from_url(self, url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "iframe", "header"]):
                element.decompose()
                
            # Get main content
            main_content = soup.find("main") or soup.find("article") or soup.find("div", class_="main") or soup
            return main_content.get_text(separator=" ", strip=True)
        except Exception as e:
            st.error(f"URL extraction error: {str(e)}")
            return ""
    
    def _split_text(self, text):
        if not text.strip():
            return []
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        return splitter.split_text(text)
    
    def process_text(self, text):
        """Processes text into the knowledge base"""
        chunks = self._split_text(text)
        if not chunks:
            return False
            
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=google_api_key
            )
            self.vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            return True
        except Exception as e:
            st.error(f"Knowledge base processing failed: {str(e)}")
            return False
    
    def initialize_qa_chain(self):
        """Initializes the QA chain with medical-specific prompt"""
        prompt_template = """You are an oncology specialist assistant. Provide accurate, evidence-based information about cancer.
        
        Use the following context to answer the question. If you don't know the answer, say you don't know - don't make up information.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer in this format:
        1. Summary of key information
        2. Relevant statistics (if available)
        3. Current treatment approaches
        4. Recent research findings (if available)
        5. Always cite sources when possible
        
        Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            temperature=0.3,
            google_api_key=google_api_key
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt}
        )
    
    def query(self, question):
        """Query the oncology knowledge base"""
        if not self.qa_chain:
            return "System not properly initialized. Please load knowledge base first."
            
        try:
            result = self.qa_chain.invoke({"query": question})
            return result["result"]
        except Exception as e:
            return f"Error processing query: {str(e)}"

# --- Streamlit UI ---
def main():
    # st.set_page_config(
    #     page_title="Oncology RAG Chatbot",
    #     page_icon="🩺",
    #     layout="wide"
    # )
    
    # Initialize session state
    if "kb" not in st.session_state:
        st.session_state.kb = OncologyKnowledgeBase()
        with st.spinner("Loading default oncology resources..."):
            st.session_state.kb.load_default_resources()
            st.session_state.kb.initialize_qa_chain()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Hello! I'm your oncology information assistant. Ask me anything about cancer types, treatments, or research."
        })
    
    # Sidebar for knowledge management
    with st.sidebar:
        st.header("Knowledge Base Management")
        
        # PDF Upload
        uploaded_file = st.file_uploader(
            "Upload medical PDFs/research papers",
            type=["pdf"],
            accept_multiple_files=False
        )
        
        if uploaded_file and st.button("Add to Knowledge Base"):
            with st.spinner("Processing document..."):
                text = st.session_state.kb._extract_text_from_pdf(uploaded_file)
                if text:
                    success = st.session_state.kb.process_text(text)
                    if success:
                        st.session_state.kb.initialize_qa_chain()
                        st.success("Document added to knowledge base!")
                    else:
                        st.error("Failed to process document")
        
        # URL Input
        url = st.text_input("Add medical resource URL")
        if url and st.button("Add URL Content"):
            with st.spinner("Processing URL..."):
                text = st.session_state.kb._extract_text_from_url(url)
                if text:
                    success = st.session_state.kb.process_text(text)
                    if success:
                        st.session_state.kb.initialize_qa_chain()
                        st.success("URL content added to knowledge base!")
                    else:
                        st.error("Failed to process URL content")
        
        st.markdown("---")
        st.markdown("### Default Sources Include:")
        st.markdown("- NCI Treatment Information")
        st.markdown("- Cancer Type Overviews")
        st.markdown("- Latest Research Areas")
        st.markdown("- PDQ Cancer Information")
    
    # Main chat interface
    st.title("🩺 Oncology Information Assistant")
    st.caption("Powered by RAG with medical knowledge base")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about cancer treatments, types, or research"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Researching your question..."):
                start_time = time.time()
                response = st.session_state.kb.query(prompt)
                elapsed = time.time() - start_time
                
                # Display sources if available
                st.markdown(response)
                st.caption(f"Retrieved in {elapsed:.2f} seconds")
                
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

def app():
    main()
