import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import traceback
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK data for sentiment analysis
try:
    nltk.data.find('vader_lexicon')
except:
    nltk.download('vader_lexicon')

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("GOOGLE_API_KEY not found in environment variables")
    st.stop()
genai.configure(api_key=google_api_key)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """Analyze sentiment of text and return scores"""
    return sia.polarity_scores(text)

def get_pdf_text(pdf_docs):
    """Extract text from PDF documents with error handling"""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:  # Only add if text was extracted
                    text += page_text
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
            continue
    return text

def get_text_chunks(text):
    """Split text into manageable chunks"""
    if not text.strip():
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, 
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create and save vector store from text chunks"""
    if not text_chunks:
        raise ValueError("No text chunks provided for vector store creation")
    
    # Using the newer text-embedding-004 model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def format_docs(docs):
    """Format documents for RAG context"""
    return "\n\n".join(doc.page_content for doc in docs)

def get_conversational_chain():
    """Create the RAG chain with sentiment-aware prompt"""
    prompt_template = """
    You are an AI assistant with sentiment analysis capabilities. First analyze the sentiment of the question and context, then provide a detailed answer.
    
    Sentiment Analysis:
    - Question Sentiment: {question_sentiment}
    - Context Sentiment: {context_sentiment}
    
    Answer the question as detailed as possible from the provided context. 
    If the question seems negative or frustrated, respond with extra care and empathy.
    If the answer is not in the provided context, say "answer is not available in the context".
    
    Context:\n{context}\n
    Question: {question}
    
    Thoughtful Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        temperature=0.3
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "question_sentiment", "context_sentiment"]
    )
    
    # Create RAG chain
    rag_chain = (
        {
            "context": lambda x: format_docs(x["input_documents"]),
            "question": RunnablePassthrough(),
            "question_sentiment": lambda x: str(analyze_sentiment(x["question"])),
            "context_sentiment": lambda x: str(analyze_sentiment(format_docs(x["input_documents"])))
        }
        | prompt
        | model
        | StrOutputParser()
    )
    
    return rag_chain

def user_input(user_question):
    """Handle user questions with RAG and sentiment analysis"""
    try:
        if not os.path.exists("faiss_index"):
            st.error("Vector store not found. Please process PDFs first.")
            return

        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        # Load FAISS with dangerous deserialization allowed (trusted source only)
        new_db = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Retrieve relevant documents
        docs = new_db.similarity_search(user_question)
        
        # Get RAG chain
        chain = get_conversational_chain()
        
        # Prepare inputs including sentiment analysis
        inputs = {
            "input_documents": docs,
            "question": user_question
        }
        
        # Invoke the chain
        response = chain.invoke(inputs)
        
        # Display results with sentiment information
        st.subheader("Answer")
        st.write(response)
        
        # Show sentiment analysis
        with st.expander("Sentiment Analysis Details"):
            st.write("**Question Sentiment:**", analyze_sentiment(user_question))
            context_text = format_docs(docs)
            st.write("**Context Sentiment:**", analyze_sentiment(context_text))
            
    except Exception as e:
        st.error(f"An error occurred while processing your question: {str(e)}")
        st.error("Please try again or upload your documents once more.")

def main():
    # """Main application function"""
    # st.set_page_config("Chat PDF with Sentiment & RAG", page_icon="📄")
    st.header("Chat with PDF using Gemini with Sentiment Analysis 💁")
    
    # Warning about dangerous deserialization
    st.warning(
        "Note: This application processes PDFs you upload. "
        "Only upload documents from trusted sources."
    )

    user_question = st.text_input("Ask a question about your PDF documents:")
    
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        st.markdown("""
        ### Enhanced PDF Chat Features:
        1. **Sentiment Analysis**: Understands emotional tone
        2. **RAG Pipeline**: More accurate document-based answers
        3. **Context-Aware**: Adapts responses based on sentiment
        
        ### Instructions:
        1. Upload your PDF files
        2. Click 'Submit & Process'
        3. Ask questions about your documents
        """)
        
        pdf_docs = st.file_uploader(
            "Upload PDF Files",
            accept_multiple_files=True,
            type=["pdf"]
        )
        
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
                return

            with st.spinner("Processing your documents..."):
                try:
                    # Extract text
                    raw_text = get_pdf_text(pdf_docs)
                    
                    if not raw_text.strip():
                        st.error("No text could be extracted from the PDF(s). They might be image-based or empty.")
                        return
                    
                    # Perform initial sentiment analysis on the document
                    doc_sentiment = analyze_sentiment(raw_text)
                    with st.expander("Document Sentiment Analysis"):
                        st.write(doc_sentiment)
                    
                    # Split into chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    if not text_chunks:
                        st.error("Failed to split the text into meaningful chunks.")
                        return
                    
                    # Create and store vector store
                    get_vector_store(text_chunks)
                    st.success("Processing complete! You can now ask questions about your documents.")
                    
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}")
                    st.error("Please check your documents and try again.")

if __name__ == "__main__":
    main()

def app():
    main()