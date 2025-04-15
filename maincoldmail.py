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
import requests
from bs4 import BeautifulSoup
import traceback

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("GOOGLE_API_KEY not found in environment variables")
    st.stop()
genai.configure(api_key=google_api_key)

# Constants
JOB_DESCRIPTION_PROMPT = """
You are an expert career coach helping candidates write personalized cold emails for job applications.
Generate a professional and compelling cold email based on the following job description and candidate information.
The email should be tailored to highlight how the candidate's skills and experience match the job requirements.

Job Description:
{job_description}

Candidate Information:
{resume_text}

Additional Instructions:
- Keep the email concise (150-200 words)
- Use a professional but approachable tone
- Highlight 2-3 most relevant qualifications
- Include a clear call to action
- Personalize the opening if possible

Email Structure:
1. Personalized greeting
2. Brief introduction
3. Relevant qualifications/skills
4. Why you're interested in the role/company
5. Call to action (request for interview)
6. Professional closing
"""

def extract_text_from_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Get text from common job description containers
        job_description = ""
        possible_containers = [
            'div.job-description',
            'div.description',
            'div.job-details',
            'section.job-description',
            'div#jobDescriptionText',
            'div.job_description'
        ]
        
        for selector in possible_containers:
            elements = soup.select(selector)
            if elements:
                job_description = ' '.join([e.get_text(separator=' ', strip=True) for e in elements])
                break
        
        if not job_description:
            job_description = soup.get_text(separator=' ', strip=True)
        
        return job_description
    
    except Exception as e:
        st.error(f"Error extracting text from URL: {str(e)}")
        return None

def get_pdf_text(pdf_docs):
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
    if not text.strip():
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, 
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("No text chunks provided for vector store creation")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are an expert career coach helping candidates with job applications. 
    Generate a professional and compelling cold email based on the provided job description and candidate's resume.
    
    Job Description Context:\n{context}\n
    
    Candidate's Resume:\n{resume_text}\n
    
    Additional Instructions:
    - Keep the email concise (150-200 words)
    - Use a professional but approachable tone
    - Highlight 2-3 most relevant qualifications
    - Include a clear call to action
    - Personalize the opening if possible
    
    Email Structure:
    1. Personalized greeting
    2. Brief introduction
    3. Relevant qualifications/skills
    4. Why you're interested in the role/company
    5. Call to action (request for interview)
    6. Professional closing
    
    Generated Email:
    """
    
    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            temperature=0.3
        )
    except Exception as e:
        st.error(f"Failed to initialize Gemini model: {str(e)}")
        st.error("Please check your API key and ensure the model is available")
        raise
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "resume_text"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def generate_cold_email(job_description, resume_text):
    try:
        if not os.path.exists("faiss_index"):
            st.error("Vector store not found. Please process job description first.")
            return

        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        new_db = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Search for relevant parts of the job description
        docs = new_db.similarity_search(job_description)
        context = "\n".join([doc.page_content for doc in docs])
        
        chain = get_conversational_chain()
        
        response = chain.invoke(
            {"input_documents": docs, "resume_text": resume_text}
        )
        
        return response["output_text"]
        
    except Exception as e:
        st.error(f"An error occurred while generating the cold email: {str(e)}")
        st.error("Please try again or upload your documents once more.")
        return None

def main():
    #st.set_page_config("Cold Email Generator", page_icon="✉️")
    st.header("AI-Powered Cold Email Generator for Job Applications")
    
    st.warning(
        "Note: This application processes documents you upload. "
        "Only upload documents from trusted sources."
    )

    # Input options
    input_option = st.radio(
        "Select input method for job description:",
        ("Website URL", "PDF Upload", "Direct Text Input")
    )
    
    job_description_text = ""
    
    if input_option == "Website URL":
        url = st.text_input("Enter job posting URL:")
        if url:
            with st.spinner("Extracting job description from URL..."):
                job_description_text = extract_text_from_url(url)
                if job_description_text:
                    st.success("Job description extracted successfully!")
    
    elif input_option == "PDF Upload":
        job_pdf = st.file_uploader(
            "Upload Job Description PDF",
            type=["pdf"],
            accept_multiple_files=False
        )
        if job_pdf:
            with st.spinner("Extracting text from PDF..."):
                job_description_text = get_pdf_text([job_pdf])
                if job_description_text:
                    st.success("PDF text extracted successfully!")
    
    else:  # Direct Text Input
        job_description_text = st.text_area(
            "Paste job description here:",
            height=200
        )
    
    # Resume/CV input
    st.subheader("Your Resume/CV")
    resume_pdf = st.file_uploader(
        "Upload your resume/CV (PDF)",
        type=["pdf"],
        accept_multiple_files=False
    )
    
    resume_text = ""
    if resume_pdf:
        with st.spinner("Extracting text from your resume..."):
            resume_text = get_pdf_text([resume_pdf])
            if resume_text:
                st.success("Resume extracted successfully!")
    
    # Process job description and generate email
    if st.button("Generate Cold Email") and job_description_text and resume_text:
        with st.spinner("Processing job description and generating email..."):
            try:
                # Process job description
                job_chunks = get_text_chunks(job_description_text)
                if not job_chunks:
                    st.error("Failed to process job description.")
                    return
                
                get_vector_store(job_chunks)
                
                # Generate email
                email = generate_cold_email(job_description_text, resume_text)
                
                if email:
                    st.subheader("Generated Cold Email")
                    st.text_area("Email Content", email, height=300)
                    
                    # Add download button
                    st.download_button(
                        label="Download Email",
                        data=email,
                        file_name="cold_email.txt",
                        mime="text/plain"
                    )
                
            except Exception as e:
                st.error(f"Error generating email: {str(e)}")
                st.error(traceback.format_exc())

if __name__ == "__main__":
    main()

def app():
    main()