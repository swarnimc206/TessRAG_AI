import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pandas as pd
import base64
from PIL import Image

## Load API key from Streamlit secrets
google_api_key = st.secrets.get("GOOGLE_API_KEY")
if not google_api_key:
    st.error("GOOGLE_API_KEY not found in Streamlit secrets.")
    st.stop()
genai.configure(api_key=google_api_key)

# Sample Sign Language Database (would be larger in production)
SIGN_DATABASE = {
    "hello": {"video": "hello.mp4", "image": "hello.jpg", "description": "Wave hand"},
    "thank you": {"video": "thanks.mp4", "image": "thanks.jpg", "description": "Hand to chin then outward"},
    "help": {"video": "help.mp4", "image": "help.jpg", "description": "Thumb up on palm"},
    # ... more signs
}

class SignLanguageGenerator:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=google_api_key
        )
        
        # Create vector store from sign descriptions
        self.sign_descriptions = {k: v["description"] for k, v in SIGN_DATABASE.items()}
        self.vector_store = FAISS.from_texts(
            list(self.sign_descriptions.values()),
            embedding=self.embeddings,
            metadatas=[{"sign": k} for k in self.sign_descriptions.keys()]
        )
        
    def find_closest_sign(self, query):
        """Find the closest matching sign using semantic search"""
        docs = self.vector_store.similarity_search(query, k=1)
        if docs:
            return docs[0].metadata["sign"]
        return None
        
    def generate_sign_sequence(self, text_input):
        """Convert text to sequence of signs"""
        words = text_input.lower().split()
        sign_sequence = []
        
        for word in words:
            closest_sign = self.find_closest_sign(word)
            if closest_sign:
                sign_sequence.append(SIGN_DATABASE[closest_sign])
            else:
                # Spell out unknown words
                for letter in word:
                    if letter in SIGN_DATABASE:
                        sign_sequence.append(SIGN_DATABASE[letter])
        
        return sign_sequence

# Streamlit UI
def main():
    # st.set_page_config(page_title="Sign Language Generator", page_icon="👐")
    st.title("Sign Language Generator Using RAG")
    
    # Initialize generator
    if "generator" not in st.session_state:
        st.session_state.generator = SignLanguageGenerator()
    
    # Input
    user_input = st.text_input("Enter text to convert to sign language:", "Hello thank you")
    
    if st.button("Generate Sign Language"):
        with st.spinner("Finding matching signs..."):
            sign_sequence = st.session_state.generator.generate_sign_sequence(user_input)
            
            if not sign_sequence:
                st.warning("No matching signs found")
                return
                
            st.subheader("Sign Language Sequence")
            
            cols = st.columns(3)
            for i, sign in enumerate(sign_sequence):
                with cols[i % 3]:
                    # In production, you would display actual videos/GIFs
                    st.image(Image.new('RGB', (100, 100), color=(i*40, i*20, i*60)))
                    st.caption(f"Sign for: {list(SIGN_DATABASE.keys())[i % len(SIGN_DATABASE)]}")
                    st.write(sign["description"])
            
            st.markdown("### Full Sequence Description")
            st.write(" → ".join([s["description"] for s in sign_sequence]))

if __name__ == "__main__":
    main()

def app():
    main()
