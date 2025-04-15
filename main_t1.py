import streamlit as st
from PIL import Image

# Import your module pages
import maincoldmail
import mainoncology
import mainpdfsummary
import mainqadf
import mainqgen
import mainsentiment
import mainsign

# Set page configuration
st.set_page_config(
    page_title="Tesseract AI",
    page_icon="🔮",
    layout="wide"
)

# Sidebar Navigation
PAGES = {
    "🏠 Home": None,
    "📧 Cold Mail Generator": maincoldmail,
    "🩺 Oncology Assistant": mainoncology,
    "📄 PDF Q&A": mainqadf,
    "❓ Question Generator": mainqgen,
    "📝 PDF Summarizer": mainpdfsummary,
    "😊 Sentiment Analyzer": mainsentiment,
    "🤟 Sign Language Interpreter": mainsign
}

with st.sidebar:
    st.image("tesseract_logo.png", width=220)
    st.title("🔮 Tesseract AI")
    selection = st.radio("Navigate to", list(PAGES.keys()))

# Main Container
main_container = st.empty()

# Home Page
def show_home_page():
    with main_container.container():
        st.markdown("<h1 style='color:#4B8BBE;'>Tesseract AI</h1>", unsafe_allow_html=True)
        st.markdown("### Enhancing LLM Reliability through Retrieval-Augmented and Context-Aware Generation")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("## 🧩 Problem Statement")
            st.markdown("""
            LLMs are powerful, yet face major challenges:
            - 🔹 **Limited Context Window**: They struggle to retain long input dependencies.
            - 🔹 **Hallucinations**: They can produce fluent yet factually incorrect answers.
            """)

            st.markdown("## 🛠️ Our Solution")
            st.markdown("""
            We propose a twofold strategy:
            - ✅ **Retrieval-Augmented Generation (RAG)**: Enhances factual grounding via external knowledge.
            - ✅ **Context-Aware Generation (CAG)**: Dynamically adapts to long-input scenarios.
            """)

            st.markdown("## 🎯 Core Objectives")
            st.markdown("""
            - Expand LLMs' contextual understanding.
            - Minimize hallucinations.
            - Build practical, real-world apps leveraging enhanced LLMs.
            """)

        with col2:
            st.image("tesseract_logo.png", width=300)

# Run the selected module
def run_selected_app(app_module):
    if app_module is None:
        show_home_page()
    else:
        if hasattr(app_module, "main"):
            app_module.main()
        else:
            st.error("❌ Selected module does not have a `main()` function.")

# Display selected page
run_selected_app(PAGES[selection])
