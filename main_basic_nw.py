import streamlit as st
from PIL import Image
import importlib.util
import sys
import os

# Set page config
st.set_page_config(
    page_title="Tesseract AI",
    page_icon="🔮",
    layout="wide"
)

# Mapping of radio options to python files
USE_CASE_MAPPING = {
    "Cold Email Generator": "01_main_cold_mail",
    "RAG-based Q&A Chatbot": "02_main_q_a_pdf",
    "Oncology Support Chatbot": "03_main_oncology",
    "Question Generator from PDF": "04_main_q_gen",
    "PDF Summarization System": "05_main_pdf_summary",
    "Sentiment Analysis": "06_main_sentiment",
    "Sign Language Translation": "07_main_sign"
}

# Sidebar navigation
with st.sidebar:
    st.header("Select Use Case")
    use_case = st.radio(
        "Demonstration Use Cases",
        options=list(USE_CASE_MAPPING.keys()),
        index=0,
        help="Select a use case to explore"
    )
    
    st.markdown("---")
    st.subheader(f"About {use_case}")
    # Add your descriptions here as shown in your original code

# Main content area - will be dynamically replaced
main_container = st.empty()

# Function to load and run module
def run_module(module_name):
    try:
        file_path = f"{module_name}.py"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # Clear previous content and run the selected module
        with main_container.container():
            module.main()  # Assuming each module has a main() function
    except Exception as e:
        st.error(f"Error loading {module_name}: {str(e)}")

# Initial load or when selection changes
if 'prev_use_case' not in st.session_state or st.session_state.prev_use_case != use_case:
    st.session_state.prev_use_case = use_case
    module_name = USE_CASE_MAPPING[use_case]
    run_module(module_name)

# Default content when no module is selected (shouldn't happen with radio)
else:
    with main_container.container():
        st.title("Tesseract AI")
        st.subheader("Select a use case from the sidebar to begin")
