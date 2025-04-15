import streamlit as st

import maincoldmail
import mainoncology
import mainpdfsummary
import mainqadf
import mainqgen
import mainsentiment
import mainsign

PAGES = {
    "📧 Cold Mail Generator": maincoldmail,
    "🩺 Oncology Assistant": mainoncology,
    "📝 PDF Summarizer": mainpdfsummary,
    "📄 PDF Q&A": mainqadf,
    "❓ Question Generator": mainqgen,
    "😊 Sentiment Analyzer": mainsentiment,
    "🤟 Sign Language Interpreter": mainsign,
}

st.sidebar.title("🧭 Navigation")
selection = st.sidebar.radio("Choose a module", list(PAGES.keys()))

PAGES[selection].app()
