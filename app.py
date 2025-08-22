import streamlit as st
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the GUI module
from gui import main as gui_main

def main():
    """
    Main entry point for the Comparative Financial QA System
    """
    # Configure Streamlit page
    st.set_page_config(
        page_title="Comparative Financial QA System",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .method-tag {
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        font-size: 0.8rem;
    }
    
    .rag-tag {
        background-color: #e3f2fd;
        color: #1976d2;
    }
    
    .finetune-tag {
        background-color: #f3e5f5;
        color: #7b1fa2;
    }
    
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display main header
    st.markdown('<h1 class="main-header">📊 Comparative Financial QA System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem;">RAG vs Fine-Tuning Comparison Platform</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.rag_model_loaded = False
        st.session_state.finetune_model_loaded = False
        st.session_state.documents_processed = False
        st.session_state.query_history = []
        st.session_state.comparison_results = []
        st.session_state.rag_pipeline = None
        st.session_state.rag_stats = {}
        st.session_state.finetuning_trainer = None
        st.session_state.training_progress = None
        st.session_state.training_active = False
    
    # Run the main GUI
    gui_main()

if __name__ == "__main__":
    main() 