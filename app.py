"""
Streamlit Web Application for Text Summarization

This app provides a web interface for text summarization using both extractive
and abstractive methods.
"""

import streamlit as st
from text_summarizer import summarize_text, extractive_summarize, abstractive_summarize, load_summarizer_model, load_spacy_model

# Cache model loading to avoid reloading on every rerun
@st.cache_resource
def get_spacy_model():
    """Load and cache spaCy model."""
    return load_spacy_model()

@st.cache_resource
def get_summarizer_model():
    """Load and cache summarizer model."""
    return load_summarizer_model()

def load_models():
    """Load models with caching for Streamlit."""
    try:
        nlp = get_spacy_model()
        summarizer = get_summarizer_model()
        return True, None
    except Exception as e:
        return False, str(e)

# Page configuration
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
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
    .summary-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📝 Text Summarizer</h1>', unsafe_allow_html=True)
st.markdown("---")

# Check and load models (cached, so only loads once)
try:
    with st.spinner("Loading models... This may take a moment on first run (downloading ~850MB)."):
        models_loaded, model_error = load_models()
    
    if not models_loaded:
        st.error(f"❌ Model Loading Error: {model_error}")
        st.info("""
        **Troubleshooting Steps:**
        1. Make sure all dependencies are installed: `pip install -r requirements.txt`
        2. Install spaCy model: `python -m spacy download en_core_web_sm`
        3. The T5 model will download automatically on first use (~850MB)
        4. Check your internet connection for model download
        5. Ensure you have enough disk space and memory
        6. Try restarting the Streamlit app
        7. Check the terminal/console for detailed error messages
        """)
        st.stop()
except Exception as e:
    st.error(f"❌ Unexpected Error: {str(e)}")
    st.info("Please check the terminal/console for detailed error messages.")
    st.stop()

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Method selection
    method = st.radio(
        "Select Summarization Method:",
        ["Abstractive", "Extractive"],
        help="Abstractive: Creates new sentences. Extractive: Selects existing sentences."
    )
    
    if method == "Extractive":
        num_sentences = st.slider(
            "Number of sentences:",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of sentences to include in the extractive summary"
        )
    else:
        max_length = st.slider(
            "Maximum summary length:",
            min_value=50,
            max_value=200,
            value=100,
            step=10,
            help="Maximum number of tokens in the abstractive summary"
        )
        
        min_length = st.slider(
            "Minimum summary length:",
            min_value=10,
            max_value=50,
            value=10,
            step=5,
            help="Minimum number of tokens in the abstractive summary"
        )
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.info("""
    This app uses two methods for text summarization:
    
    **Extractive**: Selects the most important sentences from the original text based on word frequency.
    
    **Abstractive**: Uses AI (T5 transformer) to generate new sentences that summarize the content.
    """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 Input Text")
    input_text = st.text_area(
        "Enter or paste your text here:",
        height=400,
        placeholder="Type or paste the text you want to summarize here...",
        help="Enter the text you want to summarize"
    )
    
    # Character and word count
    if input_text:
        char_count = len(input_text)
        word_count = len(input_text.split())
        st.caption(f"📊 Characters: {char_count} | Words: {word_count}")

with col2:
    st.subheader("✨ Summary")
    
    if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
        if input_text and len(input_text.strip()) > 0:
            with st.spinner("Generating summary... Please wait."):
                try:
                    if method == "Extractive":
                        summary = summarize_text(
                            input_text,
                            method='extractive',
                            num_sentences=num_sentences
                        )
                    else:
                        summary = summarize_text(
                            input_text,
                            method='abstractive',
                            max_length=max_length,
                            min_length=min_length
                        )
                    
                    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                    st.write(summary)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Summary statistics
                    summary_char_count = len(summary)
                    summary_word_count = len(summary.split())
                    compression_ratio = (1 - summary_char_count / char_count) * 100 if input_text else 0
                    
                    st.success(f"✅ Summary generated successfully!")
                    st.caption(f"📊 Summary - Characters: {summary_char_count} | Words: {summary_word_count} | Compression: {compression_ratio:.1f}%")
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Summary",
                        data=summary,
                        file_name="summary.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"❌ Error: {error_msg}")
                    
                    # Provide specific troubleshooting based on error
                    if "spaCy model" in error_msg or "en_core_web_sm" in error_msg:
                        st.info("""
                        **Fix:** Install the spaCy model by running:
                        ```
                        python -m spacy download en_core_web_sm
                        ```
                        """)
                    elif "transformers" in error_msg.lower() or "torch" in error_msg.lower():
                        st.info("""
                        **Fix:** Install required packages:
                        ```
                        pip install transformers torch
                        ```
                        """)
                    elif "CUDA" in error_msg or "device" in error_msg.lower():
                        st.info("""
                        **Note:** The app will use CPU if GPU is not available.
                        This is normal and will work, just slower.
                        """)
                    else:
                        st.info("""
                        **Troubleshooting:**
                        1. Check that all dependencies are installed: `pip install -r requirements.txt`
                        2. Restart the Streamlit app
                        3. Check the terminal/console for detailed error messages
                        """)
        else:
            st.warning("⚠️ Please enter some text to summarize.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Built with ❤️ using Streamlit | Text Summarizer App"
    "</div>",
    unsafe_allow_html=True
)
