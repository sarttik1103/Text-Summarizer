# Text Summarizer - Setup Guide

## Installation Steps

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install spaCy language model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Troubleshooting

### Error: "Summarizer model not loaded"

If you encounter this error, try the following:

1. **Check if transformers and torch are installed:**
   ```bash
   pip install transformers torch
   ```

2. **Verify installation:**
   ```bash
   python -c "from transformers import pipeline; print('OK')"
   ```

3. **Check internet connection:**
   - The T5 model (~850MB) downloads automatically on first use
   - Ensure you have a stable internet connection

4. **Check disk space:**
   - Ensure you have at least 2GB free disk space
   - Models are cached in `~/.cache/huggingface/`

5. **Check memory:**
   - The T5-base model requires at least 2GB RAM
   - Close other applications if needed

6. **Manual model download (if automatic fails):**
   ```python
   from transformers import pipeline
   summarizer = pipeline("summarization", model="t5-base")
   ```

### Error: "spaCy model not found"

Run:
```bash
python -m spacy download en_core_web_sm
```

### If models still don't load:

1. **Check Python version:**
   - Requires Python 3.8 or higher
   ```bash
   python --version
   ```

2. **Reinstall dependencies:**
   ```bash
   pip uninstall transformers torch spacy
   pip install transformers torch spacy
   ```

3. **Clear cache and retry:**
   ```bash
   # Clear HuggingFace cache (optional)
   rm -r ~/.cache/huggingface/
   ```

4. **Check terminal/console output:**
   - Streamlit shows detailed errors in the terminal
   - Look for specific error messages

## System Requirements

- Python 3.8+
- 2GB+ RAM
- 2GB+ free disk space
- Internet connection (for first-time model download)

## Notes

- First run will be slower as models download
- Subsequent runs will be faster due to caching
- The app uses CPU by default (GPU optional)
