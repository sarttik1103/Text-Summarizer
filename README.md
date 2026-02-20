# 📝 Text Summarizer

A powerful web application for text summarization using both **extractive** and **abstractive** methods. Built with Streamlit, spaCy, and Transformers (T5 model).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- **Two Summarization Methods:**
  - **Extractive Summarization**: Selects the most important sentences from the original text based on word frequency analysis
  - **Abstractive Summarization**: Uses AI (T5 transformer) to generate new sentences that summarize the content

- **User-Friendly Web Interface:**
  - Clean and intuitive Streamlit-based UI
  - Real-time text input with character and word count
  - Adjustable parameters for both methods
  - Download summaries as text files

- **Smart Features:**
  - Automatic model caching for faster subsequent runs
  - GPU support (automatically uses GPU if available)
  - Error handling with helpful troubleshooting messages
  - Compression ratio calculation

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection (for downloading models on first run)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/text-summarizer.git
   cd text-summarizer
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install spaCy language model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser:**
   - The app will automatically open in your default browser
   - If not, navigate to `http://localhost:8501`

## 📖 Usage

### Web Application

1. **Launch the app** using `streamlit run app.py`
2. **Select a method** (Extractive or Abstractive) from the sidebar
3. **Adjust parameters:**
   - For **Extractive**: Set the number of sentences (1-10)
   - For **Abstractive**: Set min/max summary length
4. **Enter your text** in the input area
5. **Click "Generate Summary"** to get your summary
6. **Download** the summary if needed

### Command Line Usage

You can also use the summarizer as a Python module:

```python
from text_summarizer import summarize_text

# Extractive summarization
summary = summarize_text(
    text="Your long text here...",
    method='extractive',
    num_sentences=3
)

# Abstractive summarization
summary = summarize_text(
    text="Your long text here...",
    method='abstractive',
    max_length=100,
    min_length=10
)
```

Or run it directly:

```bash
python text_summarizer.py
```

## 🏗️ Project Structure

```
text-summarizer/
│
├── app.py                 # Streamlit web application
├── text_summarizer.py     # Core summarization module
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── SETUP.md              # Detailed setup guide
```

## 🔧 Technologies Used

- **Streamlit**: Web application framework
- **spaCy**: Natural language processing library
- **Transformers (Hugging Face)**: Pre-trained T5 model for abstractive summarization
- **PyTorch**: Deep learning framework
- **Python**: Programming language

## 🧠 How It Works

### Extractive Summarization

1. Tokenizes the input text using spaCy
2. Removes stop words and punctuation
3. Calculates word frequencies
4. Scores sentences based on word importance
5. Selects top N sentences as summary

### Abstractive Summarization

1. Uses pre-trained T5-base model from Hugging Face
2. Adds "summarize:" prefix (required by T5)
3. Tokenizes input text
4. Generates summary using beam search
5. Decodes and cleans the output

## 📋 Requirements

See `requirements.txt` for full list. Main dependencies:

- `streamlit>=1.28.0`
- `spacy>=3.7.0`
- `transformers>=4.30.0`
- `torch>=2.0.0`
- `sentencepiece>=0.1.99`

## ⚙️ System Requirements

- **RAM**: Minimum 2GB (4GB+ recommended)
- **Disk Space**: ~2GB free space (for model downloads)
- **Internet**: Required for first-time model download (~850MB)

## 🐛 Troubleshooting

### Model Loading Errors

**Error: "Summarizer model not loaded"**
- Ensure transformers and torch are installed: `pip install transformers torch`
- Check internet connection (models download on first use)
- Verify sufficient disk space (~2GB)

**Error: "spaCy model not found"**
- Run: `python -m spacy download en_core_web_sm`

### Performance Issues

- **Slow first run**: Normal - models download on first use (~850MB)
- **Slow generation**: Abstractive summarization is computationally intensive
- **Memory errors**: Close other applications or use extractive method

### Other Issues

- Check the terminal/console for detailed error messages
- Ensure Python 3.8+ is installed
- Try reinstalling dependencies: `pip install -r requirements.txt --upgrade`

For more details, see [SETUP.md](SETUP.md)

## 📸 Screenshots

*Add screenshots of your application here*

## 🎯 Future Enhancements

- [ ] Support for multiple languages
- [ ] Batch processing for multiple documents
- [ ] Integration with PDF and DOCX files
- [ ] Custom model fine-tuning options
- [ ] API endpoint for programmatic access
- [ ] User authentication and history

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Sarttik Panja**
- GitHub: [@sarttik1103](https://github.com/sarttik1103)
- Email: sarttikpa1103@gmail.com

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the T5 model
- [spaCy](https://spacy.io/) for NLP capabilities
- [Streamlit](https://streamlit.io/) for the web framework

## 📊 Model Information

- **T5-base**: ~850MB, requires ~2GB RAM
- **spaCy en_core_web_sm**: ~50MB
- Models are cached locally after first download

---

⭐ If you find this project helpful, please consider giving it a star!
