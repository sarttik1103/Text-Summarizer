# -*- coding: utf-8 -*-
"""
Text Summarizer Module

This module provides functions for text summarization using spaCy and transformers.
"""

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re


# Global variables for models (will be loaded lazily)
nlp = None
summarizer = None


def load_spacy_model():
    """Load spaCy model lazily."""
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            raise OSError(
                "spaCy model 'en_core_web_sm' not found. Please install it using:\n"
                "python -m spacy download en_core_web_sm"
            )
    return nlp


def load_summarizer_model():
    """Load summarizer model lazily using direct model loading."""
    global summarizer
    if summarizer is None:
        try:
            model_name = 't5-base'
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Load tokenizer and model directly
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model.to(device)
            model.eval()
            
            # T5 models don't have a pad_token by default, set it to eos_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Store as a dict for easy access
            summarizer = {
                'tokenizer': tokenizer,
                'model': model,
                'device': device
            }
        except ImportError as e:
            raise ImportError(
                f"Required libraries not installed. Please install transformers and torch:\n"
                f"pip install transformers torch\n"
                f"Original error: {str(e)}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not load summarizer model. Error: {str(e)}\n"
                f"Make sure you have enough disk space and memory.\n"
                f"The model will be downloaded on first use (~850MB)."
            )
    return summarizer


def extractive_summarize(text, num_sentences=3):
    """
    Perform extractive summarization using frequency-based scoring.
    
    Args:
        text (str): Input text to summarize
        num_sentences (int): Number of sentences to include in summary (default: 3)
    
    Returns:
        str: Summarized text
    """
    nlp_model = load_spacy_model()
    doc = nlp_model(text)
    
    # Tokenize and filter
    tokens = [token.text.lower() for token in doc
              if not token.is_stop and
              not token.is_punct and
              token.text != '\n']
    
    # Calculate word frequencies
    word_freq = Counter(tokens)
    
    if not word_freq:
        return "Unable to generate summary. Text may be too short or contain only stop words."
    
    # Normalize frequencies
    max_freq = max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word] = word_freq[word] / max_freq
    
    # Score sentences
    sent_token = [sent.text for sent in doc.sents]
    sent_score = {}
    
    for sent in sent_token:
        for word in sent.split():
            if word.lower() in word_freq.keys():
                if sent not in sent_score.keys():
                    sent_score[sent] = word_freq[word.lower()]
                else:
                    sent_score[sent] += word_freq[word.lower()]
    
    if not sent_score:
        return "Unable to generate summary. No sentences could be scored."
    
    # Get top N sentences
    n = nlargest(min(num_sentences, len(sent_score)), sent_score, key=sent_score.get)
    return " ".join(n)


def abstractive_summarize(text, max_length=100, min_length=10):
    """
    Perform abstractive summarization using T5 transformer model.
    
    Args:
        text (str): Input text to summarize
        max_length (int): Maximum length of summary (default: 100)
        min_length (int): Minimum length of summary (default: 10)
    
    Returns:
        str: Summarized text
    """
    if not text or len(text.strip()) == 0:
        return "Please provide valid text to summarize."
    
    # Load the model (will be cached in Streamlit)
    summarizer_dict = load_summarizer_model()
    tokenizer = summarizer_dict['tokenizer']
    model = summarizer_dict['model']
    device = summarizer_dict['device']
    
    # T5 has a maximum input length, so we need to truncate if necessary
    # T5-base typically handles up to 512 tokens
    try:
        # Preprocess the text - T5 requires a prefix
        input_text = "summarize: " + text
        
        # Tokenize input (without padding - padding is not needed for generation)
        inputs = tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            padding=False,  # Don't pad - padding causes issues with generation
            return_tensors="pt"
        ).to(device)
        
        # Get pad_token_id (should be set in load_summarizer_model)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
        # Generate summary with improved parameters
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs.get('attention_mask', None),
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                early_stopping=True,
                do_sample=False,
                no_repeat_ngram_size=2,  # Prevent repetition
                length_penalty=0.6,  # Encourage shorter summaries
                pad_token_id=pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode the summary - ensure we skip all special tokens including padding
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the summary - remove any remaining padding, dots, or special characters
        summary = summary.strip()
        
        # Remove sequences of dots (padding artifacts)
        summary = re.sub(r'\.{2,}', '', summary)  # Remove multiple dots
        summary = summary.strip()
        
        # Validate that we got a real summary, not just padding or dots
        if not summary or len(summary.strip()) < 10 or all(c in ['.', ' ', '\n'] for c in summary):
            # Fallback: try with different parameters (more lenient)
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask', None),
                    max_length=max_length,
                    min_length=max(5, min(min_length, 15)),  # Cap min_length
                    num_beams=2,  # Fewer beams
                    early_stopping=True,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            summary = re.sub(r'\.{2,}', '', summary).strip()
        
        # Final validation
        if not summary or len(summary.strip()) < 5:
            return "Unable to generate a meaningful summary. The text might be too short or the model needs different parameters."
        
        return summary
    except Exception as e:
        return f"Error during summarization: {str(e)}"


def summarize_text(text, method='abstractive', num_sentences=3, max_length=100, min_length=10):
    """
    Main function to summarize text using either extractive or abstractive method.
    
    Args:
        text (str): Input text to summarize
        method (str): 'extractive' or 'abstractive' (default: 'abstractive')
        num_sentences (int): Number of sentences for extractive method (default: 3)
        max_length (int): Maximum length for abstractive method (default: 100)
        min_length (int): Minimum length for abstractive method (default: 10)
    
    Returns:
        str: Summarized text
    """
    if not text or len(text.strip()) == 0:
        return "Please provide valid text to summarize."
    
    if method == 'extractive':
        return extractive_summarize(text, num_sentences)
    elif method == 'abstractive':
        return abstractive_summarize(text, max_length, min_length)
    else:
        raise ValueError("Method must be either 'extractive' or 'abstractive'")


if __name__ == "__main__":
    # Example usage
    user_text = input("Enter the text you want to summarize: ")
    
    if user_text:
        print("\n--- Extractive Summary ---")
        extractive_summary = summarize_text(user_text, method='extractive', num_sentences=3)
        print(extractive_summary)
        
        print("\n--- Abstractive Summary ---")
        abstractive_summary = summarize_text(user_text, method='abstractive', max_length=100, min_length=10)
        print(abstractive_summary)
    else:
        print("No text provided.")
