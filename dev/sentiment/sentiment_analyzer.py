"""
Simple FinBERT sentiment analyzer for financial news.

This module provides a lean implementation for sentiment analysis using
the ProsusAI/finbert model, focusing on financial text analysis.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Any
from datetime import datetime, date

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.error("PyTorch and transformers are required. Please install 'torch' and 'transformers'.")


class FinBERTAnalyzer:
    """Simple FinBERT sentiment analyzer."""
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.tokenizer: Any = None
        self.model: Any = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Label mapping for FinBERT
        self.label_mapping = {
            0: 'positive',
            1: 'negative', 
            2: 'neutral'
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load the FinBERT model and tokenizer."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and transformers not available. Install dependencies to use FinBERT.")
        
        try: 
            logger.info(f"Loading FinBERT model: {self.model_name}")
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Prefer safetensors (avoids torch.load CVE requirement)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                use_safetensors=True,
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"FinBERT model loaded successfully on {self.device}")
        except Exception as e:
            # As a fallback, attempt standard load (may require torch>=2.6 for .bin)
            logger.warning(f"Safetensors load failed ({e}). Trying standard weights load...")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval() # Set to evaluation mode
            logger.info(f"FinBERT model loaded with standard weights on {self.device}")
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.

        Returns:
            Dictionary with sentiment analysis results:
                - p_pos: Probability of positive sentiment
                - p_neg: Probability of negative sentiment
                - p_neu: Probability of neutral sentiment
                - sentiment_score: Float in [-1, +1]
                - predicted_class: 'positive', 'negative', or 'neutral'
                - confidence: Float in [0, 1]
        """
        try:
            if self.tokenizer is None or self.model is None:
                raise RuntimeError("FinBERT is not loaded")
            # Tokenize text (truncate to 512 tokens for BERT)
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference (no gradient computation)
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Extract probabilities
            probs = predictions[0].detach().cpu().numpy()
            
            return {
                'p_pos': float(probs[0]),  # positive
                'p_neg': float(probs[1]),  # negative
                'p_neu': float(probs[2]),  # neutral
                'sentiment_score': float(probs[0] - probs[1]),  # p_pos - p_neg
                'predicted_class': int(np.argmax(probs)),
                'confidence': float(np.max(probs))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            # Return neutral sentiment on error
            return {
                'p_pos': 0.33,
                'p_neg': 0.33,
                'p_neu': 0.34,
                'sentiment_score': 0.0,
                'predicted_class': 2,
                'confidence': 0.0
            }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment of multiple texts in batch.
        Args:
            texts: List of text strings to analyze
        Returns:
            List of dictionaries with sentiment analysis results for each text
        """
        results: List[Dict[str, float]] = []
        for text in texts:
            results.append(self.analyze_text(text))
        return results