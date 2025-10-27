"""
Configuration Management System for Financial AI Pipeline

This module provides centralized configuration management for all 5 execution modes:
1. Sentiment Analysis
2. TF Models (LSTM, GRU, RNN, BiLSTM)
3. SARIMAX
4. Classification
5. Ensemble
"""

from dataclasses import dataclass
from typing import Optional, List

@dataclass
class DataConfig:
    """Core data configuration"""
    company: str = "AAPL"
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    target_feature: str = "Close"
    lookback: int = 60
    horizon: int = 1
    test_size: float = 0.2
    val_size: float = 0.0
    scale: bool = True
    use_sentiment: bool = False
    target_as_return: bool = False
    use_log_returns: bool = False

@dataclass
class TFModelConfig:
    """TensorFlow model configuration"""
    model_name: str = "lstm"
    layers: Optional[List[int]] = None
    dropout_rate: float = 0.2
    epochs: int = 25
    batch_size: int = 32
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    
    def __post_init__(self):
        if self.layers is None:
            self.layers = [50, 50, 50]

@dataclass
class SARIMAXConfig:
    """SARIMAX model configuration"""
    seasonal: bool = False
    m: int = 5

@dataclass
class EnsembleConfig:
    """Ensemble model configuration"""
    ensemble_2: str = "lstm"
    sarima_weight: float = 0.5
    model_2_weight: float = 0.5

@dataclass
class ClassificationConfig:
    """Classification model configuration"""
    enabled: bool = False
    models: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.models is None:
            self.models = ["LSTM", "Logistic Regression", "Random Forest", "SVM"]

@dataclass
class SentimentConfig:
    """Sentiment analysis configuration"""
    enabled: bool = False
    model_name: str = "ProsusAI/finbert"
    aggregation_method: str = "mean"  # mean, median, weighted
    include_news_count: bool = True

@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    data: DataConfig
    tf_model: TFModelConfig
    sarimax: SARIMAXConfig
    ensemble: EnsembleConfig
    classification: ClassificationConfig
    sentiment: SentimentConfig
    
    # Execution mode
    mode: str = "tf_model"  # sentiment, tf_model, sarimax, classification, ensemble
    
    def __post_init__(self):
        # Set mode based on configuration
        if self.classification.enabled:
            self.mode = "classification"
        elif self.data.use_sentiment and not any([
            self.tf_model.model_name in ['lstm', 'gru', 'rnn', 'bilstm'],
            self.tf_model.model_name == 'sarimax',
            self.tf_model.model_name == 'ensemble'
        ]):
            self.mode = "sentiment"
        elif self.tf_model.model_name == "sarimax":
            self.mode = "sarimax"
        elif self.tf_model.model_name == "ensemble":
            self.mode = "ensemble"
        else:
            self.mode = "tf_model"

class ConfigManager:
    """Centralized configuration manager"""
    
    def __init__(self):
        self.config = PipelineConfig(
            data=DataConfig(),
            tf_model=TFModelConfig(),
            sarimax=SARIMAXConfig(),
            ensemble=EnsembleConfig(),
            classification=ClassificationConfig(),
            sentiment=SentimentConfig()
        )
    
    def from_args(self, args) -> 'PipelineConfig':
        """Create configuration from command line arguments"""
        
        # Data configuration
        self.config.data.company = args.company
        self.config.data.start_date = args.start_date
        self.config.data.end_date = args.end_date
        self.config.data.target_feature = args.target_feature
        self.config.data.lookback = args.lookback
        self.config.data.horizon = args.horizon
        self.config.data.test_size = args.test_size
        self.config.data.val_size = args.val_size
        self.config.data.scale = args.scale
        self.config.data.use_sentiment = args.use_sentiment
        self.config.data.target_as_return = args.target_ret or args.log_ret
        self.config.data.use_log_returns = args.log_ret
        
        # TF Model configuration
        self.config.tf_model.model_name = args.model_name
        self.config.tf_model.layers = args.layers
        self.config.tf_model.dropout_rate = args.dropout_rate
        self.config.tf_model.epochs = args.epochs
        self.config.tf_model.batch_size = args.batch_size
        self.config.tf_model.optimizer = args.optimizer
        self.config.tf_model.learning_rate = args.learning_rate
        
        # SARIMAX configuration
        self.config.sarimax.seasonal = args.sarimax_seasonal
        self.config.sarimax.m = args.sarimax_m
        
        # Ensemble configuration
        self.config.ensemble.ensemble_2 = args.ensemble_2
        self.config.ensemble.sarima_weight = args.sarima_weight
        self.config.ensemble.model_2_weight = args.model_2_weight
        
        # Classification configuration
        self.config.classification.enabled = args.classification
        
        # Sentiment configuration
        self.config.sentiment.enabled = args.use_sentiment
        
        # Determine execution mode
        self._determine_mode()
        
        return self.config
    
    def _determine_mode(self):
        """Determine execution mode based on configuration"""
        if self.config.classification.enabled:
            self.config.mode = "classification"
        elif self.config.tf_model.model_name == "sarimax":
            self.config.mode = "sarimax"
        elif self.config.tf_model.model_name == "ensemble":
            self.config.mode = "ensemble"
        elif self.config.sentiment.enabled and not any([
            self.config.tf_model.model_name in ['lstm', 'gru', 'rnn', 'bilstm'],
            self.config.tf_model.model_name == 'sarimax',
            self.config.tf_model.model_name == 'ensemble'
        ]):
            self.config.mode = "sentiment"
        else:
            self.config.mode = "tf_model"
    
    def get_model_path(self, meta_path: str) -> str:
        """Get model save path - simple path construction"""
        return f"cache/trained_models/{self.config.data.company}_{self.config.tf_model.model_name}_{meta_path.replace('.keras', '')}.keras"
    
    def get_results_path(self, filename: str) -> str:
        """Get results save path - simple path construction"""
        return f"results/{filename}"
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any errors"""
        errors = []
        
        # Data validation
        if self.config.data.lookback <= 0:
            errors.append("Lookback must be positive")
        
        if self.config.data.test_size <= 0 or self.config.data.test_size >= 1:
            errors.append("Test size must be between 0 and 1")
        
        if self.config.data.val_size < 0 or self.config.data.val_size >= 1:
            errors.append("Validation size must be between 0 and 1")
        
        if self.config.data.test_size + self.config.data.val_size >= 1:
            errors.append("Test size + validation size must be less than 1")
        
        # Model validation
        if self.config.tf_model.epochs <= 0:
            errors.append("Epochs must be positive")
        
        if self.config.tf_model.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        if self.config.tf_model.dropout_rate < 0 or self.config.tf_model.dropout_rate >= 1:
            errors.append("Dropout rate must be between 0 and 1")
        
        if self.config.tf_model.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        
        # Ensemble validation
        if self.config.mode == "ensemble":
            if abs(self.config.ensemble.sarima_weight + self.config.ensemble.model_2_weight - 1.0) > 1e-6:
                errors.append("Ensemble weights must sum to 1.0")
        
        return errors
    
    def print_config(self):
        """Print current configuration"""
        print("=" * 60)
        print("PIPELINE CONFIGURATION")
        print("=" * 60)
        print(f"Execution Mode: {self.config.mode.upper()}")
        print(f"Company: {self.config.data.company}")
        print(f"Date Range: {self.config.data.start_date} to {self.config.data.end_date}")
        print(f"Target Feature: {self.config.data.target_feature}")
        print(f"Lookback: {self.config.data.lookback}")
        print(f"Horizon: {self.config.data.horizon}")
        print(f"Test Size: {self.config.data.test_size}")
        print(f"Validation Size: {self.config.data.val_size}")
        print(f"Scaling: {'Enabled' if self.config.data.scale else 'Disabled'}")
        print(f"Sentiment: {'Enabled' if self.config.data.use_sentiment else 'Disabled'}")
        
        if self.config.mode == "tf_model":
            print(f"Model: {self.config.tf_model.model_name.upper()}")
            print(f"Layers: {self.config.tf_model.layers}")
            print(f"Dropout: {self.config.tf_model.dropout_rate}")
            print(f"Epochs: {self.config.tf_model.epochs}")
            print(f"Batch Size: {self.config.tf_model.batch_size}")
            print(f"Optimizer: {self.config.tf_model.optimizer}")
            print(f"Learning Rate: {self.config.tf_model.learning_rate}")
        
        elif self.config.mode == "sarimax":
            print(f"SARIMAX Seasonal: {'Enabled' if self.config.sarimax.seasonal else 'Disabled'}")
            print(f"Seasonal Period: {self.config.sarimax.m}")
        
        elif self.config.mode == "ensemble":
            print(f"Ensemble Components: SARIMAX + {self.config.ensemble.ensemble_2.upper()}")
            print(f"SARIMAX Weight: {self.config.ensemble.sarima_weight}")
            print(f"TF Model Weight: {self.config.ensemble.model_2_weight}")
        
        elif self.config.mode == "classification":
            models = self.config.classification.models or []
            print(f"Classification Models: {', '.join(models)}")
        
        print("=" * 60)

# Global configuration manager instance
config_manager = ConfigManager()

def get_config() -> PipelineConfig:
    """Get current configuration"""
    return config_manager.config

def update_config_from_args(args) -> PipelineConfig:
    """Update configuration from command line arguments"""
    return config_manager.from_args(args)

def validate_current_config() -> List[str]:
    """Validate current configuration"""
    return config_manager.validate_config()

def print_current_config():
    """Print current configuration"""
    config_manager.print_config()
