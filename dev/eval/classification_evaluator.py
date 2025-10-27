"""
Clean classification evaluation system.
Handles binary classification evaluation with multiple models and ablation study.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline import prepare_data
from model.tf_models import TFModel
from config.pipeline_config import update_config_from_args


class ClassificationEvaluator:
    """
    Clean classification evaluation system.
    Handles binary classification with multiple models and ablation study.
    """
    
    def __init__(self, ticker, start_date, end_date, use_sentiment, scale, test_size, val_size):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.use_sentiment = use_sentiment
        self.scale = scale
        self.test_size = test_size
        self.val_size = val_size
        
        # Create configuration from args (reuse main pipeline logic)
        self.config = self._create_config()
    
    def _create_config(self):
        """Create configuration using main pipeline's configuration system."""
        from argparse import Namespace
        
        # Create args namespace (reuse main pipeline logic)
        args = Namespace(
            company=self.ticker,
            start_date=self.start_date,
            end_date=self.end_date,
            target_feature='Close',
            lookback=10,  # Adaptive lookback for classification
            horizon=1,
            test_size=self.test_size,
            val_size=self.val_size,
            scale=self.scale,
            target_ret=True,  # Classification uses returns
            log_ret=False,
            use_sentiment=self.use_sentiment,
            classification=True,
            model_name='lstm',  # Base model for classification
            layers=[32, 16],  # Smaller layers for classification
            dropout_rate=0.2,
            epochs=10,  # Fewer epochs for classification
            batch_size=32,
            optimizer='adam',
            learning_rate=1e-3,
            sarimax_seasonal=False,
            sarimax_m=5,
            ensemble_2='lstm',
            sarima_weight=0.5,
            model_2_weight=0.5,
            inspect_plots=False
        )
        
        # Use main pipeline's config system
        config = update_config_from_args(args)
        return config
    
    def evaluate(self):
        """Single method: data -> training -> evaluation -> ablation -> results"""
        # 1. Prepare data
        bundle = self._prepare_data()
        
        # 2. Train models
        results = self._train_models(bundle)
        
        # 3. Run ablation study
        baseline_results = self._run_ablation_study()
        
        # 4. Save results
        self._save_results(results, baseline_results)
        
        # 5. Print summary
        self._print_summary(results, baseline_results)
    
    def _prepare_data(self):
        """Prepare data for classification using main pipeline logic."""
        # Use the same prepare_data function from pipeline.py
        return prepare_data(
            ticker=self.config.data.company,
            start_date=self.config.data.start_date,
            end_date=self.config.data.end_date,
            target_feature=self.config.data.target_feature,
            lookback=self.config.data.lookback,
            horizon=self.config.data.horizon,
            test_size=self.config.data.test_size,
            val_size=self.config.data.val_size,
            target_as_return=self.config.data.target_as_return,
            use_log_returns=self.config.data.use_log_returns,
            scale=self.config.data.scale,
            cache_dir='cache',
            use_sentiment=self.config.data.use_sentiment,
        )
    
    def _train_models(self, bundle):
        """Train all classification models using main pipeline logic."""
        # Convert to binary
        y_train_binary = self._to_binary(bundle.y_train, bundle.target_scaler)
        y_test_binary = self._to_binary(bundle.y_test, bundle.target_scaler)
        
        print(f"\nBinary distribution - Train: {np.mean(y_train_binary):.3f} up, Test: {np.mean(y_test_binary):.3f} up")
        
        # Train models
        results = {}
        
        # 1. Train LSTM using main pipeline's model creation logic
        print(f"  Training LSTM...")
        lstm_model = self._create_lstm_model(bundle)
        lstm_pred = self._predict_lstm(lstm_model, bundle)
        results['LSTM'] = self._evaluate_classification(y_test_binary, lstm_pred, 'LSTM')
        
        # 2. Train sklearn models
        sklearn_models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        for model_name, model in sklearn_models.items():
            print(f"  Training {model_name}...")
            pred = self._train_and_predict_sklearn(model, bundle, y_train_binary)
            results[model_name] = self._evaluate_classification(y_test_binary, pred, model_name)
        
        return results
    
    def _create_lstm_model(self, bundle):
        """Create and train LSTM model using main pipeline's TFModel class."""
        # Use the same TFModel class as main pipeline
        model = TFModel(
            input_size=bundle.X_train.shape[2], 
            model_name=self.config.tf_model.model_name, 
            layers=self.config.tf_model.layers, 
            dropout_rate=self.config.tf_model.dropout_rate, 
            output_steps=self.config.data.horizon
        )
        
        # Convert targets to binary
        y_train_binary = self._to_binary(bundle.y_train, bundle.target_scaler)
        y_val_binary = self._to_binary(bundle.y_val, bundle.target_scaler) if bundle.y_val is not None else None
        
        # Train with binary targets
        val_data = (bundle.X_val, y_val_binary) if bundle.X_val is not None and y_val_binary is not None else None
        
        model.fit(
            bundle.X_train,
            y_train_binary,
            epochs=self.config.tf_model.epochs,
            batch_size=self.config.tf_model.batch_size,
            learning_rate=self.config.tf_model.learning_rate,
            optimizer=self.config.tf_model.optimizer,
            validation_data=val_data,
            meta_path=f"{self.ticker}_classification",
        )
        
        return model
    
    def _predict_lstm(self, model, bundle):
        """Predict using LSTM model."""
        pred = model.predict(bundle.X_test)
        return self._to_binary(pred)
    
    def _train_and_predict_sklearn(self, model, bundle, y_train_binary):
        """Train and predict using sklearn model."""
        # Reshape data for sklearn
        X_train_flat = bundle.X_train.reshape(bundle.X_train.shape[0], -1)
        X_test_flat = bundle.X_test.reshape(bundle.X_test.shape[0], -1)
        
        # Train and predict
        model.fit(X_train_flat, y_train_binary.ravel())
        pred = model.predict(X_test_flat)
        return pred
    
    def _run_ablation_study(self):
        """Run ablation study (with/without sentiment) using main pipeline logic."""
        print(f"\n{'='*60}")
        print(f"COMPARISON: SENTIMENT IMPACT")
        print(f"{'='*60}")
        
        # Create baseline config (without sentiment)
        baseline_config = self._create_baseline_config()
        
        # Prepare baseline data
        bundle_baseline = prepare_data(
            ticker=baseline_config.data.company,
            start_date=baseline_config.data.start_date,
            end_date=baseline_config.data.end_date,
            target_feature=baseline_config.data.target_feature,
            lookback=baseline_config.data.lookback,
            horizon=baseline_config.data.horizon,
            test_size=baseline_config.data.test_size,
            val_size=baseline_config.data.val_size,
            target_as_return=baseline_config.data.target_as_return,
            use_log_returns=baseline_config.data.use_log_returns,
            scale=baseline_config.data.scale,
            cache_dir='cache',
            use_sentiment=baseline_config.data.use_sentiment,
        )
        
        # Train baseline LSTM using main pipeline logic
        baseline_model = self._create_lstm_model_baseline(bundle_baseline, baseline_config)
        baseline_pred = self._predict_lstm(baseline_model, bundle_baseline)
        
        y_test_baseline = self._to_binary(bundle_baseline.y_test, bundle_baseline.target_scaler)
        return self._evaluate_classification(y_test_baseline, baseline_pred, "Baseline LSTM (No Sentiment)")
    
    def _create_baseline_config(self):
        """Create baseline configuration without sentiment."""
        from argparse import Namespace
        
        args = Namespace(
            company=self.ticker,
            start_date=self.start_date,
            end_date=self.end_date,
            target_feature='Close',
            lookback=10,
            horizon=1,
            test_size=self.test_size,
            val_size=self.val_size,
            scale=self.scale,
            target_ret=True,
            log_ret=False,
            use_sentiment=False,  # NO SENTIMENT for baseline
            classification=True,
            model_name='lstm',
            layers=[32, 16],
            dropout_rate=0.2,
            epochs=10,
            batch_size=32,
            optimizer='adam',
            learning_rate=1e-3,
            sarimax_seasonal=False,
            sarimax_m=5,
            ensemble_2='lstm',
            sarima_weight=0.5,
            model_2_weight=0.5,
            inspect_plots=False
        )
        
        return update_config_from_args(args)
    
    def _create_lstm_model_baseline(self, bundle, config):
        """Create baseline LSTM model using main pipeline's TFModel class."""
        # Use the same TFModel class as main pipeline
        model = TFModel(
            input_size=bundle.X_train.shape[2], 
            model_name=config.tf_model.model_name, 
            layers=config.tf_model.layers, 
            dropout_rate=config.tf_model.dropout_rate, 
            output_steps=config.data.horizon
        )
        
        # Convert targets to binary
        y_train_binary = self._to_binary(bundle.y_train, bundle.target_scaler)
        y_val_binary = self._to_binary(bundle.y_val, bundle.target_scaler) if bundle.y_val is not None else None
        
        # Train with binary targets
        val_data = (bundle.X_val, y_val_binary) if bundle.X_val is not None and y_val_binary is not None else None
        
        model.fit(
            bundle.X_train,
            y_train_binary,
            epochs=config.tf_model.epochs,
            batch_size=config.tf_model.batch_size,
            learning_rate=config.tf_model.learning_rate,
            optimizer=config.tf_model.optimizer,
            validation_data=val_data,
            meta_path=f"{self.ticker}_baseline_classification",
        )
        
        return model
    
    def _to_binary(self, y, scaler=None):
        """Convert to binary labels (0/1)."""
        if scaler is not None:
            # Descale if scaler is present
            y_descaled = scaler.inverse_transform(y)
            return (y_descaled > 0).astype(int)
        else:
            return (y > 0).astype(int)
    
    def _evaluate_classification(self, y_true, y_pred, model_name="Model"):
        """Evaluate binary classification performance."""
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Print results
        print(f"\n{model_name} Classification Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Print confusion matrix
        self._print_confusion_matrix(cm, y_true)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
    
    def _print_confusion_matrix(self, cm, y_true_binary):
        """Print confusion matrix."""
        print(f"\n  Confusion Matrix:")
        if cm.shape == (2, 2):
            print(f"    True Neg: {cm[0,0]:4d} | False Pos: {cm[0,1]:4d}")
            print(f"    False Neg: {cm[1,0]:4d} | True Pos:  {cm[1,1]:4d}")
        elif cm.shape == (1, 1):
            if y_true_binary[0] == 1:
                print(f"    All Positive: {cm[0,0]:4d} samples")
            else:
                print(f"    All Negative: {cm[0,0]:4d} samples")
        else:
            print(f"    Confusion Matrix Shape: {cm.shape}")
            print(f"    Matrix: {cm}")
    
    def _save_results(self, results, baseline_results):
        """Save classification results and create plots."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{self.ticker}_{self.start_date}_to_{self.end_date}_classification_{timestamp}"
        csv_filename = f"results/{base_filename}.csv"
        
        # Prepare results data
        results_data = []
        for model_name, metrics in results.items():
            results_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1_score'],
                'Sentiment_Features': self.use_sentiment,
            })
        
        # Add baseline comparison
        if baseline_results:
            results_data.append({
                'Model': 'LSTM_Baseline',
                'Accuracy': baseline_results['accuracy'],
                'Precision': baseline_results['precision'],
                'Recall': baseline_results['recall'],
                'F1_Score': baseline_results['f1_score'],
                'Sentiment_Features': False,
            })
        
        # Save to CSV
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(csv_filename, index=False)
        print(f"\nResults saved to: {csv_filename}")
        
        # Create plots
        self._create_plots(results, baseline_results, base_filename)
    
    def _create_plots(self, results, baseline_results, base_filename):
        """Create classification plots."""
        # 1. Confusion Matrix Plot
        self._plot_confusion_matrices(results, base_filename)
        
        # 2. Metrics Comparison Plot
        self._plot_metrics_comparison(results, baseline_results, base_filename)
        
        # 3. Sentiment Impact Plot (if baseline available)
        if baseline_results:
            self._plot_sentiment_impact(results, baseline_results, base_filename)
    
    def _plot_confusion_matrices(self, results, base_filename):
        """Plot confusion matrices for all models."""
        n_models = len(results)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (model_name, metrics) in enumerate(results.items()):
            if i >= 4:  # Only plot first 4 models
                break
                
            cm = metrics['confusion_matrix']
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{model_name} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
            
            # Add labels
            if cm.shape == (2, 2):
                axes[i].set_xticklabels(['Down', 'Up'])
                axes[i].set_yticklabels(['Down', 'Up'])
        
        # Hide unused subplots
        for i in range(len(results), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_path = f"results/{base_filename}_confusion_matrices.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrices plot saved to: {plot_path}")
    
    def _plot_metrics_comparison(self, results, baseline_results, base_filename):
        """Plot metrics comparison across models."""
        # Prepare data
        models = list(results.keys())
        if baseline_results:
            models.append('LSTM_Baseline')
        
        metrics_data = {
            'Accuracy': [results[model]['accuracy'] for model in results.keys()] + 
                       ([baseline_results['accuracy']] if baseline_results else []),
            'Precision': [results[model]['precision'] for model in results.keys()] + 
                        ([baseline_results['precision']] if baseline_results else []),
            'Recall': [results[model]['recall'] for model in results.keys()] + 
                     ([baseline_results['recall']] if baseline_results else []),
            'F1-Score': [results[model]['f1_score'] for model in results.keys()] + 
                       ([baseline_results['f1_score']] if baseline_results else [])
        }
        
        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, (metric, values) in enumerate(metrics_data.items()):
            ax.bar(x + i * width, values, width, label=metric, alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title(f'Classification Metrics Comparison - {self.ticker}')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f"results/{base_filename}_metrics_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Metrics comparison plot saved to: {plot_path}")
    
    def _plot_sentiment_impact(self, results, baseline_results, base_filename):
        """Plot sentiment impact analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        models = ['LSTM + Sentiment', 'LSTM Baseline']
        accuracies = [results['LSTM']['accuracy'], baseline_results['accuracy']]
        colors = ['#2E8B57', '#DC143C']  # Green for sentiment, Red for baseline
        
        bars1 = ax1.bar(models, accuracies, color=colors, alpha=0.7)
        ax1.set_title('Accuracy: Sentiment vs Baseline')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # F1-Score comparison
        f1_scores = [results['LSTM']['f1_score'], baseline_results['f1_score']]
        bars2 = ax2.bar(models, f1_scores, color=colors, alpha=0.7)
        ax2.set_title('F1-Score: Sentiment vs Baseline')
        ax2.set_ylabel('F1-Score')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, f1 in zip(bars2, f1_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom')
        
        plt.suptitle(f'Sentiment Impact Analysis - {self.ticker}', fontsize=16)
        plt.tight_layout()
        plot_path = f"results/{base_filename}_sentiment_impact.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Sentiment impact plot saved to: {plot_path}")
    
    def _print_summary(self, results, baseline_results):
        """Print classification summary."""
        print(f"\n{'='*60}")
        print(f"SUMMARY COMPARISON")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 60)
        
        for model_name, metrics in results.items():
            print(f"{model_name:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f}")
        
        if baseline_results:
            print(f"\n{'='*60}")
            print(f"SENTIMENT IMPACT ANALYSIS")
            print(f"{'='*60}")
            print(f"Model                     Accuracy   Precision  Recall     F1-Score")
            print("-" * 60)
            print(f"LSTM + Sentiment          {results['LSTM']['accuracy']:<10.4f} {results['LSTM']['precision']:<10.4f} {results['LSTM']['recall']:<10.4f} {results['LSTM']['f1_score']:<10.4f}")
            print(f"LSTM Baseline             {baseline_results['accuracy']:<10.4f} {baseline_results['precision']:<10.4f} {baseline_results['recall']:<10.4f} {baseline_results['f1_score']:<10.4f}")
            
            accuracy_improvement = results['LSTM']['accuracy'] - baseline_results['accuracy']
            f1_improvement = results['LSTM']['f1_score'] - baseline_results['f1_score']
            
            print(f"\nSentiment Impact:")
            print(f"  Accuracy improvement: {accuracy_improvement:+.4f} ({accuracy_improvement*100:+.1f}%)")
            print(f"  F1-Score improvement:  {f1_improvement:+.4f} ({f1_improvement*100:+.1f}%)")
            
            if abs(accuracy_improvement) < 0.001 and abs(f1_improvement) < 0.001:
                print("Sentiment features have NO IMPACT")
            else:
                print("Sentiment features have IMPACT")


def run_classification_evaluation(ticker, start_date, end_date, use_sentiment=False, scale=False, test_size=0.2, val_size=0.2):
    """
    Run classification evaluation using ClassificationEvaluator.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        use_sentiment: Whether to use sentiment features
        scale: Whether to scale features
        test_size: Test set size ratio
        val_size: Validation set size ratio
    """
    evaluator = ClassificationEvaluator(ticker, start_date, end_date, use_sentiment, scale, test_size, val_size)
    evaluator.evaluate()  # This does everything: data prep, training, evaluation, ablation, results
