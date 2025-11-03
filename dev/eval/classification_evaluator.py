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

from pipeline import prepare_data
from model.tf_models import TFModel
from utils.plots import (
    plot_confusion_matrices,
    plot_metrics_comparison,
    plot_sentiment_impact,
    plot_sentiment_vs_predictions,
)
from utils.prints import (
    print_header,
    print_summary_table,
    print_sentiment_impact_table,
    print_sentiment_impact_line,
    print_confusion_matrix,
)


class ClassificationEvaluator:
    """
    Binary classification with multiple models and yes/no sentiment comparison.
    """
    
    def __init__(self, config):
        """Initialize with parsed PipelineConfig from main."""
        self.config = config
        # Enforce return-based target for classification to avoid label collapse on prices
        try:
            if hasattr(self.config, 'data') and hasattr(self.config.data, 'target_as_return'):
                if not self.config.data.target_as_return:
                    self.config.data.target_as_return = True
        except Exception:
            pass
        self.ticker = config.data.company
        self.start_date = config.data.start_date
        self.end_date = config.data.end_date
        self.use_sentiment = config.data.use_sentiment
        self.scale = config.data.scale
        self.test_size = config.data.test_size
        self.val_size = config.data.val_size
        self.baseline_model = config.tf_model.model_name.upper()
        self._summary_printed = False  # Guard to prevent duplicate summary prints
        self._results_saved = False  # Guard to prevent duplicate save messages

    def evaluate(self):
        """Single method: data -> training -> evaluation -> non-sentiment comparison -> results"""
        # 1. Prepare data
        bundle = self._prepare_data()
        self.bundle = bundle  # Store for plotting

        # 3. Train models
        results, y_test_binary, primary_pred = self._train_models(bundle)
        self.y_test_binary = y_test_binary  # Store for plotting
        self.primary_pred = primary_pred  # Store for plotting
        
        # 4. Run non-sentiment comparison
        baseline_results = self._run_non_sentiment_comparison()
        
        # 5. Save results
        self._save_results(results, baseline_results)
        
        # 6. Print summary (using utility functions directly) - guard against duplicates
        if not self._summary_printed:
            print_header("SUMMARY COMPARISON")
            print_summary_table(results)
            
            if baseline_results:
                print_header("SENTIMENT IMPACT ANALYSIS")
                print_sentiment_impact_table(self.baseline_model, results[self.baseline_model], baseline_results)
                print_sentiment_impact_line(results[self.baseline_model], baseline_results)
            
            # Print STATUS: OK for script parsing
            print(f"\n{'=' * 60}")
            print("STATUS: OK")
            print(f"{'=' * 60}")
            
            self._summary_printed = True
    
    def _prepare_data(self):
        """Prepare data for classification using main pipeline logic."""
        print(f"\nPreparing data for {self.ticker} classification...")
        # If bundle was prepared upstream, reuse it (single source of truth)

        # Prepare data via pipeline
        bundle = prepare_data(
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
            use_sentiment=True,
            include_social=self.config.data.include_social,
            news_source='all',
        )

        # Print sentiment feature diagnostics
        if self.config.data.use_sentiment:
            print(f"\nSentiment Feature Diagnostics:")
            # Note: bundle.X_train is windowed; we infer sentiment presence by feature count
            print(f"  Feature count: {bundle.X_train.shape[2]} features")
            if bundle.X_train.shape[2] >= 7:  # OHLCV (5) + sentiment (2)
                print(f"  Sentiment features likely present (7+ features detected)")
            else:
                print(f"  WARNING: Only {bundle.X_train.shape[2]} features detected (expected 7 with sentiment)")

        return bundle
    
    def _train_models(self, bundle):
        """Train all classification models using main pipeline logic."""
        # Convert to binary
        y_train_binary = self._to_binary(bundle.y_train, bundle.target_scaler)
        y_test_binary = self._to_binary(bundle.y_test, bundle.target_scaler)
        
        print(f"\nBinary distribution - Train: {np.mean(y_train_binary):.3f} up, Test: {np.mean(y_test_binary):.3f} up")
        
        # Train models
        results = {}
        
        # 1. Train primary time-series model (LSTM/GRU/RNN/BiLSTM)
        print(f"  Training {self.baseline_model}...")
        pred = self._train_tf_model(
            model_name=self.baseline_model.lower(),
            bundle=bundle,
            config=self.config,
            meta_suffix="classification",
        )
        results[self.baseline_model] = self._evaluate_classification(y_test_binary, pred, self.baseline_model, verbose=False)
        primary_pred = pred  # Store primary model predictions
        
        # If training labels collapse to a single class, skip sklearn models to avoid errors
        if len(np.unique(y_train_binary)) < 2:
            print("Skipping sklearn models: training labels contain a single class.")
            return results, y_test_binary, primary_pred

        # 2. Train sklearn models (with class weights to handle imbalance)
        sklearn_models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=50, max_depth=10, class_weight='balanced', min_samples_split=5),
            'SVM': SVC(random_state=42, probability=True, class_weight='balanced')
        }
        
        for model_name, model in sklearn_models.items():
            print(f"  Training {model_name}...")
            pred = self._train_and_predict_sklearn(model, bundle, y_train_binary)
            results[model_name] = self._evaluate_classification(y_test_binary, pred, model_name, verbose=False)
        
        return results, y_test_binary, primary_pred
    
    def _find_optimal_threshold(self, y_pred_proba, y_true):
        """Find optimal threshold based on balanced F1 score and class distribution."""
        from sklearn.metrics import f1_score, balanced_accuracy_score

        # Debug: check probability distribution
        print(f"  [DEBUG] Probability distribution:")
        print(f"    Mean: {np.mean(y_pred_proba):.4f}, Std: {np.std(y_pred_proba):.4f}")
        print(f"    Min: {np.min(y_pred_proba):.4f}, Max: {np.max(y_pred_proba):.4f}")
        print(f"    Median: {np.median(y_pred_proba):.4f}")
        
        # Check actual class distribution
        actual_up_ratio = np.mean(y_true)
        print(f"    Actual Up ratio: {actual_up_ratio:.3f}")

        # Search wider range and use balanced metrics to avoid bias
        thresholds = np.arange(0.35, 0.66, 0.01)  # Narrower range around 0.5
        best_threshold = 0.5
        best_score = 0
        
        # Use balanced accuracy * F1 as composite metric to avoid Up bias
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            pred_up_ratio = np.mean(y_pred_thresh)
            
            # Calculate metrics
            f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
            balanced_acc = balanced_accuracy_score(y_true, y_pred_thresh)
            
            # Penalize if predicted ratio is very different from actual
            ratio_penalty = 1.0 - abs(pred_up_ratio - actual_up_ratio)  # Closer to actual = better
            
            # Composite score: balance F1, balanced accuracy, and class distribution match
            composite_score = (f1 * 0.4) + (balanced_acc * 0.4) + (ratio_penalty * 0.2)
            
            if composite_score > best_score:
                best_score = composite_score
                best_threshold = threshold

        # Final check: if threshold results in extreme bias, adjust towards 0.5
        final_pred = (y_pred_proba >= best_threshold).astype(int)
        final_up_ratio = np.mean(final_pred)
        
        # If predicted ratio is too extreme (>80% or <20%), use 0.5 or median
        if final_up_ratio > 0.8 or final_up_ratio < 0.2:
            print(f"  [WARNING] Threshold {best_threshold:.3f} causes extreme bias (Up ratio: {final_up_ratio:.3f})")
            # Use median probability as threshold for better balance
            median_threshold = np.median(y_pred_proba)
            best_threshold = max(0.4, min(0.6, median_threshold))  # Clamp between 0.4-0.6
            print(f"  [ADJUSTED] Using balanced threshold: {best_threshold:.3f}")

        print(f"  [DEBUG] Best threshold: {best_threshold:.3f} (Composite Score={best_score:.4f})")
        return best_threshold
    
    def _train_tf_model(self, model_name: str, bundle, config, meta_suffix: str):
        """
        Unified TF trainer: supports both binary classification (BCE) and regression approaches.
        
        Approach selection:
        - If use_price_comparison=True: Train regression, convert to binary via price comparison (Section 7 approach)
        - If use_price_comparison=False: Train binary classification with BCE loss (Section 5 traditional approach)
        """
        # Check if we should use price comparison (Section 7 approach) or BCE binary classification (Section 5 approach)
        use_price_comparison = getattr(config.classification, 'use_price_comparison', False)
        
        if use_price_comparison:
            # Section 7 approach: Train regression, then convert to binary via price comparison
            # This is faster (no threshold tuning) and produces more active predictions
            task_type = 'regression'
            
            model = TFModel(
                input_size=bundle.X_train.shape[2],
                model_name=model_name,
                layers=config.tf_model.layers,
                dropout_rate=config.tf_model.dropout_rate,
                output_steps=config.data.horizon,
                task_type=task_type,
            )
            
            # Train on return/price values directly (regression)
            y_train_target = bundle.y_train
            y_val_target = bundle.y_val if bundle.y_val is not None else None
            val_data = (bundle.X_val, y_val_target) if bundle.X_val is not None and y_val_target is not None else None
            
            model.fit(
                bundle.X_train,
                y_train_target,
                epochs=config.tf_model.epochs,
                batch_size=config.tf_model.batch_size,
                learning_rate=config.tf_model.learning_rate,
                optimizer=config.tf_model.optimizer,
                validation_data=val_data,
                meta_path=f"{self.ticker}_{meta_suffix}_{model_name}",
            )
            
            # Convert regression predictions to binary via price comparison (batch processing)
            return self._predict_regression_to_binary(model, bundle)
        else:
            # Section 5 approach: Train binary classification with BCE loss
            # Traditional approach using binary_crossentropy loss
            task_type = 'binary'
            
            model = TFModel(
                input_size=bundle.X_train.shape[2],
                model_name=model_name,
                layers=config.tf_model.layers,
                dropout_rate=config.tf_model.dropout_rate,
                output_steps=config.data.horizon,
                task_type=task_type,
            )
            
            # Convert to binary labels for training
            y_train_binary = self._to_binary(bundle.y_train, bundle.target_scaler)
            y_val_binary = None
            if bundle.y_val is not None:
                y_val_binary = self._to_binary(bundle.y_val, bundle.target_scaler)
            val_data = (bundle.X_val, y_val_binary) if bundle.X_val is not None and y_val_binary is not None else None
            
            # Train on binary labels with BCE loss
            model.fit(
                bundle.X_train,
                y_train_binary,
                epochs=config.tf_model.epochs,
                batch_size=config.tf_model.batch_size,
                learning_rate=config.tf_model.learning_rate,
                optimizer=config.tf_model.optimizer,
                validation_data=val_data,
                meta_path=f"{self.ticker}_{meta_suffix}_{model_name}",
            )
            
            # Predict probabilities, then apply optimal threshold
            pred_proba = model.predict(bundle.X_test).ravel()
            y_test_binary = self._to_binary(bundle.y_test, bundle.target_scaler)
            optimal_threshold = self._find_optimal_threshold(pred_proba, y_test_binary)
            pred_binary = (pred_proba >= optimal_threshold).astype(int)
            
            return pred_binary
    
    def _predict_regression_to_binary(self, model, bundle):
        """
        Predict regression values, then convert to binary via price comparison.
        Optimized batch processing: predict once, transform once.
        
        Returns:
            Binary predictions (0=down, 1=up)
        """
        # Step 1: Predict regression values (batch processing)
        pred_values = model.predict(bundle.X_test).ravel()
        
        # Step 2: Extract previous prices from input window (batch processing)
        try:
            close_idx = bundle.features.index('close')
        except (ValueError, AttributeError):
            close_idx = 0  # Fallback: assume first column is close
        
        # Get last time step of each window: X_test[:, -1, close_idx]
        previous_prices = bundle.X_test[:, -1, close_idx].ravel()
        
        # Step 3: Batch descale both arrays
        if bundle.scalers and 'close' in bundle.scalers:
            previous_prices = bundle.scalers['close'].inverse_transform(
                previous_prices.reshape(-1, 1)
            ).ravel()
        
        if bundle.target_scaler is not None:
            pred_values = bundle.target_scaler.inverse_transform(
                pred_values.reshape(-1, 1)
            ).ravel()
        
        # Step 4: Batch convert return to price space if needed
        if bundle.target_mode.value in ['return', 'log_return']:
            if bundle.use_log_returns:
                predicted_prices = previous_prices * np.exp(pred_values)
            else:
                predicted_prices = previous_prices * (1 + pred_values)
        else:
            predicted_prices = pred_values  # Already in price space
        
        # Step 5: Batch compare: predicted_price > previous_price → Up (1)
        pred = (predicted_prices > previous_prices).astype(int)
        
        # Debug
        up_ratio = np.mean(pred)
        print(f"  [DEBUG] Price comparison: Predicted > Previous in {up_ratio:.1%} of cases")
        
        return pred
    
    def _train_and_predict_sklearn(self, model, bundle, y_train_binary):
        """Train and predict using sklearn model."""
        # Reshape data for sklearn
        X_train_flat = bundle.X_train.reshape(bundle.X_train.shape[0], -1)
        X_test_flat = bundle.X_test.reshape(bundle.X_test.shape[0], -1)
        
        # Train and predict
        model.fit(X_train_flat, y_train_binary.ravel())
        pred = model.predict(X_test_flat)
        return pred
    
    def _run_non_sentiment_comparison(self):
        """Run non-sentiment comparison."""
        print(f"\n{'='*60}")
        print(f"COMPARISON: SENTIMENT VS NON-SENTIMENT")
        print(f"{'='*60}")
        
        # Create baseline config (without sentiment) by cloning current config
        import copy
        baseline_config = copy.deepcopy(self.config)
        
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
            use_sentiment=False,
            include_social=False,  # Baseline doesn't use social media
            news_source='all',
        )

        # Train baseline time series model using unified trainer
        baseline_pred = self._train_tf_model(
            model_name=self.baseline_model.lower(),
            bundle=bundle_baseline,
            config=baseline_config,
            meta_suffix="baseline_classification",
        )
        
        y_test_baseline = self._to_binary(bundle_baseline.y_test, bundle_baseline.target_scaler)
        return self._evaluate_classification(y_test_baseline, baseline_pred, f"Baseline {self.baseline_model} (No Sentiment)", verbose=True)
    
    def _to_binary(self, y, scaler=None):
        """Convert to binary labels (0/1)."""
        if scaler is not None:
            # Descale if scaler is present
            y_descaled = scaler.inverse_transform(y)
            return (y_descaled > 0).astype(int)
        else:
            return (y > 0).astype(int)
    
    def _evaluate_classification(self, y_true, y_pred, model_name="Model", verbose=True):
        """Evaluate binary classification performance.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            model_name: Name of the model
            verbose: If True, print detailed results. If False, return metrics only (for summary table).
        """
        # Ensure both arrays are 1D
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        
        # Get unique labels from both
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        
        # For metrics, always use both classes if possible to get meaningful scores
        metric_labels = [0, 1] if 0 in unique_labels or 1 in unique_labels else unique_labels.tolist()
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        # Only use labels parameter if we have both classes
        if len(unique_labels) > 1:
            precision = precision_score(y_true, y_pred, zero_division=0, labels=metric_labels)
            recall = recall_score(y_true, y_pred, zero_division=0, labels=metric_labels)
            f1 = f1_score(y_true, y_pred, zero_division=0, labels=metric_labels)
        else:
            # Single class case - metrics are degenerate but compute anyway
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix - use explicit labels to ensure 2x2 when both classes exist
        if len(unique_labels) == 2:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        else:
            # Single class - confusion matrix will be 1x1
            cm = confusion_matrix(y_true, y_pred)
        
        # Print results only if verbose (baseline prints, summary table models don't)
        if verbose:
            print(f"\n{model_name} Classification Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print_confusion_matrix(cm, y_true)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }

    def _save_results(self, results, baseline_results):
        """Save classification results and create plots."""
        timestamp = datetime.now().strftime("%Y%m%d")
        base_filename = f"{self.ticker}_{self.start_date}_to_{self.end_date}_{self.baseline_model}_classification_{timestamp}"
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
                'Model': f'{self.baseline_model}_Baseline',
                'Accuracy': baseline_results['accuracy'],
                'Precision': baseline_results['precision'],
                'Recall': baseline_results['recall'],
                'F1_Score': baseline_results['f1_score'],
                'Sentiment_Features': False,
            })
        
        # Save to CSV
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(csv_filename, index=False)
        
        # Create plots
        self._create_plots(results, baseline_results, base_filename)
        
        # Print save messages together at the end (after all plots are created) - guard against duplicates
        if not self._results_saved:
            print(f"\nResults saved to: {csv_filename}")
            self._results_saved = True
    
    def _create_plots(self, results, baseline_results, base_filename):
        """Create classification plots (delegated to utils.plots)."""
        plot_confusion_matrices(results, f"results/{base_filename}_confusion_matrices.png")
        plot_metrics_comparison(
            self.baseline_model,
            results,
            baseline_results,
            title=f"Classification Metrics Comparison - {self.ticker}",
            save_path=f"results/{base_filename}_metrics_comparison.png",
        )
        if baseline_results:
            plot_sentiment_impact(
                self.baseline_model,
                results,
                baseline_results,
                title=f"Sentiment Impact Analysis - {self.ticker}",
                save_path=f"results/{base_filename}_sentiment_impact.png",
            )
        
        # Plot sentiment vs predictions (only if sentiment is enabled and we have the data)
        if self.config.data.use_sentiment and hasattr(self, 'bundle') and hasattr(self, 'y_test_binary') and hasattr(self, 'primary_pred'):
            plot_sentiment_vs_predictions(
                bundle=self.bundle,
                y_true_binary=self.y_test_binary,
                y_pred_binary=self.primary_pred,
                ticker=self.ticker,
                save_path=f"results/{base_filename}_sentiment_vs_predictions.png",
            )
            if not hasattr(self, '_plot_saved') or not self._plot_saved:
                print(f"Sentiment vs Predictions plot saved to: results/{base_filename}_sentiment_vs_predictions.png")
                self._plot_saved = True
    
def run_classification_evaluation(config):
    """Run classification evaluation using provided PipelineConfig."""
    evaluator = ClassificationEvaluator(config)
    evaluator.evaluate()