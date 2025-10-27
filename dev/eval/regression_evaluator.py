"""
Clean regression evaluation system.
Handles predictions, metrics, plots, and results for regression models.
"""
import pandas as pd
import numpy as np
from pipeline import predict, predictions_to_prices
from eval.metrics import compute_metrics
from utils.plots import plot_predictions, plot_sentiment


class RegressionEvaluator:
    """
    Clean regression evaluation system.
    Handles predictions, metrics, plots, and results.
    """
    
    def __init__(self, model, bundle, args, meta_path):
        self.model = model
        self.bundle = bundle
        self.args = args
        self.meta_path = meta_path
    
    def evaluate(self):
        """Single method: predictions -> metrics -> plots -> results"""
        # 1. Generate predictions
        print(f"\nGenerating predictions...")
        y_hat = predict(self.model, self.bundle)
        y_true = self.bundle.y_test
        
        print(f"Predictions shape: {y_hat.shape}")
        print(f"Ground truth shape: {y_true.shape}")
        
        # 2. Convert to price space
        print(f"\nConverting to price space...")
        y_true_px, y_hat_px = predictions_to_prices(y_true, y_hat, self.bundle)
        
        print(f"Price space - y_true: [{y_true_px.min():.2f}, {y_true_px.max():.2f}]")
        print(f"Price space - y_hat: [{y_hat_px.min():.2f}, {y_hat_px.max():.2f}]")
        
        # 3. Compute metrics
        epochs_trained = self._get_epochs_trained()
        metrics = compute_metrics(y_true_px, y_hat_px, epochs_trained=epochs_trained)
        
        # 4. Generate plots
        self._generate_plots(y_true_px, y_hat_px)
        
        # 5. Save results
        self._save_results(metrics)
        
        # 6. Print summary
        self._print_summary(metrics)
        
        # 7. Final status
        self._print_final_status(y_hat_px)
    
    def _get_epochs_trained(self):
        """Get number of epochs trained from model."""
        epochs_trained = getattr(self.model, 'epochs_trained', None)
        if epochs_trained is None:
            model_2 = getattr(self.model, 'model_2', None)
            if model_2 is not None:
                epochs_trained = getattr(model_2, 'epochs_trained', None)
        return epochs_trained
    
    def _generate_plots(self, y_true_px, y_hat_px):
        """Generate all plots."""
        dates = self._get_dates()
        
        # Predictions plot
        plot_predictions(
            y_true_px[:, 0], 
            y_hat_px[:, 0], 
            self.args.company, 
            f"results/{self.meta_path}_predictions.png", 
            dates
        )
        
        # Sentiment plot if applicable
        if self.args.use_sentiment:
            plot_sentiment(
                self.bundle, 
                f"results/{self.meta_path}_sentiment_vs_price_splits.png"
            )
    
    def _get_dates(self):
        """Get dates for plotting."""
        dates = None
        if hasattr(self.bundle, 'test_df') and self.bundle.test_df is not None:
            try:
                dates = self.bundle.test_df.index[-len(self.bundle.y_test):]
            except Exception:
                dates = None
        return dates
    
    def _save_results(self, metrics):
        """Save results to CSV."""
        metric_names = ['mae', 'rmse', 'directional_accuracy']
        metric_values = [metrics['mae'], metrics['rmse'], metrics['directional_accuracy']]
        
        if 'epochs_trained' in metrics:
            metric_names.append('epochs_trained')
            metric_values.append(metrics['epochs_trained'])
        
        results_df = pd.DataFrame({'metric': metric_names, 'value': metric_values})
        report_path = f"results/{self.meta_path}.csv"
        results_df.to_csv(report_path, index=False)
        print(f"\nResults saved to: {report_path}")
    
    def _print_summary(self, metrics):
        """Print evaluation summary."""
        print(f"\n{'=' * 60}")
        print(f"EVALUATION RESULTS - {self.args.model_name.upper()}")
        print(f"{'=' * 60}")
        print(f"MAE: {metrics['mae']:.6f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"Directional Accuracy (DA@1): {metrics['directional_accuracy']:.3f}")
        if 'epochs_trained' in metrics:
            print(f"Epochs Trained: {metrics['epochs_trained']}/{self.args.epochs}")
        print(f"{'=' * 60}")
    
    def _print_final_status(self, y_hat_px):
        """Print final status."""
        print(f"\n{'=' * 60}")
        if np.isfinite(y_hat_px).all() and (y_hat_px > 0).all():
            print("STATUS: OK")
        else:
            print("STATUS: FAIL (NaN/Inf/negative prices detected)")
        print(f"{'=' * 60}")


def run_regression_evaluation(model, bundle, args, meta_path):
    """
    Run regression evaluation using RegressionEvaluator.
    
    Args:
        model: Trained model
        bundle: DataBundle with test data
        args: Command line arguments
        meta_path: Meta path for saving results
        
    Returns:
        dict: Evaluation metrics
    """
    evaluator = RegressionEvaluator(model, bundle, args, meta_path)
    evaluator.evaluate()
    
    # Return metrics for potential further use
    y_hat = predict(model, bundle)
    y_true = bundle.y_test
    y_true_px, y_hat_px = predictions_to_prices(y_true, y_hat, bundle)
    epochs_trained = evaluator._get_epochs_trained()
    return compute_metrics(y_true_px, y_hat_px, epochs_trained=epochs_trained)
