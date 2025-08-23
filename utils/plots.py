from loguru import logger
import matplotlib.pyplot as plt

def plot_predictions(actual_prices, predicted_prices, test_dates, save_path, company="STOCK"):
    """Generate clean prediction plots with better formatting"""
    plt.figure(figsize=(14, 7))
    
    # Plot with better styling
    plt.plot(test_dates, actual_prices, color="black", linewidth=2, label=f"Actual {company} Price")
    plt.plot(test_dates, predicted_prices, color="green", linewidth=2, linestyle="--", label=f"Predicted {company} Price")
    
    # Improve formatting
    plt.title(f"{company} Share Price Prediction", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(f"{company} Share Price ($)", fontsize=12)
    plt.legend(fontsize=11, loc='upper left')
    
    # Better date formatting
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save and display
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"Plot saved: {save_path}")
    logger.info(f"Plot saved to: {save_path}")


