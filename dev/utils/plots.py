import matplotlib.pyplot as plt
from config.data import COMPANY

def plot_predictions(actual_prices, predicted_prices, save_path="results/", dates=None):
    """
    Plot the actual and predicted prices
    Args:
        actual_prices: The actual prices
        predicted_prices: The predicted prices
        dates: The dates of the prices
        save_path: The path to save the plot
    """
    plt.plot(dates, actual_prices, color="black", label=f"Actual {COMPANY} Price")
    plt.plot(dates, predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
    plt.xlabel("Date")
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    plt.title(f"{COMPANY} Share Price")
    plt.ylabel(f"{COMPANY} Share Price")
    plt.legend()
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()  # Display the plot
    plt.close()
    