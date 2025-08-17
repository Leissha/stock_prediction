import matplotlib.pyplot as plt
from config.data import COMPANY

def plot_predictions(actual_prices, predicted_prices, save_path="results/"):
    """
    Plot the actual and predicted prices
    Args:
        actual_prices: The actual prices
        predicted_prices: The predicted prices
        save_path: The path to save the plot
    """
    plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
    plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
    plt.title(f"{COMPANY} Share Price")
    plt.xlabel("Time")
    plt.ylabel(f"{COMPANY} Share Price")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    