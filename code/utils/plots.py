from loguru import logger
import matplotlib.pyplot as plt
from config.data import COMPANY

# def plot_predictions(actual_prices, predicted_prices, save_path="results/"):
#     """
#     Plot the actual and predicted prices
#     Args:
#         actual_prices: The actual prices
#         predicted_prices: The predicted prices
#         save_path: The path to save the plot
#     """
#     plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
#     plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
#     plt.title(f"{COMPANY} Share Price")
#     plt.xlabel("Time")
#     plt.ylabel(f"{COMPANY} Share Price")
#     plt.legend()
#     plt.savefig(save_path)
#     plt.show()  # Display the plot
#     plt.close()
    
def plot_predictions(actual_prices, predicted_prices, test_dates, save_path, company="STOCK"):
    """Generate prediction plots"""
    plt.figure(figsize=(16, 8))
    plt.plot(test_dates, actual_prices, color="black", label=f"Actual {company} Price")
    plt.plot(test_dates, predicted_prices, color="green", label=f"Predicted {company} Price")
    plt.title(f"{company} Share Price")
    plt.xlabel("Date")
    plt.ylabel(f"{company} Share Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    logger.info(f"Plot saved to: {save_path}")