from matplotlib import pyplot as plt
import numpy as np


def create_scatter_plot(y_true, y_pred, title="True vs Predicted Values"):
    """Create a scatter plot of true vs predicted values"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Convert to numpy for plotting
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    # Create scatter plot
    ax.scatter(y_true_np, y_pred_np, alpha=0.5, s=10)
    
    # Add diagonal line (perfect predictions)
    min_val = min(y_true_np.min(), y_pred_np.min())
    max_val = max(y_true_np.max(), y_pred_np.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Labels and title
    ax.set_xlabel('True Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig

