import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from plotting import ensure_plot_dir, plot_training_metrics, plot_final_comparison

def compare_models(results, model_type=None, plot_dir='plots'):
    """
    Compare models and save plots to the specified directory
    
    Args:
        results: Dictionary containing model results
        model_type: Optional string indicating model type (e.g., 'TensorFlow', 'NumPy')
        plot_dir: Directory to save plots
    """
    ensure_plot_dir(plot_dir)
    
    if model_type:
        plot_training_metrics(results, model_type, plot_dir)
    else:
        # Legacy compatibility - simple comparison as before but save instead of display
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        for model_name, result in results.items():
            plt.plot(result['batch_numbers'], 
                     result['mse_values'], 
                     label=f"{model_name}")
        plt.xlabel('Batch')
        plt.ylabel('MSE')
        plt.title('Mean Squared Error During Training')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        for model_name, result in results.items():
            plt.plot(result['batch_numbers'], 
                     result['rmse_values'], 
                     label=f"{model_name}")
        plt.xlabel('Batch')
        plt.ylabel('RMSE')
        plt.title('Root Mean Squared Error During Training')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        for model_name, result in results.items():
            plt.plot(result['batch_numbers'], 
                     result['train_losses'], 
                     label=f"{model_name}")
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        for model_name, result in results.items():
            plt.plot(result['batch_numbers'], 
                     result['val_losses'], 
                     label=f"{model_name}")
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/general_metrics_comparison.png", dpi=300)
        plt.close()
        
        # Bar chart of final metrics
        metrics = ['mse', 'rmse', 'mae']
        model_names = list(results.keys())
        
        x = np.arange(len(model_names))
        width = 0.25
        
        plt.figure(figsize=(12, 6))
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in model_names]
            plt.bar(x + i*width, values, width, label=metric.upper())
        
        plt.xlabel('Models')
        plt.ylabel('Error')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width, model_names)
        plt.legend()
        plt.savefig(f"{plot_dir}/metrics_bar_comparison.png", dpi=300)
        plt.close()
