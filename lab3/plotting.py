import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_plot_dir(plot_dir='plots'):
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def plot_training_metrics(results, model_type, plot_dir='plots'):
    ensure_plot_dir(plot_dir)
    
    plt.figure(figsize=(15, 10))

    data_cutoff = 0.75 if model_type == 'Library' else 0

    plt.subplot(2, 2, 1)
    for model_name, result in results.items():
        if data_cutoff > 0:
            start_idx = int(len(result['batch_numbers']) * (1 - data_cutoff))
        else:
            start_idx = 0

        plt.plot(result['batch_numbers'][start_idx:], 
                 result['train_losses'][start_idx:], 
                 label=f"{model_name} (Train)")
        plt.plot(result['batch_numbers'][start_idx:], 
                 result['val_losses'][start_idx:], 
                 label=f"{model_name} (Val)", 
                 linestyle='--')

    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title(f'{model_type} Models - Training and Validation Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    for model_name, result in results.items():
        if data_cutoff > 0:
            start_idx = int(len(result['batch_numbers']) * (1 - data_cutoff))
        else:
            start_idx = 0
            
        plt.plot(result['batch_numbers'][start_idx:], 
                 result['mse_values'][start_idx:], 
                 label=f"{model_name}")

    plt.xlabel('Batch')
    plt.ylabel('MSE')
    plt.title(f'{model_type} Models - Mean Squared Error')
    plt.legend()

    plt.subplot(2, 2, 3)
    for model_name, result in results.items():
        if data_cutoff > 0:
            start_idx = int(len(result['batch_numbers']) * (1 - data_cutoff))
        else:
            start_idx = 0
            
        plt.plot(result['batch_numbers'][start_idx:], 
                 result['rmse_values'][start_idx:], 
                 label=f"{model_name}")

    plt.xlabel('Batch')
    plt.ylabel('RMSE')
    plt.title(f'{model_type} Models - Root Mean Squared Error')
    plt.legend()

    if 'mae_values' in next(iter(results.values())):
        plt.subplot(2, 2, 4)
        for model_name, result in results.items():
            if data_cutoff > 0:
                start_idx = int(len(result['batch_numbers']) * (1 - data_cutoff))
            else:
                start_idx = 0
                
            plt.plot(result['batch_numbers'][start_idx:], 
                     result['mae_values'][start_idx:],
                     label=f"{model_name}")

        plt.xlabel('Batch')
        plt.ylabel('MAE')
        plt.title(f'{model_type} Models - Mean Absolute Error')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{model_type.lower()}_training_metrics.png", dpi=300)
    plt.close()


def plot_final_comparison(tf_results, np_results, plot_dir='plots'):
    ensure_plot_dir(plot_dir)

    all_results = {**tf_results, **np_results}

    model_names = list(all_results.keys())
    metrics = ['mse', 'rmse', 'mae']

    x = np.arange(len(model_names))
    width = 0.25
    
    fig, ax = plt.figure(figsize=(15, 8)), plt.subplot()

    for i, metric in enumerate(metrics):
        values = [all_results[model][metric] for model in model_names]
        ax.bar(x + i*width - width, values, width, label=metric.upper())

    ax.set_xlabel('Models')
    ax.set_ylabel('Error')
    ax.set_title('Final Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/final_comparison.png", dpi=300)
    plt.close()
