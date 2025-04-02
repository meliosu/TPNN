import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_metrics(history, implementation_name, save_path=None):
    plt.figure(figsize=(15, 5))
    
    if implementation_name == 'tensorflow':
        batch_history = history['batch_history']
        epoch_history = history['epoch_history']
        
        plt.subplot(1, 3, 1)
        plt.plot(batch_history['batch'], batch_history['loss'], 'b-', label='Training Loss')
        plt.plot(batch_history['batch'], batch_history['val_loss'], 'r-', label='Validation Loss')
        plt.title('Loss over Training')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(batch_history['batch'], batch_history['accuracy'], 'g-', label='Training Accuracy')
        plt.plot(batch_history['batch'], batch_history['val_accuracy'], 'm-', label='Validation Accuracy')
        plt.title('Accuracy over Training')
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        epochs = range(1, len(epoch_history['accuracy']) + 1)
        plt.plot(epochs, epoch_history['accuracy'], 'g-o', label='Training Accuracy')
        plt.plot(epochs, epoch_history['val_accuracy'], 'm-o', label='Validation Accuracy')
        plt.title('Per-Epoch Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    else:
        plt.subplot(1, 3, 1)
        plt.plot(history['batch'], history['batch_loss'], 'b-', label='Training Loss')
        plt.plot(history['batch'], history['batch_val_loss'], 'r-', label='Validation Loss')
        plt.title('Loss over Training')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(history['batch'], history['batch_accuracy'], 'g-', label='Training Accuracy')
        plt.plot(history['batch'], history['batch_val_accuracy'], 'm-', label='Validation Accuracy')
        plt.title('Accuracy over Training')
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        epochs = range(1, len(history['accuracy']) + 1)
        plt.plot(epochs, history['accuracy'], 'g-o', label='Training Accuracy')
        plt.plot(epochs, history['val_accuracy'], 'm-o', label='Validation Accuracy')
        plt.title('Per-Epoch Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_{implementation_name}_metrics.png")
    plt.show()

def plot_confusion_matrix(model, X_test, y_test, implementation_name, save_path=None):
    plt.figure(figsize=(10, 8))
    
    if implementation_name == 'tensorflow':
        y_pred = model.predict(X_test)
    else:
        y_pred = model.forward(X_test)
    
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.title(f'Confusion Matrix - {implementation_name.capitalize()}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if save_path:
        plt.savefig(f"{save_path}_{implementation_name}_confusion.png")
    plt.show()

def plot_misclassified(model, X_test, y_test, implementation_name, save_path=None):
    if implementation_name == 'tensorflow':
        y_pred = model.predict(X_test)
    else:
        y_pred = model.forward(X_test)
    
    y_true = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    misclassified_idx = np.where(y_true != y_pred_labels)[0]
    
    if len(misclassified_idx) == 0:
        print("No misclassified samples found.")
        return
    
    num_samples = min(16, len(misclassified_idx))
    samples_idx = misclassified_idx[:num_samples]
    
    rows = int(np.ceil(num_samples / 4))
    cols = min(4, num_samples)
    
    plt.figure(figsize=(15, 3 * rows))
    for i, idx in enumerate(samples_idx):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f'True: {y_true[idx]}, Pred: {y_pred_labels[idx]}')
        plt.axis('off')
    
    plt.suptitle(f'Misclassified Digits - {implementation_name.capitalize()}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(f"{save_path}_{implementation_name}_misclassified.png")
    plt.show()

def visualize_results(tf_model, manual_model, tf_history, manual_history, X_test, y_test, save_path=None):
    plot_metrics(tf_history, 'tensorflow', save_path)
    plot_confusion_matrix(tf_model, X_test, y_test, 'tensorflow', save_path)
    plot_misclassified(tf_model, X_test, y_test, 'tensorflow', save_path)
    
    plot_metrics(manual_history, 'manual', save_path)
    plot_confusion_matrix(manual_model, X_test, y_test, 'manual', save_path)
    plot_misclassified(manual_model, X_test, y_test, 'manual', save_path)
