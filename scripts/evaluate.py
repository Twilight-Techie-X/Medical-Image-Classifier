import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def load_data(data_dir):
    X_val = np.load(f'{data_dir}/X_val.npy')
    y_val = np.load(f'{data_dir}/y_val.npy')
    return X_val, y_val

def evaluate_model(model_path, data_dir):
    model = load_model(model_path)
    X_val, y_val = load_data(data_dir)
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    return model.history

def plot_training_history(history, dataset_type):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{dataset_type.capitalize()} Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{dataset_type.capitalize()} Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()

if __name__ == "__main__":
    datasets = ['brain_tumor', 'bone_fracture']
    
    for dataset_type in datasets:
        print(f"Evaluating {dataset_type} model...")
        model_path = f'../models/{dataset_type}_cnn_model.h5'
        data_dir = f'../data/processed/{dataset_type}'
        
        history = evaluate_model(model_path, data_dir)
        plot_training_history(history, dataset_type)