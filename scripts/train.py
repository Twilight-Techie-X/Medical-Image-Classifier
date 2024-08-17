import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model

def load_data(data_dir):
    X_train = np.load(f'{data_dir}/X_train.npy')
    X_val = np.load(f'{data_dir}/X_val.npy')
    y_train = np.load(f'{data_dir}/y_train.npy')
    y_val = np.load(f'{data_dir}/y_val.npy')
    return X_train, X_val, y_train, y_val

def train_model(X_train, y_train, X_val, y_val, save_path, epochs=10, batch_size=32):
    model = create_model(input_shape=X_train.shape[1:], num_classes=y_train.shape[1])
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator()

    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=val_datagen.flow(X_val, y_val),
        epochs=epochs
    )
    model.save(save_path)
    return history

if __name__ == "__main__":
    # For brain tumor
    data_dir_brain_tumor = '../data/processed/brain_tumor'
    save_path_brain_tumor = '../models/cnn_model_brain_tumor.h5'
    X_train_bt, X_val_bt, y_train_bt, y_val_bt = load_data(data_dir_brain_tumor)
    train_model(X_train_bt, y_train_bt, X_val_bt, y_val_bt, save_path_brain_tumor)
    
    # For bone fracture
    data_dir_bone_fracture = '../data/processed/bone_fracture'
    save_path_bone_fracture = '../models/cnn_model_bone_fracture.h5'
    X_train_bf, X_val_bf, y_train_bf, y_val_bf = load_data(data_dir_bone_fracture)
    train_model(X_train_bf, y_train_bf, X_val_bf, y_val_bf, save_path_bone_fracture)