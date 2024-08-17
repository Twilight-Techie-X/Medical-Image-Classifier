import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from albumentations import Compose, HorizontalFlip, Normalize, Resize
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def load_data(data_dir, dataset_type):
    images = []
    labels = []

    if dataset_type == 'brain_tumor_dataset1':
        for label in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, label)
            for img in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img)
                image = cv2.imread(img_path)
                images.append(image)
                labels.append(label)
    elif dataset_type == 'brain_tumor_dataset2':
        image_dir = os.path.join(data_dir, 'images')
        label_dir = os.path.join(data_dir, 'labels')
        for img in os.listdir(image_dir):
            img_path = os.path.join(image_dir, img)
            image = cv2.imread(img_path)
            images.append(image)
            label_path = os.path.join(label_dir, img.replace('.jpg', '.txt'))
            with open(label_path, 'r') as file:
                label = file.read().strip().split()
                labels.append(int(label[0]))  # assuming labels are integers in the text file
    else:  # bone_fracture
        for class_dir in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_dir)
            for subdir in ['train', 'test']:
                subdir_path = os.path.join(class_path, subdir)
                for img in os.listdir(subdir_path):
                    img_path = os.path.join(subdir_path, img)
                    image = cv2.imread(img_path)
                    images.append(image)
                    labels.append(class_dir)

    return np.array(images), np.array(labels)

def preprocess_image(image, target_size=(224, 224)):
    transform = Compose([
        Resize(target_size[0], target_size[1]),
        HorizontalFlip(),
        Normalize()
    ])
    return transform(image=image)['image']

def preprocess_data(images, labels, target_size=(224, 224)):
    X = np.array([preprocess_image(img, target_size) for img in images])
    X = X / 255.0
    y = to_categorical(labels)
    return X, y

def save_data(X_train, X_val, y_train, y_val, save_dir):
    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'y_val.npy'), y_val)

def main(data_dir, save_dir, dataset_type, test_size=0.2, random_state=42):
    images, labels = load_data(data_dir, dataset_type)
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    X_train, X_val, y_train, y_val = train_test_split(images, labels_encoded, test_size=test_size, random_state=random_state)
    X_train, y_train = preprocess_data(X_train, y_train)
    X_val, y_val = preprocess_data(X_val, y_val)
    save_data(X_train, X_val, y_train, y_val, save_dir)

if __name__ == "__main__":
    # For brain tumor
    brain_tumor_dir1 = '../data/raw/brain_tumor/dataset1'
    brain_tumor_dir2 = '../data/raw/brain_tumor/dataset2/train'
    save_dir_brain_tumor = '../data/processed/brain_tumor'
    main(brain_tumor_dir1, save_dir_brain_tumor, dataset_type='brain_tumor_dataset1')
    main(brain_tumor_dir2, save_dir_brain_tumor, dataset_type='brain_tumor_dataset2')
    
    # For bone fracture
    bone_fracture_dir = '../data/raw/bone_fracture'
    save_dir_bone_fracture = '../data/processed/bone_fracture'
    main(bone_fracture_dir, save_dir_bone_fracture, dataset_type='bone_fracture')