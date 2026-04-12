# src/train_cnn.py

from image_preprocess import get_data_generators
from cnn_model import train_cnn
import os

os.makedirs('models', exist_ok=True)
os.makedirs('plots',  exist_ok=True)

print("Loading image data generators...")
train_gen, val_gen = get_data_generators(
    data_dir='data/plantvillage_split'
)

print("\nStarting CNN training...")
model, class_names = train_cnn(train_gen, val_gen)

print(f"\nCNN training complete!")
print(f"Model saved  : models/cnn_disease_model.h5")
print(f"Classes saved: models/class_names.pkl")
print(f"Classes      : {len(class_names)}")