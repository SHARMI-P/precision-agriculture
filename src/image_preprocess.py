# src/image_preprocess.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE   = 224
BATCH_SIZE = 32

def get_data_generators(
    data_dir = 'data/plantvillage_split'   # ← exact path after splitting
):
    train_datagen = ImageDataGenerator(
        rescale            = 1.0/255,
        rotation_range     = 25,
        width_shift_range  = 0.2,
        height_shift_range = 0.2,
        shear_range        = 0.15,
        zoom_range         = 0.2,
        horizontal_flip    = True,
        fill_mode          = 'nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1.0/255)

    train_gen = train_datagen.flow_from_directory(
        f'{data_dir}/train',
        target_size = (IMG_SIZE, IMG_SIZE),
        batch_size  = BATCH_SIZE,
        class_mode  = 'categorical'
    )

    val_gen = val_datagen.flow_from_directory(
        f'{data_dir}/val',
        target_size = (IMG_SIZE, IMG_SIZE),
        batch_size  = BATCH_SIZE,
        class_mode  = 'categorical',
        shuffle     = False
    )

    print(f"Total classes     : {train_gen.num_classes}")
    print(f"Training images   : {train_gen.samples}")
    print(f"Validation images : {val_gen.samples}")

    # Print all class names so you can verify
    print("\nClass names detected:")
    for name, idx in train_gen.class_indices.items():
        print(f"  [{idx:03d}] {name}")

    return train_gen, val_gen