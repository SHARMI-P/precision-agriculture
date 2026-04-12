# src/cnn_model.py

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
import matplotlib.pyplot as plt
import pickle, numpy as np

IMG_SIZE = 224

def build_model(num_classes):
    """
    MobileNetV2 pretrained base + custom head.
    num_classes is detected automatically from your dataset folders.
    For this dataset it will be ~70 classes.
    """
    base = MobileNetV2(
        input_shape = (IMG_SIZE, IMG_SIZE, 3),
        include_top = False,
        weights     = 'imagenet'
    )
    base.trainable = False   # freeze pretrained layers

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
        loss      = 'categorical_crossentropy',
        metrics   = ['accuracy']
    )

    print(f"Model built for {num_classes} classes")
    model.summary()
    return model, base


def train_cnn(train_gen, val_gen):
    num_classes = train_gen.num_classes
    model, base = build_model(num_classes)

    # Save class names for prediction and app
    class_names = list(train_gen.class_indices.keys())
    with open('models/class_names.pkl', 'wb') as f:
        pickle.dump(class_names, f)
    print(f"Saved {len(class_names)} class names")

    callbacks = [
        EarlyStopping(
            monitor              = 'val_accuracy',
            patience             = 5,
            restore_best_weights = True,
            verbose              = 1
        ),
        ModelCheckpoint(
            'models/cnn_disease_model.h5',
            monitor        = 'val_accuracy',
            save_best_only = True,
            verbose        = 1
        ),
        ReduceLROnPlateau(
            monitor  = 'val_loss',
            factor   = 0.5,
            patience = 3,
            min_lr   = 1e-7,
            verbose  = 1
        )
    ]

    # ── Phase 1: Train head only (base frozen) ──
    print("\n=== Phase 1: Training classification head ===")
    h1 = model.fit(
        train_gen,
        epochs          = 20,
        validation_data = val_gen,
        callbacks       = callbacks,
        verbose         = 1
    )

    # ── Phase 2: Fine-tune last 40 layers of base ──
    print("\n=== Phase 2: Fine-tuning ===")
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss      = 'categorical_crossentropy',
        metrics   = ['accuracy']
    )

    h2 = model.fit(
        train_gen,
        epochs          = 10,
        validation_data = val_gen,
        callbacks       = callbacks,
        verbose         = 1
    )

    # ── Evaluate ──
    print("\n=== Final Evaluation ===")
    loss, acc = model.evaluate(val_gen, verbose=1)
    print(f"Validation Accuracy : {acc*100:.2f}%")
    print(f"Validation Loss     : {loss:.4f}")

    plot_history(h1, h2)
    return model, class_names


def plot_history(h1, h2):
    acc     = h1.history['accuracy']     + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss    = h1.history['loss']         + h2.history['loss']
    val_loss= h1.history['val_loss']     + h2.history['val_loss']
    split   = len(h1.history['accuracy'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(acc,     label='Train Accuracy')
    ax1.plot(val_acc, label='Val Accuracy')
    ax1.axvline(x=split-1, color='gray',
                linestyle='--', label='Fine-tune start')
    ax1.set_title('Accuracy over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(loss,     label='Train Loss')
    ax2.plot(val_loss, label='Val Loss')
    ax2.axvline(x=split-1, color='gray',
                linestyle='--', label='Fine-tune start')
    ax2.set_title('Loss over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('plots/cnn_training_curves.png', dpi=150)
    plt.show()
    print("Training curves saved to plots/cnn_training_curves.png")

    # src/cnn_model.py (continued)

from tensorflow.keras.preprocessing import image as kimage

def predict_disease(img_path, model=None, class_names=None):
    if model is None:
        model = tf.keras.models.load_model('models/cnn_disease_model.h5')
    if class_names is None:
        class_names = pickle.load(open('models/class_names.pkl', 'rb'))

    img       = kimage.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr       = kimage.img_to_array(img) / 255.0
    arr       = np.expand_dims(arr, axis=0)

    preds     = model.predict(arr, verbose=0)
    idx       = np.argmax(preds[0])
    confidence= float(preds[0][idx]) * 100

    raw_class = class_names[idx]

    # Class names follow pattern: CropName__Disease
    # e.g. "Apple__black_rot", "Corn__healthy"
    parts      = raw_class.split('__')
    crop       = parts[0].replace('_', ' ')
    condition  = parts[1].replace('_', ' ') if len(parts) > 1 else 'unknown'
    is_healthy = 'healthy' in condition.lower()

    # Top 3 predictions for display
    top3_idx  = np.argsort(preds[0])[::-1][:3]
    top3      = [
        {
            'class'     : class_names[i].replace('__', ' — '),
            'confidence': round(float(preds[0][i]) * 100, 2)
        }
        for i in top3_idx
    ]

    result = {
        'raw_class'  : raw_class,
        'crop'       : crop,
        'condition'  : condition,
        'is_healthy' : is_healthy,
        'confidence' : round(confidence, 2),
        'risk_score' : 0.0 if is_healthy else round(1 - (confidence/100), 3),
        'top3'       : top3
    }

    print(f"\nImage     : {img_path}")
    print(f"Crop      : {crop}")
    print(f"Condition : {condition}")
    print(f"Confidence: {confidence:.1f}%")
    print(f"Status    : {'Healthy' if is_healthy else 'DISEASED'}")
    print(f"Top 3     :")
    for t in top3:
        print(f"  {t['class']:<40} {t['confidence']}%")

    return result