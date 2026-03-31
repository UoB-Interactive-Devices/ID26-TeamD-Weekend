import os

# Suppress TensorFlow informational messages and oneDNN warnings
# Must be set before importing tensorflow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv
import pickle

import matplotlib.pyplot as plt
import numpy as np

# Import TensorFlow and Keras
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import callbacks, layers, models, regularizers

DATA_FILE = "processed_gesture_data.csv"
MODEL_PATH = "gesture_cnn_model.keras"
ENCODER_PATH = "label_encoder.pkl"
TOTAL_SENSORS = 91


def load_data_grouped_by_session(filename):
    X, y, sessions = [], [], []

    with open(filename, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row in reader:
            if len(row) >= TOTAL_SENSORS + 3:
                label = row[TOTAL_SENSORS]

                X.append([float(val) for val in row[:TOTAL_SENSORS]])
                y.append(label)
                sessions.append(row[TOTAL_SENSORS + 2])

    return np.array(X), np.array(y), np.array(sessions)


def format_for_cnn(X_flat):
    """
    Reshapes the flat 91-array into a 7x7x2 3D volume.
    Front matrix: 7x7. Back matrix: 6x7 (padded with zeros to 7x7).
    """
    front = X_flat[:, :49].reshape(-1, 7, 7)
    back = X_flat[:, 49:].reshape(-1, 6, 7)

    # Pad the back matrix with a row of zeros at the bottom to make it 7x7
    back_padded = np.pad(
        back, ((0, 0), (0, 1), (0, 0)), mode="constant", constant_values=0
    )

    X_cnn = np.stack([front, back_padded], axis=-1)
    return X_cnn


def main():
    print("Loading preprocessed data...")
    X_flat, y, session_groups = load_data_grouped_by_session(DATA_FILE)
    print(f"Loaded {len(X_flat)} samples.")

    print("Reshaping arrays into 7x7x2 matrices for convolutional processing...")
    X = format_for_cnn(X_flat)

    unique_sessions = np.unique(session_groups)
    print(f"Found {len(unique_sessions)} unique collection passes (chunks).")

    if len(unique_sessions) < 2:
        print("\nWARNING: You need at least 2 distinct passes to do a chunked split.")
        return

    # Encode labels mathematically
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    # Split strictly by session_id (chunk)
    print("\nSplitting data by session chunks to test on unseen hand placements...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_encoded, session_groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    train_chunks = np.unique(session_groups[train_idx])
    test_chunks = np.unique(session_groups[test_idx])

    print(f"Training on {len(train_chunks)} chunks ({len(X_train)} samples)")
    print(f"Testing on {len(test_chunks)} unseen chunks ({len(X_test)} samples)")

    # Build the Generalisable CNN Architecture
    print("\nBuilding Convolutional Neural Network...")
    model = models.Sequential(
        [
            # Updated syntax for input shape
            layers.Input(shape=(7, 7, 2)),
            # Data Augmentation/Regularisation: Add slight noise to prevent memorisation
            layers.GaussianNoise(0.05),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            # Add spatial dropout early to force reliance on multiple features
            layers.Dropout(0.2),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.Flatten(),
            # L2 Regularisation penalises the model if weights get too large
            layers.Dense(
                64, activation="relu", kernel_regularizer=regularizers.l2(0.01)
            ),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,  # Wait a bit longer before stopping
        restore_best_weights=True,
        verbose=1,
    )

    # Smoothly reduce learning rate if validation accuracy plateaus
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-5, verbose=1
    )

    # Train the network
    print("\nTraining CNN...")
    history = model.fit(
        X_train,
        y_train,
        epochs=40,  # Increased max epochs since we have heavier regularisation
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )

    # Evaluate
    print("\nEvaluating on hold-out chunks...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc * 100:.2f}%\n")

    # Generate detailed report
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Visualise the Confusion Matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=label_encoder.classes_
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Gesture Classification Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Save the model
    model.save(MODEL_PATH)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"CNN Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
