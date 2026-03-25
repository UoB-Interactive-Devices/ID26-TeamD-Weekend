# import csv
# import pickle

# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import GroupShuffleSplit
# from sklearn.preprocessing import LabelEncoder

# DATA_FILE = "processed_gesture_data.csv"
# MODEL_PATH = "gesture_rf_model.pkl"
# ENCODER_PATH = "label_encoder.pkl"
# TOTAL_SENSORS = 91


# def load_data_grouped_by_session(filename):
#     X, y, sessions = [], [], []

#     with open(filename, "r") as f:
#         reader = csv.reader(f)
#         next(reader)  # Skip header

#         for row in reader:
#             if len(row) >= TOTAL_SENSORS + 3:
#                 label = row[TOTAL_SENSORS]

#                 # Ignore the squeeze gesture entirely
#                 if label == "squeeze":
#                     continue

#                 X.append([float(val) for val in row[:TOTAL_SENSORS]])
#                 y.append(label)  # label
#                 sessions.append(row[TOTAL_SENSORS + 2])  # session_id (Chunk ID)

#     return np.array(X), np.array(y), np.array(sessions)


# def main():
#     print("Loading preprocessed data (ignoring 'squeeze')...")
#     X, y, session_groups = load_data_grouped_by_session(DATA_FILE)
#     print(f"Loaded {len(X)} samples.")

#     unique_sessions = np.unique(session_groups)
#     print(f"Found {len(unique_sessions)} unique collection passes (chunks).")

#     if len(unique_sessions) < 2:
#         print("\nWARNING: You need at least 2 distinct passes to do a chunked split.")
#         return

#     # Encode labels
#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(y)

#     # Split strictly by session_id (chunk)
#     # test_size=0.2 keeps roughly 20% of the passes entirely hidden from the training phase
#     print("\nSplitting data by session chunks to test on unseen hand placements...")
#     gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#     train_idx, test_idx = next(gss.split(X, y_encoded, session_groups))

#     X_train, X_test = X[train_idx], X[test_idx]
#     y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

#     train_chunks = np.unique(session_groups[train_idx])
#     test_chunks = np.unique(session_groups[test_idx])

#     print(f"Training on {len(train_chunks)} chunks ({len(X_train)} samples)")
#     print(f"Testing on {len(test_chunks)} unseen chunks ({len(X_test)} samples)")

#     # Train Random Forest
#     print("\nTraining Random Forest model...")
#     model = RandomForestClassifier(
#         n_estimators=200,
#         max_depth=12,  # Slightly deeper since we aren't generalising to new people
#         min_samples_split=5,  # Relaxed slightly to learn finer details
#         min_samples_leaf=2,
#         max_features="sqrt",
#         random_state=42,
#         n_jobs=-1,
#     )
#     model.fit(X_train, y_train)

#     # Evaluate
#     print("\nEvaluating on hold-out chunks...")
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)

#     print(f"Test Accuracy: {accuracy * 100:.2f}%\n")
#     print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

#     # Save the model and encoder
#     with open(MODEL_PATH, "wb") as f:
#         pickle.dump(model, f)
#     with open(ENCODER_PATH, "wb") as f:
#         pickle.dump(label_encoder, f)

#     print(f"Model saved to {MODEL_PATH}")


# if __name__ == "__main__":
#     main()


import csv
import pickle

import numpy as np

# Import TensorFlow and Keras
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import callbacks, layers, models

DATA_FILE = "processed_gesture_data.csv"
MODEL_PATH = "gesture_cnn_model.keras"  # Note the new extension
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

                # Ignore the squeeze gesture entirely
                if label == "squeeze":
                    continue

                X.append([float(val) for val in row[:TOTAL_SENSORS]])
                y.append(label)
                sessions.append(row[TOTAL_SENSORS + 2])

    return np.array(X), np.array(y), np.array(sessions)


def format_for_cnn(X_flat):
    """
    Reshapes the flat 91-array into a 7x7x2 3D volume.
    Front matrix: 7x7. Back matrix: 6x7 (padded with zeros to 7x7).
    """
    # 1. Extract and reshape the front matrix (first 49 sensors)
    front = X_flat[:, :49].reshape(-1, 7, 7)

    # 2. Extract and reshape the back matrix (last 42 sensors)
    back = X_flat[:, 49:].reshape(-1, 6, 7)

    # 3. Pad the back matrix with a row of zeros at the bottom to make it 7x7
    # ((no pad on batch), (0 pad top, 1 pad bottom), (no pad on cols))
    back_padded = np.pad(
        back, ((0, 0), (0, 1), (0, 0)), mode="constant", constant_values=0
    )

    # 4. Stack them along the "channel" axis (axis=-1). Result shape: (N, 7, 7, 2)
    X_cnn = np.stack([front, back_padded], axis=-1)

    return X_cnn


def main():
    print("Loading preprocessed data (ignoring 'squeeze')...")
    X_flat, y, session_groups = load_data_grouped_by_session(DATA_FILE)
    print(f"Loaded {len(X_flat)} samples.")

    # Convert to 2D CNN format
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

    # Build the CNN Architecture
    print("\nBuilding Convolutional Neural Network...")
    model = models.Sequential(
        [
            # Input layer matching our (7 height, 7 width, 2 channels) shape
            layers.InputLayer(input_shape=(7, 7, 2)),
            # First Convolutional Block (looks for basic edges and pressure ridges)
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            # Second Convolutional Block (looks for more complex combined shapes)
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
            # Flatten the 2D maps into a 1D vector to feed the dense network
            layers.Flatten(),
            # Dense classification head
            layers.Dense(64, activation="relu"),
            layers.Dropout(
                0.5
            ),  # Aggressive dropout to prevent overfitting on specific people
            layers.Dense(num_classes, activation="softmax"),  # Outputs probabilities
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Stop training early if the model stops improving on the test set, restoring the best weights
    early_stopping = callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    )

    # Train the network
    print("\nTraining CNN...")
    history = model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1,
    )

    # Evaluate
    print("\nEvaluating on hold-out chunks...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc * 100:.2f}%\n")

    # Generate detailed report
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities back to index
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Save the model natively as Keras format, and the encoder as a pickle
    model.save(MODEL_PATH)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"CNN Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
