import pickle
import time

import numpy as np
import serial
import torch
from train_model import GestureClassifier

PORT = "COM5"
BAUD_RATE = 115200
MODEL_PATH = "gesture_model.pth"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"


def load_model_and_preprocessing():
    """Load the trained model, scaler, and label encoder."""
    # Load scaler
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    
    # Load label encoder
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    
    # Load model
    num_classes = len(label_encoder.classes_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GestureClassifier(input_size=20, hidden_size=64, num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    return model, scaler, label_encoder, device


def classify_live():
    """Read sensor data from Arduino and classify gestures in real-time."""
    try:
        # Load model and preprocessing
        print("Loading model...")
        model, scaler, label_encoder, device = load_model_and_preprocessing()
        print(f"Model loaded successfully!")
        print(f"Classes: {label_encoder.classes_}")
        print(f"Using device: {device}")
        
        # Connect to Arduino
        print(f"\nConnecting to {PORT}...")
        ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print("Connected! Starting classification...\n")
        print("="*50)
        
        while True:
            line = ser.readline().decode("utf-8").strip()
            
            if line:
                try:
                    # Parse sensor data
                    data_point = [int(val) for val in line.split(",")]
                    
                    if len(data_point) == 20:
                        # Preprocess
                        data_array = np.array(data_point).reshape(1, -1)
                        data_scaled = scaler.transform(data_array)
                        
                        # Convert to tensor
                        data_tensor = torch.FloatTensor(data_scaled).to(device)
                        
                        # Predict
                        with torch.no_grad():
                            output = model(data_tensor)
                            probabilities = torch.softmax(output, dim=1)
                            confidence, predicted = torch.max(probabilities, 1)
                        
                        # Decode prediction
                        gesture = label_encoder.inverse_transform([predicted.cpu().numpy()[0]])[0]
                        conf = confidence.item() * 100
                        
                        # Display result
                        print(f"Gesture: {gesture.upper():8s} | Confidence: {conf:5.1f}%", end="\r")
                
                except ValueError:
                    continue
    
    except KeyboardInterrupt:
        print("\n\nStopping classification...")
    
    except FileNotFoundError as e:
        print(f"Error: Could not find model files. Please train the model first.")
        print(f"Missing file: {e.filename}")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        if "ser" in locals():
            ser.close()
            print("Serial connection closed.")


if __name__ == "__main__":
    classify_live()
