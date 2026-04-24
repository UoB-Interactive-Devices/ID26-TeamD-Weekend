from pathlib import Path
import platform

PORT = "COM5" if platform.system() == "Windows" else "/dev/ttyACM0"
BAUD_RATE = 2_000_000
TOTAL_SENSORS = 91

NUM_BASELINE_SAMPLES = 30
SMOOTHING_FRAMES = 3
MAX_EXPECTED_VALUE = 4000.0

FINE_TUNE_CLASSES = ["left", "up", "right", "down", "squeeze", "none"]
SAMPLES_PER_CLASS = 250
FINE_TUNE_EPOCHS = 4
LEARNING_RATE = 1e-4

STABLE_GESTURE_FRAMES = 20
STABLE_GESTURE_COOLDOWN_SECONDS = 1.0

BASE_MOVE_SPEED = 5.0
MAX_MOVE_SPEED = 20.0
ACCELERATION_RATE = 0.45
MOUSE_MOVE_DELAY = 0.06
CLICK_DELAY = 0.5
PHOTO_SORT_GESTURE_COOLDOWN_SECONDS = 0.5

DEMO_DIR = Path(__file__).resolve().parent.parent
ML_DIR = DEMO_DIR / "ml"
ASSETS_DIR = DEMO_DIR.parent.parent / "assets"
DATA_DIR = DEMO_DIR.parent.parent / "data"
MODEL_DIR = DEMO_DIR.parent.parent / "model"
MODEL_PATH = MODEL_DIR / "gesture_cnn_model.pth"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
FINE_TUNING_CSV = DATA_DIR / "fine_tuning_gesture_data.csv"
