const int FRONT_ROWS = 7;
const int FRONT_COLS = 7;
const int BACK_ROWS = 6;
const int BACK_COLS = 7;

// Original pins for the front matrix
const int FRONT_ROW_PINS[FRONT_ROWS] = {26, 27, 28, 29, 30, 31, 32}; 
const int FRONT_COL_PINS[FRONT_COLS] = {33, 34, 35, 36, 37, 38, 39};

// NEW: Pins for the back matrix (
const int BACK_ROW_PINS[BACK_ROWS] = {5,4,3,2,1,0}; 
const int BACK_COL_PINS[BACK_COLS] = {14,15,16,17,18,19,20};

// Buffer size increased to hold 91 four-digit numbers plus commas (~455 chars max)
char outputBuffer[512]; 

void setup() {
  Serial.begin(2000000); 
  
  // Setup Front Matrix Pins
  for (int i = 0; i < FRONT_ROWS; i++) {
    pinMode(FRONT_ROW_PINS[i], OUTPUT);
    digitalWrite(FRONT_ROW_PINS[i], LOW);
  }
  for (int i = 0; i < FRONT_COLS; i++) {
    pinMode(FRONT_COL_PINS[i], INPUT);
  }

  // Setup Back Matrix Pins
  for (int i = 0; i < BACK_ROWS; i++) {
    pinMode(BACK_ROW_PINS[i], OUTPUT);
    digitalWrite(BACK_ROW_PINS[i], LOW);
  }
  for (int i = 0; i < BACK_COLS; i++) {
    pinMode(BACK_COL_PINS[i], INPUT);
  }

  analogReadResolution(12);
  analogReadAveraging(1); 
}

void loop() {
  int bufferIndex = 0;

  // 1. Scan Front Matrix
  for (int rowIndex = 0; rowIndex < FRONT_ROWS; rowIndex++) {
    digitalWrite(FRONT_ROW_PINS[rowIndex], HIGH);
    delayMicroseconds(10); 

    for (int colIndex = 0; colIndex < FRONT_COLS; colIndex++) {
      int val = analogRead(FRONT_COL_PINS[colIndex]);
      bufferIndex += sprintf(&outputBuffer[bufferIndex], "%d,", val);
    }
    digitalWrite(FRONT_ROW_PINS[rowIndex], LOW);
  }

  // 2. Scan Back Matrix
  for (int rowIndex = 0; rowIndex < BACK_ROWS; rowIndex++) {
    digitalWrite(BACK_ROW_PINS[rowIndex], HIGH);
    delayMicroseconds(10); 

    for (int colIndex = 0; colIndex < BACK_COLS; colIndex++) {
      int val = analogRead(BACK_COL_PINS[colIndex]);
      
      // Omit the trailing comma ONLY on the absolute final reading of the back matrix
      if (rowIndex == BACK_ROWS - 1 && colIndex == BACK_COLS - 1) {
         bufferIndex += sprintf(&outputBuffer[bufferIndex], "%d", val);
      } else {
         bufferIndex += sprintf(&outputBuffer[bufferIndex], "%d,", val);
      }
    }
    digitalWrite(BACK_ROW_PINS[rowIndex], LOW);
  }
  
  // Send all 91 numbers (Front + Back) in a single USB packet
  Serial.println(outputBuffer);
}