const int ROWS = 7;
const int COLS = 7;

// OUTPUTS
const int ROW_PINS[ROWS] = {7,8,9,10,11,12}; 

// INPUTS - with resistors to ground
const int COL_PINS[COLS] = {14,15,16,17,18,19};

void setup() {
  Serial.begin(115200);
  
  for (int i = 0; i < ROWS; i++) {
    pinMode(ROW_PINS[i], OUTPUT);
    digitalWrite(ROW_PINS[i], LOW);
  }

  for (int i = 0; i < COLS; i++) {
    pinMode(COL_PINS[i], INPUT);
  }

  analogReadResolution(12);
}

void loop() {
  for (int rowIndex = 0; rowIndex < ROWS; rowIndex++) {
    digitalWrite(ROW_PINS[rowIndex], HIGH);
    delayMicroseconds(50); 

    for (int colIndex = 0; colIndex < COLS; colIndex++) {
      int val = analogRead(COL_PINS[colIndex]);
      Serial.print(val);
      
      if (!(rowIndex == ROWS - 1 && colIndex == COLS - 1)) {
        Serial.print(",");
      }
    }
    digitalWrite(ROW_PINS[rowIndex], LOW);
  }
  
  Serial.println();
  
  delay(100);
}