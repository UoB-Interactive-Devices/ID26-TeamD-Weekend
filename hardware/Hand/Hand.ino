#include <TeensyThreads.h>
#include <Servo.h>

const int FRONT_ROWS = 7, FRONT_COLS = 7;
const int BACK_ROWS = 6, BACK_COLS = 7;
const int FRONT_ROW_PINS[FRONT_ROWS] = {26, 27, 28, 29, 30, 31, 32}; 
const int FRONT_COL_PINS[FRONT_COLS] = {33, 34, 35, 36, 37, 38, 39};
const int BACK_ROW_PINS[BACK_ROWS] = {5, 4, 3, 2, 1, 0}; 
const int BACK_COL_PINS[BACK_COLS] = {14, 15, 16, 17, 18, 19, 20};
char outputBuffer[512]; 

const int NUM_SERVOS = 5;
Servo myServos[NUM_SERVOS];
const int servoPins[NUM_SERVOS] = {6, 7, 8, 9, 10};
int currentAngle = 0;

void sensorTask() {
  while (1) {
    int bufferIndex = 0;

    for (int rowIndex = 0; rowIndex < FRONT_ROWS; rowIndex++) {
      digitalWrite(FRONT_ROW_PINS[rowIndex], HIGH);
      delayMicroseconds(10); 
      for (int colIndex = 0; colIndex < FRONT_COLS; colIndex++) {
        int val = analogRead(FRONT_COL_PINS[colIndex]);
        bufferIndex += sprintf(&outputBuffer[bufferIndex], "%d,", val);
      }
      digitalWrite(FRONT_ROW_PINS[rowIndex], LOW);
    }

    for (int rowIndex = 0; rowIndex < BACK_ROWS; rowIndex++) {
      digitalWrite(BACK_ROW_PINS[rowIndex], HIGH);
      delayMicroseconds(10); 
      for (int colIndex = 0; colIndex < BACK_COLS; colIndex++) {
        int val = analogRead(BACK_COL_PINS[colIndex]);
        if (rowIndex == BACK_ROWS - 1 && colIndex == BACK_COLS - 1) {
           bufferIndex += sprintf(&outputBuffer[bufferIndex], "%d", val);
        } else {
           bufferIndex += sprintf(&outputBuffer[bufferIndex], "%d,", val);
        }
      }
      digitalWrite(BACK_ROW_PINS[rowIndex], LOW);
    }
    
    Serial.println(outputBuffer);
    threads.yield();
  }
}

void setAll(int angle) {
  currentAngle = angle;
  for (int i = 0; i < NUM_SERVOS; i++) {
    myServos[i].write(angle);
  }
}

void curle() {
  for (int s = 0; s < NUM_SERVOS; s++) {
    myServos[s].write(180);
    delay(250); 
  }
  delay(500);
  for (int s = 0; s < NUM_SERVOS; s++) {
    myServos[s].write(0);
    delay(250);
  }
}

void servoTask() {
  while (1) {
    if (Serial.available() > 0) {
      char cmd = Serial.read();

      if (cmd == 'S' || cmd == 's') {
        for (int i = 0; i < NUM_SERVOS; i++) {
          int targetAngle = Serial.parseInt(); 
          
          targetAngle = constrain(targetAngle, 0, 180);
          
          myServos[i].write(targetAngle);
        }
      } 
      else if (cmd == 'F') setAll(0);
      else if (cmd == 'O') setAll(180);
    }
    threads.yield();
  }
}

void setup() {
  Serial.begin(2000000); 
  
  for (int i = 0; i < FRONT_ROWS; i++) { pinMode(FRONT_ROW_PINS[i], OUTPUT); digitalWrite(FRONT_ROW_PINS[i], LOW); }
  for (int i = 0; i < FRONT_COLS; i++) { pinMode(FRONT_COL_PINS[i], INPUT); }
  for (int i = 0; i < BACK_ROWS; i++) { pinMode(BACK_ROW_PINS[i], OUTPUT); digitalWrite(BACK_ROW_PINS[i], LOW); }
  for (int i = 0; i < BACK_COLS; i++) { pinMode(BACK_COL_PINS[i], INPUT); }

  analogReadResolution(12);
  analogReadAveraging(1); 

  for (int i = 0; i < NUM_SERVOS; i++) {
    myServos[i].attach(servoPins[i]);
    myServos[i].write(180);
  }

  threads.addThread(sensorTask);
  threads.addThread(servoTask);
}

void loop() {
}