#include <Servo.h>

const int NUM_SERVOS = 5;

Servo myServos[NUM_SERVOS];

const int servoPins[NUM_SERVOS] = {2,3,4,5,6};

const int CLOSE_ANGLE = 0;
const int OPEN_ANGLE = 180;

const int MOVE_DELAY = 1500;

void setup() {
  Serial.begin(9600);
  for (int i = 0; i < NUM_SERVOS; i++) {
    myServos[i].attach(servoPins[i]);
    
    myServos[i].write(OPEN_ANGLE);
  }
  
  delay(1000); 
} 

void loop() {
  // for (int i = 0; i < NUM_SERVOS; i++) {
  //   Serial.println(i);
    
  //   myServos[i].write(CLOSE_ANGLE);
  //   delay(MOVE_DELAY);

  //   myServos[i].write(OPEN_ANGLE);
  //   delay(MOVE_DELAY);
    
  // }
  for (int i = 0; i < NUM_SERVOS; i++) {
    myServos[i].write(CLOSE_ANGLE);
  }
  delay(1000);
  for (int i = 0; i < NUM_SERVOS; i++) {
    for (int j=0; j<3; j++) {
      myServos[i].write(OPEN_ANGLE);
      delay(200);
      myServos[i].write(CLOSE_ANGLE);
      delay(200);
    }
    delay(1000);
  }
  for (int i = 0; i < NUM_SERVOS; i++) {
    myServos[i].write(OPEN_ANGLE);    
  }
  delay(1000);
}