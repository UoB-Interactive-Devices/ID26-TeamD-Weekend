#include <Servo.h>
#include <Arduino.h>

const int NUM_SERVOS = 5;

Servo myServos[NUM_SERVOS];

const int servoPins[NUM_SERVOS] = {2,3,4,5,6};

const int CLOSE_ANGLE = 0;
const int OPEN_ANGLE = 180;

int angle = 180;

const int MOVE_DELAY = 250;
int update = 0;

int servo = 0;

void curle(){
  int now = millis();
  if(now - update > MOVE_DELAY){
    update = now;
    myServos[servo].write(angle);
    Serial.println(servo);
    servo++;
    if(servo > 4){
      servo = 0;
      if(angle == 180)angle = 0;
      else angle = 180;
    }
  }
}

void flat(){
  angle = 0;
  for (int i = 0; i < NUM_SERVOS; i++) {
    myServos[i].write(angle);
  }
}

void  fist(){
  angle = 180;
  for (int i = 0; i < NUM_SERVOS; i++) {
    myServos[i].write(angle);
  }
}

void setAll(int servoAngle){
  angle = servoAngle;
  for (int i = 0; i < NUM_SERVOS; i++) {
    myServos[i].write(servoAngle);
  }
}

void sweep(int servoNum, int servoAngle){
  if(angle < servoAngle){
    while (angle < servoAngle){
      angle ++;
      myServos[servoNum].write(angle);
      delay(5);
    }
  }
  else{
    while (angle > servoAngle){
      angle --;
      myServos[servoNum].write(angle);
      delay(5);
    }
  }
  
}

void setup() {
  Serial.begin(9600);
  angle = 0;
  for (int i = 0; i < NUM_SERVOS; i++) {
    myServos[i].attach(servoPins[i]);
    
    myServos[i].write(angle);
  }
  
  delay(1000); 
} 

void loop() {
  //sweep(0,180);
  //sweep(0,0);
  //flat();
}