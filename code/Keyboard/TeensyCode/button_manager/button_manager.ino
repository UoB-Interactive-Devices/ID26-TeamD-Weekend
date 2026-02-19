const int pins[] = {36, 35, 34, 33, 29, 30, 31, 32, 27, 28}; // Pins in use
const char* labels[] = {"a", "s", "e", "t", "n", "i", "o", "p", "v", "b"}; // ASETNIOP keys V and B are SHIFT and SPACE respectively
bool pressed[] = {false, false, false, false, false, false, false, false, false, false};
const int pinCount = 10; // Keyboard size'

int lastState[] = {0,0,0,0,0,0,0,0,0,0};
bool send = false;
bool stateChange = false;

char input[20];


void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  Serial.setTimeout(1);

  pinMode(LED_BUILTIN, OUTPUT);  

  for (int i = 0; i < pinCount; i++) {
    pinMode(pins[i], INPUT_PULLDOWN);
  }

  Serial.println("Board initialised...");
  strcpy(input,"");

}

void loop() {
  // put your main code here, to run repeatedly:
  for (int i = 0; i < pinCount; i++) {
    int buttonState = digitalRead(pins[i]);
    if (buttonState == HIGH && !pressed[i]) {
      strcat(input,labels[i]);
      pressed[i] = true;
      digitalWrite(LED_BUILTIN, HIGH);
    }
    if (buttonState != lastState[i]){
      stateChange = true;
    }
    lastState[i] = buttonState;
  }

  if(stateChange){
    stateChange = false;
    for (int i = 0; i < pinCount; i++) {
      int buttonState = digitalRead(pins[i]);
      if(buttonState == HIGH){
        send = false;
        break;
      }
      send = true;
    }
  }

  if (send) { // released
    Serial.println(input);
    strcpy(input,"");
    for (int p = 0; p < pinCount; p++) {
      pressed[p] = false;
    }
    send = false;
  }
  delay(250);
  digitalWrite(LED_BUILTIN, LOW);
}
