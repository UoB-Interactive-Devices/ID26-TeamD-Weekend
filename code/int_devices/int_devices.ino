const int pins[] = {25, 26, 27, 28, 29, 30, 31, 32}; // Pins in use
const char* labels[] = {"A", "S", "E", "T", "N", "I", "O", "P"}; // ASETNIOP keys
const int pinCount = 8; // Keyboard size

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  Serial.setTimeout(1);

  pinMode(LED_BUILTIN, OUTPUT);  

  for (int i = 0; i < pinCount; i++) {
    pinMode(pins[i], INPUT_PULLDOWN);
  }

}

void loop() {
  // put your main code here, to run repeatedly:
  for (int i = 0; i < pinCount; i++) {
    if (digitalRead(pins[i]) == HIGH) {
      Serial.println(labels[i]);
      digitalWrite(LED_BUILTIN, HIGH);
    }
  }
  delay(250);
  digitalWrite(LED_BUILTIN, LOW);
}
