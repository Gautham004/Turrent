#include <Arduino.h>

void setup() {
  Serial.begin(115200);
  delay(1000);

  for (int i = 1; i <= 70; i++) {
    Serial.printf("[%d/10] Hello World!\n", i);
    delay(500);
  }
  Serial.println("--- Done ---");
}

void loop() {
  // nothing
}
