#include <Arduino.h>

/*
 * serial_led_test.cpp
 * ===================
 * Blinks onboard LED when NUC detects a person.
 * No servos needed — pure serial comms test.
 * 
 * ESP32 onboard LED = GPIO 2
 */

#define LED_PIN 2

void setup() {
    Serial.begin(115200);
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);
    Serial.println("[ESP32] Ready. Waiting for NUC...");
}

void loop() {
    if (Serial.available()) {
        String packet = Serial.readStringUntil('\n');
        packet.trim();

        int cIdx = packet.indexOf('C');
        int fIdx = packet.indexOf('F');

        if (cIdx >= 0 && fIdx >= 0) {
            float conf = packet.substring(cIdx+1, fIdx).toFloat();
            int   fire = packet.substring(fIdx+1).toInt();

            if (conf > 0.5) {
                digitalWrite(LED_PIN, HIGH);
                delay(50);
                digitalWrite(LED_PIN, LOW);
                Serial.printf("[ESP32] Person detected! CONF:%.2f FIRE:%d\n", conf, fire);
            } else {
                digitalWrite(LED_PIN, LOW);
            }
        }
    }
}