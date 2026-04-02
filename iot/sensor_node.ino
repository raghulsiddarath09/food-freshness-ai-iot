// ============================================================
// AI & IoT Food Freshness System — ESP8266 Sensor Node
// Authors: Kirthic Madavan, Raghul Siddarth, Lavanyah
// SRMIST Final Year Project — 2025
// ============================================================

#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <DHT.h>
#include "config.h"   // WiFi & MQTT credentials (DO NOT share this file)

// ── Pin Definitions ──────────────────────────────────────────
#define DHT_PIN      D4      // DHT11 data pin
#define DHT_TYPE     DHT11
#define MQ135_PIN    A0      // MQ135 analog pin

// ── Sensor Interval ──────────────────────────────────────────
#define SEND_INTERVAL 10000  // Send data every 10 seconds

// ── Objects ──────────────────────────────────────────────────
DHT dht(DHT_PIN, DHT_TYPE);
WiFiClient espClient;
PubSubClient mqttClient(espClient);

unsigned long lastSendTime = 0;

// ── Setup ────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  dht.begin();

  connectWiFi();
  mqttClient.setServer(MQTT_SERVER, MQTT_PORT);

  Serial.println("✅ System Ready — Food Freshness Monitor");
}

// ── Main Loop ────────────────────────────────────────────────
void loop() {
  if (!mqttClient.connected()) {
    reconnectMQTT();
  }
  mqttClient.loop();

  unsigned long now = millis();
  if (now - lastSendTime >= SEND_INTERVAL) {
    lastSendTime = now;
    readAndPublishSensors();
  }
}

// ── Read Sensors & Publish to Cloud ──────────────────────────
void readAndPublishSensors() {
  float temperature = dht.readTemperature();
  float humidity    = dht.readHumidity();
  int   gasLevel    = analogRead(MQ135_PIN);

  // Validate DHT readings
  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("⚠️  DHT11 read failed. Skipping...");
    return;
  }

  // Apply moving average filter (simple 3-sample smoothing)
  static float tempHistory[3] = {0};
  static float humHistory[3]  = {0};
  static int   gasHistory[3]  = {0};
  static int   idx = 0;

  tempHistory[idx] = temperature;
  humHistory[idx]  = humidity;
  gasHistory[idx]  = gasLevel;
  idx = (idx + 1) % 3;

  float avgTemp = (tempHistory[0] + tempHistory[1] + tempHistory[2]) / 3.0;
  float avgHum  = (humHistory[0]  + humHistory[1]  + humHistory[2])  / 3.0;
  int   avgGas  = (gasHistory[0]  + gasHistory[1]  + gasHistory[2])  / 3;

  // Print to Serial Monitor
  Serial.println("────────────────────────────");
  Serial.print("🌡️  Temperature : "); Serial.print(avgTemp); Serial.println(" °C");
  Serial.print("💧 Humidity    : "); Serial.print(avgHum);  Serial.println(" %");
  Serial.print("💨 Gas Level   : "); Serial.println(avgGas);

  // Check spoilage thresholds
  checkAlerts(avgTemp, avgHum, avgGas);

  // Publish to Adafruit IO via MQTT
  char tempStr[10], humStr[10], gasStr[10];
  dtostrf(avgTemp, 4, 2, tempStr);
  dtostrf(avgHum,  4, 2, humStr);
  itoa(avgGas, gasStr, 10);

  mqttClient.publish(MQTT_TOPIC_TEMP,  tempStr);
  mqttClient.publish(MQTT_TOPIC_HUM,   humStr);
  mqttClient.publish(MQTT_TOPIC_GAS,   gasStr);

  Serial.println("📤 Data published to cloud.");
}

// ── Threshold Alert Check ─────────────────────────────────────
void checkAlerts(float temp, float hum, int gas) {
  if (temp > 30.0) {
    Serial.println("🚨 ALERT: Temperature too high! Spoilage risk.");
    mqttClient.publish(MQTT_TOPIC_ALERT, "HIGH_TEMP");
  }
  if (hum > 85.0) {
    Serial.println("🚨 ALERT: Humidity too high! Mold risk.");
    mqttClient.publish(MQTT_TOPIC_ALERT, "HIGH_HUMIDITY");
  }
  if (gas > 400) {
    Serial.println("🚨 ALERT: Gas levels elevated! Possible spoilage.");
    mqttClient.publish(MQTT_TOPIC_ALERT, "HIGH_GAS");
  }
}

// ── WiFi Connection ──────────────────────────────────────────
void connectWiFi() {
  Serial.print("📶 Connecting to WiFi: ");
  Serial.println(WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✅ WiFi connected!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n❌ WiFi connection failed. Restarting...");
    ESP.restart();
  }
}

// ── MQTT Reconnect ───────────────────────────────────────────
void reconnectMQTT() {
  while (!mqttClient.connected()) {
    Serial.print("🔌 Connecting to MQTT broker...");
    if (mqttClient.connect("FoodSensorNode", MQTT_USER, MQTT_KEY)) {
      Serial.println(" ✅ Connected!");
    } else {
      Serial.print(" ❌ Failed (rc=");
      Serial.print(mqttClient.state());
      Serial.println("). Retrying in 5s...");
      delay(5000);
    }
  }
}
