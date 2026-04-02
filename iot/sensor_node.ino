// ================================================================
// sensor_node.ino — ESP8266 IoT Sensor Node
// AI & IoT Food Freshness & Safety System
// SRMIST Final Year Project 2025
// Authors: Kirthic Madavan, Raghul Siddarth, Lavanyah
// ================================================================
//
// Hardware:
//   - ESP8266 NodeMCU
//   - DHT11 (Pin D4)  → Temperature & Humidity
//   - MQ135  (Pin A0) → Gas / Air Quality
//
// Libraries needed (install via Arduino Library Manager):
//   - ESP8266WiFi      (built-in)
//   - PubSubClient     by Nick O'Leary
//   - DHT sensor library by Adafruit
//
// Fill in your credentials in config.h before uploading.
// ================================================================

#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <DHT.h>
#include "config.h"

// ── Pin & Sensor Setup ────────────────────────────────────────
#define DHT_PIN      D4
#define DHT_TYPE     DHT11
#define MQ135_PIN    A0

// ── Timing ───────────────────────────────────────────────────
#define SEND_INTERVAL   10000   // ms between sensor readings
#define RECONNECT_DELAY  5000   // ms between reconnect attempts

// ── Alert Thresholds ─────────────────────────────────────────
#define TEMP_THRESHOLD   30.0   // °C — spoilage risk above this
#define HUM_THRESHOLD    85.0   // %  — mold risk above this
#define GAS_THRESHOLD    400    // ppm — spoilage indicator

// ── Moving Average Filter (window = 5) ───────────────────────
#define FILTER_SIZE 5
float tempHistory[FILTER_SIZE] = {0};
float humHistory[FILTER_SIZE]  = {0};
int   gasHistory[FILTER_SIZE]  = {0};
int   filterIdx = 0;
bool  filterFull = false;

// ── Objects ──────────────────────────────────────────────────
DHT            dht(DHT_PIN, DHT_TYPE);
WiFiClient     espClient;
PubSubClient   mqtt(espClient);
unsigned long  lastSendTime = 0;

// ── Setup ─────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("\n\n🍎 Food Freshness IoT Node — Starting...");

  dht.begin();
  pinMode(MQ135_PIN, INPUT);

  connectWiFi();
  mqtt.setServer(MQTT_SERVER, MQTT_PORT);

  Serial.println("✅ System ready!\n");
}

// ── Main Loop ─────────────────────────────────────────────────
void loop() {
  if (!mqtt.connected()) reconnectMQTT();
  mqtt.loop();

  if (millis() - lastSendTime >= SEND_INTERVAL) {
    lastSendTime = millis();
    readAndPublish();
  }
}

// ── Read Sensors & Publish ────────────────────────────────────
void readAndPublish() {
  float rawTemp = dht.readTemperature();
  float rawHum  = dht.readHumidity();
  int   rawGas  = analogRead(MQ135_PIN);

  if (isnan(rawTemp) || isnan(rawHum)) {
    Serial.println("⚠️  DHT11 read error — skipping cycle.");
    return;
  }

  // Apply moving average filter
  tempHistory[filterIdx] = rawTemp;
  humHistory[filterIdx]  = rawHum;
  gasHistory[filterIdx]  = rawGas;
  filterIdx = (filterIdx + 1) % FILTER_SIZE;
  if (filterIdx == 0) filterFull = true;

  int   count   = filterFull ? FILTER_SIZE : filterIdx;
  float avgTemp = 0, avgHum = 0;
  int   avgGas  = 0;

  for (int i = 0; i < count; i++) {
    avgTemp += tempHistory[i];
    avgHum  += humHistory[i];
    avgGas  += gasHistory[i];
  }
  avgTemp /= count;
  avgHum  /= count;
  avgGas  /= count;

  // Serial output
  Serial.println("─────────────────────────────");
  Serial.printf("🌡️  Temperature : %.1f °C\n",  avgTemp);
  Serial.printf("💧 Humidity    : %.1f %%\n",   avgHum);
  Serial.printf("💨 Gas Level   : %d ppm\n",    avgGas);

  // Publish to MQTT
  char buf[16];

  dtostrf(avgTemp, 5, 1, buf);
  mqtt.publish(MQTT_TOPIC_TEMP, buf, true);

  dtostrf(avgHum, 5, 1, buf);
  mqtt.publish(MQTT_TOPIC_HUM, buf, true);

  itoa(avgGas, buf, 10);
  mqtt.publish(MQTT_TOPIC_GAS, buf, true);

  // Check thresholds & send alerts
  checkAlerts(avgTemp, avgHum, avgGas);

  Serial.println("📤 Published to Adafruit IO.");
}

// ── Threshold Alert Check ─────────────────────────────────────
void checkAlerts(float temp, float hum, int gas) {
  bool alert = false;

  if (temp > TEMP_THRESHOLD) {
    Serial.printf("🚨 ALERT: Temp %.1f°C > %.1f°C threshold!\n", temp, TEMP_THRESHOLD);
    mqtt.publish(MQTT_TOPIC_ALERT, "HIGH_TEMP");
    alert = true;
  }
  if (hum > HUM_THRESHOLD) {
    Serial.printf("🚨 ALERT: Humidity %.1f%% > %.1f%% threshold!\n", hum, HUM_THRESHOLD);
    mqtt.publish(MQTT_TOPIC_ALERT, "HIGH_HUMIDITY");
    alert = true;
  }
  if (gas > GAS_THRESHOLD) {
    Serial.printf("🚨 ALERT: Gas %d ppm > %d ppm threshold!\n", gas, GAS_THRESHOLD);
    mqtt.publish(MQTT_TOPIC_ALERT, "HIGH_GAS");
    alert = true;
  }
  if (!alert) {
    mqtt.publish(MQTT_TOPIC_ALERT, "OK");
  }
}

// ── WiFi Connection ───────────────────────────────────────────
void connectWiFi() {
  Serial.printf("📶 Connecting to WiFi: %s ", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries < 30) {
    delay(500);
    Serial.print(".");
    tries++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n✅ WiFi connected! IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\n❌ WiFi failed. Restarting in 5s...");
    delay(5000);
    ESP.restart();
  }
}

// ── MQTT Reconnect ────────────────────────────────────────────
void reconnectMQTT() {
  while (!mqtt.connected()) {
    Serial.print("🔌 Connecting MQTT...");
    String clientId = "FoodNode-" + String(ESP.getChipId(), HEX);

    if (mqtt.connect(clientId.c_str(), MQTT_USER, MQTT_KEY)) {
      Serial.println(" ✅ Connected!");
    } else {
      Serial.printf(" ❌ Failed (rc=%d). Retry in 5s...\n", mqtt.state());
      delay(RECONNECT_DELAY);
    }
  }
}
