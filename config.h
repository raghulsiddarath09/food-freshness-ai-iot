// ================================================================
// config.h — Credentials Template
// ⚠️  FILL IN YOUR VALUES. Never commit the real version to GitHub.
// ================================================================

#ifndef CONFIG_H
#define CONFIG_H

// WiFi
#define WIFI_SSID      "YOUR_WIFI_SSID"
#define WIFI_PASSWORD  "YOUR_WIFI_PASSWORD"

// Adafruit IO (https://io.adafruit.com → My Key)
#define MQTT_SERVER    "io.adafruit.com"
#define MQTT_PORT      1883
#define MQTT_USER      "YOUR_ADAFRUIT_USERNAME"
#define MQTT_KEY       "YOUR_ADAFRUIT_IO_KEY"

// Feed topics (replace YOUR_USERNAME)
#define MQTT_TOPIC_TEMP   "YOUR_USERNAME/feeds/temperature"
#define MQTT_TOPIC_HUM    "YOUR_USERNAME/feeds/humidity"
#define MQTT_TOPIC_GAS    "YOUR_USERNAME/feeds/gas-level"
#define MQTT_TOPIC_ALERT  "YOUR_USERNAME/feeds/alerts"

#endif
