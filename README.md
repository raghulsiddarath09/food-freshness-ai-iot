# 🥦 AI & IoT Solutions for Food Freshness & Safety
📄 **[View Full Project Report](docs/project_report.pdf)**

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-97.25%25-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Latency-60ms-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Platform-AWS%20%7C%20Adafruit%20IO-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Published-ICRCET--2025-red?style=for-the-badge"/>
</p>

<p align="center">
  <b>A real-time AI + IoT system for quality control of perishable goods — fruits, vegetables, and meat.</b><br/>
  Detects spoilage using CNN-based image classification and IoT environmental sensors.
</p>

---

## 📖 About the Project

This is the final year B.Tech project submitted to **SRM Institute of Science and Technology, Kattankulathur (2025)**, accepted and published at the **14th ICRCET-2025 International Conference (Scopus Indexed)** held in Bangalore, India.

Food spoilage causes over **$1 trillion** in losses annually. Our system tackles this using:
- 🤖 **AI (CNN)** — classifies food freshness from images
- 📡 **IoT Sensors** — monitors temperature, humidity, and gas levels in real time
- ☁️ **Cloud Dashboard** — visualizes data and sends alerts instantly

---

## 👥 Team

| Name | Register No. | Role |
|------|-------------|------|
| Kirthic Madavan S | RA2111003011196 | AI Model & Integration |
| Raghul Siddarth C | RA2111003011209 | IoT Hardware & Sensors |
| Lavanyah B | RA2111003011181 | Dashboard & Cloud |

**Guide:** Dr. Anbazhagu U.V — Assistant Professor, Dept. of Computing Technologies, SRMIST

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────┐
│                    INPUT LAYER                        │
│  📷 Camera Image  +  🌡️ DHT11  +  💨 MQ135 Sensor   │
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│               ESP8266 NodeMCU (MQTT)                 │
│         Sends data to cloud every 10 seconds         │
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│              AI PROCESSING LAYER (Cloud)             │
│   CNN Model (ShuffleNet/SqueezeNet) classifies:      │
│   Fresh  |  Slightly Spoiled  |  Spoiled             │
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│           DASHBOARD (Adafruit IO / Flask)            │
│   📊 Live graphs  |  🚨 Alerts  |  📥 CSV Export    │
└──────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
food-freshness-ai-iot/
│
├── 📂 models/               # CNN model code (Python + TensorFlow)
│   ├── cnn_baseline.py      # Basic CNN (Sprint I)
│   ├── squeezenet.py        # SqueezeNet architecture
│   ├── shufflenet.py        # ShuffleNet architecture
│   └── train.py             # Training script
│
├── 📂 iot/                  # ESP8266 Arduino code
│   ├── sensor_node.ino      # Main IoT sensor code
│   └── config.h             # WiFi & MQTT credentials (template)
│
├── 📂 dashboard/            # Flask web dashboard
│   ├── app.py               # Flask application
│   └── templates/           # HTML templates
│
├── 📂 dataset/              # Dataset structure guide
│   └── sample_images/       # Sample food images
│
├── 📂 docs/                 # Project documentation
│   ├── project_report.pdf   # Full B.Tech project report
│   └── research_paper.pdf   # ICRCET-2025 published paper
│
├── 📂 results/              # Performance graphs & confusion matrix
│
├── requirements.txt         # Python dependencies
├── .gitignore               # Files to ignore
└── README.md                # This file
```

---

## ⚙️ Hardware Components

| Component | Purpose |
|-----------|---------|
| ESP8266 NodeMCU | Wi-Fi microcontroller — sends sensor data to cloud |
| DHT11 Sensor | Measures temperature (±5% accuracy) and humidity |
| MQ135 Gas Sensor | Detects ammonia, CO₂ — indicators of spoilage |
| Raspberry Pi Camera / Webcam | Captures food images for CNN analysis |
| Breadboard + Jumper Wires | Circuit connections |

---

## 🧠 AI Model Performance

| Method | Accuracy | Latency | Robustness |
|--------|----------|---------|------------|
| **Proposed (AI+IoT)** | **97.5%** | **60ms** | **0.88** |
| CNN Only | 94.0% | 50ms | 0.75 |
| IoT + Random Forest | 91.0% | 40ms | 0.70 |
| MobileNet | 95.5% | 70ms | 0.80 |
| HOG + SVM | 89.0% | 35ms | 0.65 |

**Detailed Metrics (Sprint II):**
- ✅ Accuracy: **97.25%**
- ✅ Precision: **98.1%**
- ✅ Recall: **98.5%**
- ✅ F1-Score: **98.3%**
- ✅ Inference Time: **< 0.9 seconds**

---

## 🚀 Getting Started

### 1. Clone this repository
```bash
git clone https://github.com/YOUR_USERNAME/food-freshness-ai-iot.git
cd food-freshness-ai-iot
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare your dataset
Place your food images in this structure:
```
dataset/
├── train/
│   ├── fresh/
│   ├── slightly_spoiled/
│   └── spoiled/
└── test/
    ├── fresh/
    ├── slightly_spoiled/
    └── spoiled/
```

### 4. Train the model
```bash
python models/train.py
```

### 5. Upload IoT code
- Open `iot/sensor_node.ino` in Arduino IDE
- Fill in your WiFi and Adafruit IO credentials in `config.h`
- Upload to your ESP8266 board

### 6. Run the dashboard
```bash
python dashboard/app.py
```
Open `http://localhost:5000` in your browser.

---

## 📊 CNN Architecture (Layer Configuration)

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Conv2D | (222, 222, 32) | 896 |
| MaxPooling2D | (111, 111, 32) | 0 |
| Conv2D | (109, 109, 64) | 18,496 |
| Conv2D | (52, 52, 64) | 36,928 |
| Conv2D | (24, 24, 128) | 73,856 |
| Conv2D | (10, 10, 128) | 147,584 |
| Flatten | (12,544) | 0 |
| Dense | (256) | 3,211,520 |
| Dense (Output) | (4) | 260 |
| **Total** | | **1,523,716** |

---

## 🌍 SDG Alignment

This project supports the following UN Sustainable Development Goals:

- 🌾 **SDG 2 — Zero Hunger**: Reduces post-harvest food losses
- 💊 **SDG 3 — Good Health & Well-being**: Prevents consumption of spoiled food
- ♻️ **SDG 12 — Responsible Consumption**: Promotes smart, sustainable food handling

---

## 📄 Publication

> **"AI AND IOT SOLUTIONS FOR QUALITY CONTROL IN FRUITS, VEGETABLES, AND MEAT: ENSURING FRESHNESS AND SAFETY"**
> 
> Accepted at **14th ICRCET-2025** — International Conference on Recent Challenges in Engineering and Technology  
> 📅 26–27 April 2025 | 📍 Bangalore, India | 🔖 Scopus Indexed  
> Ref No: 78369

---

## 🔮 Future Enhancements

- [ ] Mobile app (Android/iOS) for remote monitoring
- [ ] Edge AI — run model directly on microcontroller (offline mode)
- [ ] Blockchain integration for supply chain traceability
- [ ] Hyperspectral / infrared imaging for internal spoilage detection
- [ ] Expand dataset to seafood, dairy, and processed foods

---

## 📜 License

This project is submitted as academic work at SRM Institute of Science and Technology.  
For research or educational use, please cite the ICRCET-2025 paper.

---

<p align="center">Made with ❤️ by Kirthic, Raghul & Lavanyah | SRMIST 2025</p>
