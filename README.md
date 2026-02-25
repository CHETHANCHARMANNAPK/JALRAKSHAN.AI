# JALRAKSHAN.AI

**AI-Powered Community Water Leakage & Safety Intelligence System**

## Overview
JALRAKSHAN.AI is an AI-driven web application designed to detect hidden water leakages early in urban communities using smart water meter data. By leveraging advanced anomaly detection, the system helps prevent massive water loss and supports responsible, human-in-the-loop decision making.

## Features
- Unsupervised anomaly detection (Isolation Forest)
- Animated, interactive data visualizations
- Node/community selector for scalability
- Impact dashboard with real-world scenario estimates
- Responsible AI: human-in-the-loop, transparent, and privacy-respecting

## How it Works
We train an unsupervised anomaly detection model on smart water meter flow data. Since leakages appear as **continuous abnormal consumption rather than spikes**, the AI learns normal usage patterns and flags deviations early—enabling preventive action before major water loss.

## Getting Started
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the app:**
   ```bash
   streamlit run app.py
   ```
3. **Open in browser:**
   The app will open automatically, or visit the URL shown in your terminal (default: http://localhost:8501).

## Deployment
- Deploy easily on [Streamlit Community Cloud](https://streamlit.io/cloud) or any cloud VM.
- Push your code to GitHub and connect your repo for one-click deployment.

## Data
- Uses sample smart water meter data (see `archive/` folder).
- No personal data is collected or processed; only community-level aggregate readings are analyzed.

## Responsible AI Commitment
- Decision support only—final actions remain with human authorities.
- No automated enforcement or penalties.
- Transparent, explainable logic.

## Real-World Scenario
> In a 100-home community, detecting leaks 64 days earlier could save enough water to supply ~400 families for a month.

## License
MIT License

---
JALRAKSHAN.AI — AI for Good | SDG 6 | Responsible AI | Hackathon 2026
