# 🦅 U-Scout: AI Transfer & Scouting Assistant

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Gemini AI](https://img.shields.io/badge/Gemini_2.5-8E75B2?logo=google&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Scikit_Learn-F7931E?logo=scikit-learn&logoColor=white)

**U-Scout** is a data-driven prototype built specifically for **FC Universitatea Cluj (U Cluj)**. It ingests massive amounts of raw Wyscout match data to identify high-potential prospects, calculate 1-to-1 tactical matches using Machine Learning, and generate executive AI scouting reports.

🏆 *This project was developed during the **SuperLiga Hackathon** organized by **Google Developer Group (GDG) UTCN** in collaboration with **FC Universitatea Cluj**.*

**Hackathon Theme:** *"Transfer and Scouting Assistant - An AI powered scouting tool that recommends players based on performance data, filters and growth potential."*

---

## ✨ Key Features

* **Triple-Engine Replacement Analysis:** Automatically recommends tactical replacements based on three distinct strategies:
    * **The Future Star:** Identifies top U21 prospects ranked by a custom Age/Performance decay algorithm.
    * **The Tactical Twin:** 1-to-1 player matching using Machine Learning (Pearson Correlation & Euclidean distances) to find players with the exact same tactical footprint.
    * **The Value Performer:** Pure statistical outliers based on positional averages and market value.
* **Interactive Spider Charts:** Visualizes player compatibility across role-specific tactical metrics.
* **Gemini AI Integration:** Generates ruthless, 3-sentence executive summaries balancing 'Growth Potential' against 'Market Value' to save the Sporting Director's time.
* **Advanced Database Filtering:** Granular search with custom UI segmented controls and real-time debounce typing.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit, Plotly (for interactive radar charts), Custom CSS
* **Backend:** Python, Pandas, NumPy
* **AI / LLM:** Google Generative AI (Gemini 2.5 Flash)
* **Machine Learning:** Scikit-learn, Custom mathematical modeling for similarity scoring

---

## 📁 Project Structure

```text
CodeInBlackWite/
│
├── Backend/
│   ├── data_processor.py      # Parses JSON/CSV data and calculates Growth Potential
│   └── models/                # Machine Learning Engine
│       ├── data_loader.py
│       ├── feature_engineering.py
│       ├── normalization.py
│       ├── similarity.py      # Calculates the "Tactical Twin" score
│       └── train.py           # ML Training script
│
├── Data/                      # 🚨 Required local data folder
│   ├── players.csv            # Transfermarkt value data
│   ├── u_cluj_current_squad.csv 
│   └── Date - meciuri/        # Raw Wyscout JSON match files go here
│
├── Frontend/
│   ├── main.py                # Main Streamlit application and routing
│   ├── tab_search.py          # Database filter and Gemini UI
│   └── tab_squad.py           # Triple-engine replacement and Radar Charts
│
└── requirements.txt
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone [https://github.com/yourusername/u-scout.git](https://github.com/yourusername/u-scout.git)
cd u-scout
```

### 2. Add your Data
Because Wyscout data is proprietary, the raw match JSON files are not included in this repository. 
* Create a folder named `Data` in the root directory.
* Inside `Data`, create a folder named `Date - meciuri`.
* Place your `players (1).json` and all `_players_stats.json` files inside `Data/Date - meciuri/`.
* Place `players.csv` and `u_cluj_current_squad.csv` directly inside the `Data/` folder.

### 3. Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv

# Activate on Windows (Command Prompt):
venv\Scripts\activate
# Activate on Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Activate on Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Set up your Gemini API Key
Get a free API key from [Google AI Studio](https://aistudio.google.com/).
* **On Windows (Command Prompt):** `set API_KEY=your_api_key_here`
* **On Windows (PowerShell):** `$env:API_KEY="your_api_key_here"`
* **On Mac/Linux:** `export API_KEY="your_api_key_here"`

### 5. Train the Machine Learning Engine
Before running the app for the first time, you must train the ML models to generate the pre-calculated tactical lookup tables.
```bash
python Backend/models/train.py
```

### 6. Run the Application
Make sure you are in the root folder, then launch Streamlit:
```bash
streamlit run Frontend/main.py
```

---

## 🤝 Acknowledgments
* **GDG UTCN** for organizing an incredible hackathon environment.
* **FC Universitatea Cluj** for providing the problem statement, data, and real-world sporting context.