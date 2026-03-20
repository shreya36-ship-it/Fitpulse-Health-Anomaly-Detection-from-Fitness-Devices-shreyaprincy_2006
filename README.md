# 🏃 FitPulse Health Analytics Dashboard

FitPulse is an interactive health analytics dashboard built using Streamlit that analyzes fitness tracking data and provides insights using Machine Learning and Time Series Forecasting.

---

## 📌 Project Overview

FitPulse helps users and analysts:

* 📊 Explore fitness and health data
* 🧠 Extract advanced time-series features
* 🔮 Forecast heart rate trends
* 👥 Segment users based on behavior patterns

This project demonstrates the integration of data analytics, machine learning, and visualization into a single dashboard.

---

## 🚀 Features

### 📂 Dataset Preview

* View dataset structure and sample records
* Understand data distribution

### ⚙️ TSFresh Feature Extraction

* Extract meaningful time-series features
* Visualize feature importance using heatmaps

### 🔮 Prophet Forecasting

* Predict future heart rate trends
* Visualize forecast with confidence intervals

### 👥 Clustering Analysis

* Group users using KMeans clustering
* PCA-based visualization of clusters

---

## 🛠️ Tech Stack

* Python 🐍
* Streamlit 🌐
* Pandas & NumPy 📊
* Matplotlib & Seaborn 📈
* TSFresh ⚙️
* Prophet 🔮
* Scikit-learn 🤖

---

## 📁 Project Structure

```
Fitpulse-Project/
│
├── app.py
├── Fitness_Health_Tracking_Dataset.csv
├── README.md
└── requirements.txt
```

---

## ⚡ Installation & Setup

### 1️⃣ Clone the Repository

```
git clone https://github.com/your-username/fitpulse.git
cd fitpulse
```

### 2️⃣ Create Virtual Environment (Optional but Recommended)

```
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

Or manually:

```
pip install streamlit pandas numpy matplotlib seaborn prophet tsfresh scikit-learn
```

---

## ▶️ Run the Application

```
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## 📊 Sample Use Cases

* Fitness tracking analysis
* Health anomaly detection
* User segmentation for health apps
* Time-series forecasting for wearable devices

---

## ⚠️ Notes

* Ensure dataset file is placed in the same directory as `app.py`
* Prophet may take time to install on first setup
* TSFresh feature extraction can be computationally intensive

---

## 🌟 Future Enhancements

* Real-time data integration from wearable devices
* Deployment using Docker & AWS
* Advanced anomaly detection models
* User authentication system

---

## 👩‍💻 Author

**Shreya**

* Developed as part of internship/project work
* Focused on health analytics and AI-driven insights

---

## 📜 License

This project is for educational and research purposes.

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!
