## 🚆 PathsPredict - KRL & TransJakarta Congestion Predictor 🚌

A web-based application with a machine learning model to predict congestion levels for KRL and Transjakarta in Jakarta, helping commuters choose the least crowded transport option.
This project was developed as an End-of-Semester Project for "Sistem Informasi Cerdas" course by ARK (Andrew, Runi, Kae)

## 🚀 Features
**Congestion Prediction**: Utilizes a Logistic Regression model to predict "TINGGI" (High) or "RENDAH" (Low) congestion levels for KRL and Transjakarta.  
**Predictive Dashboard**: Forecasts congestion for upcoming days based on historical and real-time data.  
**Data Management (CRUD)**: Allows for adding, viewing, updating, and deleting historical ridership data.  
**Data Export**: Exports historical and crowdsourced data to an Excel file for further analysis.  
**Real-time Data Integration**: Continuously updates predictions based on new data to maintain accuracy.  

## 🛠 Tech Stack
Backend: Python (Flask, Pandas, scikit-learn), Joblib
Database: In-memory simulation
Frontend: HTML, CSS, JavaScript
Data Source: Satu Data Jakarta (2024-2025)

## 📂 Folder Structure
├── Jumlah_Penumpang_Angkutan_Umum_yang_Terlayani_Perhari.csv
├── app.py
├── categorical_features.pkl
├── index.html
├── logistic_regression_penumpang_pipeline.pkl
├── model_features_with_moda.pkl
├── model_training.py
└── numerical_features.pkl

## 🧪 Getting Started
`git clone <repository_url>
cd PredictJakarta
pip install -r requirements.txt
python app.py
Visit http://localhost:5000 to explore the application.`

## ✅ Future Improvements
* Integrate with a real live database for persistent data storage.
* Implement a more sophisticated machine learning model (e.g., a time-series model like ARIMA) for more accurate long-term predictions.  
* Build a more interactive and user-friendly front-end dashboard with data visualizations (e.g., charts and graphs).  
* Add a feature to predict congestion for specific routes or times of day, rather than just daily averages.
