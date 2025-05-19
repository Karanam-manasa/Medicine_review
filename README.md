# 💊 Medicine Review Predictor
This is a Machine Learning web app built with Flask to predict the Excellent Review % of a medicine based on its composition, uses, side effects, manufacturer, and other review metrics.

## 🚀 Features
- Predicts Excellent Review % using Random Forest Regressor
- Input form for medicine details
- Clean and simple UI with Flask + HTML
- Trained on real medicine dataset

## 🛠 Tech Stack
- Python
- Flask
- scikit-learn
- pandas
- HTML/CSS

## 📦 Installation
- git clone https://github.com/yourusername/medicine-review-predictor.git
- cd medicine-review-predictor
- python -m venv venv
- venv\Scripts\activate  
- pip install -r requirements.txt
- python train_model.py
- python app.py

## 📁 File Structure

![image](https://github.com/user-attachments/assets/c2c7f08d-21ca-4f78-bb9b-0cfb6925c37c)

- train_model.py → Trains and saves the ML model
- app.py → Flask web server
- templates/index.html → UI form
- Medicine_Details.csv → Dataset
- medicine_review_model.pkl → Trained model

## Output

![image](https://github.com/user-attachments/assets/643ca09f-9b84-4e21-bf66-9a9d5a47642c)

![image](https://github.com/user-attachments/assets/ebbd5963-5a60-4206-a399-ccf8cb139d40)

