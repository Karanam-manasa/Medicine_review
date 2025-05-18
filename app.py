from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("medicine_review_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return jsonify({'predicted_review_percent': prediction})

if __name__ == '__main__':
    app.run(debug=False)
