
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('taxi_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    price = float(request.form['price'])
    population = float(request.form['population'])
    income = float(request.form['income'])
    parking = float(request.form['parking'])

    prediction = model.predict([[price, population, income, parking]])[0]

    return render_template('result.html',
                           price=price,
                           population=population,
                           income=income,
                           parking=parking,
                           prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)