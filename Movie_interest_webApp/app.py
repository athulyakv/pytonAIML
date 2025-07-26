from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model
model1 = pickle.load(open('C:\\Users\\ASUS\\PycharmProjects\\Movie_interest_webApp\\movie_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')  # The form to input values

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from HTML form
    val1 = float(request.form['feature1'])
    val2 = float(request.form['feature2'])


    # Format input data for prediction
    input_data = [[val1, val2]] # Two users' inputs

    # Make prediction using the model
    predictions = model1.predict(input_data)

    # Render result
    return render_template('result.html', prediction=predictions)

if __name__ == '__main__':
    app.run(debug=True)
