from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Page
@app.route('/predictpage')
def predictpage():
    return render_template('predict.html')

# Prediction Logic
@app.route('/predict', methods=['POST'])
def predict():

    absences = float(request.form['absences'])
    studytime = float(request.form['studytime'])
    failures = float(request.form['failures'])
    g1 = float(request.form['G1'])
    g2 = float(request.form['G2'])

    age = float(request.form['age'])

    sex = int(request.form['sex'])

    internet = int(request.form['internet'])

    schoolsup = int(request.form['schoolsup'])

    famsup = int(request.form['famsup'])

    features = np.array([[
        absences,
        studytime,
        failures,
        g1,
        g2,
        age,
        sex,
        internet,
        schoolsup,
        famsup
    ]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "Student is at Risk of Dropping Out"
    else:
        result = "Student is Academically Safe"

    return render_template(
        'result.html',
        prediction_text=result
    )

if __name__ == "__main__":
    app.run(debug=True)