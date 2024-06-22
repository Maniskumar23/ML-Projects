import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the sonar data
sonar_data = pd.read_csv('sonar data.csv', header=None)

# Separate data and labels
x = sonar_data.drop(columns=60, axis=1)
y = sonar_data[60]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=1)

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.get('data')
    if data:
        input_data = data.split(',')
        if len(input_data) == 60:
            try:
                input_data = [float(x) for x in input_data]
                input_data_as_np_array = np.asarray(input_data)
                input_data_reshape = input_data_as_np_array.reshape(1, -1)
                prediction = model.predict(input_data_reshape)

                if prediction[0] == "R":
                    result = "It is Rock"
                else:
                    result = "It is Mine"
            except ValueError:
                result = "Invalid input data. Please enter 60 valid numbers."
        else:
            result = "Invalid input data. Please enter exactly 60 comma-separated values."
    else:
        result = "No input data provided."

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
