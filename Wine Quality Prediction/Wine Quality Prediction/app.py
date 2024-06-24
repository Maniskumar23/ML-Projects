from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the dataset and train the model
wine_datasets = pd.read_csv('winequality-red.csv')
X = wine_datasets.drop('quality', axis=1)
Y = wine_datasets['quality'].apply(lambda y_value: 2 if y_value >= 7 else (1 if 5 <= y_value <= 6 else 0))
model = RandomForestClassifier()
model.fit(X, Y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form['features']
    features = [float(x) for x in input_data.split(',')]
    input_data_as_np_array = np.asarray(features)
    input_data_reshaped = input_data_as_np_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)

    if prediction[0] == 2:
        result = 'Wine is excellent'
    elif prediction[0] == 1:
        result = 'Wine is good'
    else:
        result = 'Wine is bad'
    
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
