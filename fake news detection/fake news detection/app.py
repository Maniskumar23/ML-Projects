import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from flask import Flask, request, render_template
import threading

# Initialize Flask application
app = Flask(__name__)

# Download stopwords
nltk.download('stopwords')

# Load the dataset
news_datasets = pd.read_csv('train.csv')

# Check for missing values
news_datasets = news_datasets.fillna('')

# Merge author and title to create content
news_datasets['content'] = news_datasets['author'] + ' ' + news_datasets['title']

# Define the PorterStemmer
portstem = PorterStemmer()

# Function to perform stemming
def stemmer(content):
    stem_content = re.sub('[^a-zA-Z]', ' ', content)
    stem_content = stem_content.lower()
    stem_content = stem_content.split()
    stem_content = [portstem.stem(word) for word in stem_content if not word in stopwords.words('english')]
    stem_content = ' '.join(stem_content)
    return stem_content

# Apply stemming to the content
news_datasets['content'] = news_datasets['content'].apply(stemmer)

# Separate data and labels
X = news_datasets['content'].values
Y = news_datasets['label'].values

# Convert text data to numerical data
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=2)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Predict and calculate accuracy
X_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(X_train_pred, Y_train)

X_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(X_test_pred, Y_test)

# Print accuracy
print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    input_text = request.form['input_text']
    # Process the input text
    processed_input = vector.transform([stemmer(input_text)])
    prediction = model.predict(processed_input)
    result = 'The news is Real' if prediction[0] == 0 else 'The news is Fake'
    return render_template('index.html', prediction=result)

def run_app():
    app.run(debug=True, use_reloader=False)

# Start the Flask app in a separate thread
thread = threading.Thread(target=run_app)
thread.start()
