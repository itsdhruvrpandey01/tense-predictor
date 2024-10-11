import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Initialize Flask app
app = Flask(__name__)

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Load the dataset
file_path = r'C:\Users\Dhruv\Downloads\collab-ideas-main\collab-ideas-main\topic-readmes\pos_tag_wise_tense_form_rules.csv'  # Update with your file path
pos_tag_data = pd.read_csv(file_path)

# Rename the columns for ease of use if necessary
pos_tag_data.columns = ['pos_tags', 'tense']

# Encode the tense labels
label_encoder = LabelEncoder()
pos_tag_data['tense_label'] = label_encoder.fit_transform(pos_tag_data['tense'])

# Split dataset into features and labels
X = pos_tag_data['pos_tags']
y = pos_tag_data['tense_label']

# Convert POS tag combinations to a matrix of token counts
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Function to predict tense based on user input
def predict_tense(words):
    # Ensure the punkt tokenizer is downloaded
    nltk.download('punkt')
    
    tokens = nltk.word_tokenize(words)
    pos_tags = nltk.pos_tag(tokens)
    pos_tags_str = ','.join([tag[1] for tag in pos_tags])
    pos_tags_vectorized = vectorizer.transform([pos_tags_str])
    prediction = model.predict(pos_tags_vectorized)
    tense = label_encoder.inverse_transform(prediction)
    return tense[0]

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    predicted_tense = ''
    if request.method == 'POST':
        user_input = request.form['user_input']
        predicted_tense = predict_tense(user_input)
    return render_template('index.html', predicted_tense=predicted_tense)

if __name__ == '__main__':
    app.run(debug=True)
