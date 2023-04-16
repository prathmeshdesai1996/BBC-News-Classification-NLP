import streamlit as st
import pickle
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
nltk.download('stopwords')

# Load the vectorizer from the file
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load the saved model from the file
with open('mnbmodel.pkl', 'rb') as file:
    mnbmodel = pickle.load(file)


def predict_category(text):
    pattern = '[a-z]+|[0-9]+'

    matches = re.findall(pattern, text.lower())

    words = [word for word in matches if word not in stopwords.words('english')]
    words = ' '.join(words)

    X = vectorizer.transform([words])

    # make predictions using the saved model
    y_pred_prob = mnbmodel.predict_proba(X)

    categories = {'0': 'tech', '1': 'business', '2': 'sport', '3': 'entertainment', '4': 'politics'}

    # return the predicted category
    return categories[str(np.argmax(y_pred_prob))]


if __name__ == '__main__':
    # Start the Streamlit app
    st.set_page_config(page_title="Text Classification", page_icon=":guardsman:", layout="wide")
    st.title("Text Classification App")

    text_input = st.text_area("Enter the text you want to classify")
    
    if st.button("Classify"):
        category = predict_category(text_input)
        st.success(f"The category of the input text is: {category}")

