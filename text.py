import pandas as pd
import emoji
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import pickle
import streamlit as st

st.image(r"C:\Users\DELL\Pictures\inno_image.webp")
name=st.title('Sentiment Analysis in Mental Health')
model=pickle.load(open(r"C:\Users\DELL\Machine Learning\sentiment1.pkl",'rb'))
bow=pickle.load(open(r"C:\Users\DELL\Machine Learning\sentiment_bow.pkl",'rb'))

test = st.text_input('Enter the statement:')
st.button("Submit")
data = bow.transform([test]).toarray() 
spam_ham = model.predict(data)[0]
spam_ham
