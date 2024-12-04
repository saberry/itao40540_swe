import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.sentiment_analyzer import SentimentAnalyzer
from textblob import TextBlob

nltk.download('vader_lexicon')

# Dear God -- https://docs.streamlit.io/develop/api-reference

# We want to give people the chance
# to input their own song lyrics 
# and select the sentiment method

st.title('Sentiment Analysis App')

st.write('This app will allow you to input song lyrics and determine the sentiment of the lyrics.')

user_input = st.text_area('Enter song lyrics here:')

sentiment_selector = st.selectbox('Select a sentiment method:', ['Intensity','VADER'])

if sentiment_selector == 'Intensity':
    score = TextBlob(user_input).sentiment.polarity
    score = pd.DataFrame({'compound':score}, index=[0])
    #st.write(score)
else:
    sid = SentimentIntensityAnalyzer()
    score = sid.polarity_scores(user_input)
    score = pd.DataFrame(score, index=[0])
    #st.write(score)

st.write('The compound score is:', score['compound'][0])

if score['compound'][0] >= 0.05:
    st.write('The sentiment is positive.')
elif score['compound'][0] <= -0.05:
    st.write('The sentiment is negative.')
else:
    st.write('The sentiment is neutral.')

#python3 -m streamlit run streamlit_app.py