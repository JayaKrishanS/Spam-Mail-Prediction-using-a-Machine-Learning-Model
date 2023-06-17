import  pickle
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from sklearn.feature_extraction.text import TfidfVectorizer

model = pickle.load(open("Spam_mail_model.jpg","rb"))
x_train = pd.read_pickle(open("TF-IDF.jpg","rb"))
st.set_page_config(page_title="Spam Mail Prediction", page_icon="",layout="wide", initial_sidebar_state="expanded")

def home():
    st.title("Spam mail Prediction")
    st.header("Description:")
    st.write("This is a Spam or Ham Mail Classifier, an Streamlit application designed to predict whether an email is spam ham using a Machine Learning model. It has become crucial to filter out unwanted spam messages to ensure an efficient and secure email experience. This application aims for classifying emails, saving time and reducing the risk of falling victim to phishing or other malicious activities.")
    st.image("image1.png")
    


def prediction():
    feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)
    feature_extraction.fit_transform(x_train)

    st.header("Enter the message for predicting")
    input_mail = []
    input = st.text_area("Paste your message")
    input_mail.append(input)
    input_data_features = feature_extraction.transform(input_mail)

    mail = ''

    if st.button("Predict"):
        Spam_prediction = model.predict(input_data_features)
        if (Spam_prediction[0] == 1):
            mail = 'Your mail is Ham'
        else:
            mail = 'Your mail is a Spam.'
    st.success(mail)
with st.sidebar:
    selected = option_menu(
        menu_title="Spam mail predictor",  
        options=["Home","---","Start predicting"],
        icons=["house","", ""], 
        menu_icon="",
        default_index=0,
    )
    st.header("Github link:")
    st.markdown("https://github.com/JayaKrishanS/Spam-Mail-Prediction-using-a-Machine-Learning-Model.git")
if selected == "Home":
    home()

elif selected == "Start predicting":
    prediction()

