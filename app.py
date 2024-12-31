import os
import json
import datetime
import csv
import ssl
import random
import nltk
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

def analyze_chat_logs(file_path):
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        st.subheader("Chat Log Insights")
        if not data.empty:
            # Most frequent user inputs
            st.write("**Most Frequent User Inputs:**")
            st.write(data['User Input'].value_counts().head())

            # Most frequent chatbot responses
            st.write("**Most Frequent Chatbot Responses:**")
            st.write(data['Chatbot Response'].value_counts().head())

            # Busiest hours
            data['Timestamp'] = pd.to_datetime(data['Timestamp'])
            busiest_hours = data['Timestamp'].dt.hour.value_counts().sort_index()
            st.write("**Chat Activity by Hour:**")
            st.bar_chart(busiest_hours)
        else:
            st.write("No data available in the chat log.")
    else:
        st.write("Chat log file not found.")

def export_chat_logs(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            st.download_button(label="Download Chat Log", data=file, file_name="chat_log.csv", mime="text/csv")

def main():
    st.title("Chatbot with Enhanced Features")

    menu = ["Home", "Conversation History", "Analyze Logs", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome to the chatbot. Type a message below to start chatting.")

        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        user_input = st.text_input("You:")
        if user_input:
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
            export_chat_logs('chat_log.csv')
        else:
            st.write("No conversation history found.")

    elif choice == "Analyze Logs":
        analyze_chat_logs('chat_log.csv')

    elif choice == "About":
        st.subheader("About the Project")
        st.write("This chatbot uses NLP and Logistic Regression for intent recognition and response generation.")
        st.write("Enhanced features include chat log analysis and download options.")

if __name__ == '__main__':
    main()
