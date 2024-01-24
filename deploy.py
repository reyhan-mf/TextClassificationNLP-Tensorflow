import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the NLP model from the H5 file
model = tf.keras.models.load_model('model.h5')

# Load the tokenizer used during training
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(["Type here..."])  # Use any default text for fitting the tokenizer

st.title('NLP Model Deployment')

# User input
user_input = st.text_input('Enter Text:', 'Type here...')

# Button to trigger model prediction
if st.button('Predict'):
    # Preprocess the user input using the tokenizer
    processed_input = tokenizer.texts_to_sequences([user_input])
    processed_input = pad_sequences(processed_input, maxlen=300)  # Adjust maxlen according to your model input shape
    
    # Use the loaded model to make predictions
    prediction = model.predict(processed_input)
    
    # Display the prediction
    st.write('Prediction:', prediction)
