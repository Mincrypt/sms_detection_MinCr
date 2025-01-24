import nltk
import pickle
import string
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Porter Stemmer
ps = PorterStemmer()

# Function to preprocess the text
def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric tokens
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

    # Perform stemming
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# Load saved vectorizer and model
vectorizer = pickle.load(open("spam_vectorizer.pkl", 'rb'))
model = pickle.load(open("spam_model.pkl", 'rb'))

# Streamlit App Interface
st.set_page_config(
    page_title="SMS Spam Detection Model",
    page_icon="📩",
    layout="wide",
)

# Header
st.title("📩 SMS Spam Detection Model")
st.write(
    """
    **Detect whether an SMS message is Spam or Not Spam with ease!**
    \nThis app uses advanced Natural Language Processing (NLP) techniques to classify messages.
    """
)
st.markdown("---")

# Input Section
st.header("🔍 Enter Your SMS")
input_sms = st.text_area(
    "Type or paste the SMS message below and click 'Predict' to analyze:",
    height=150,
    placeholder="Enter your SMS here...",
)

if st.button("Predict 🚀"):
    # Check if the input is empty
    if not input_sms.strip():
        st.warning("Please enter a valid SMS message for classification.")
    else:
        # Preprocess the input
        with st.spinner("Processing the message..."):
            transformed_sms = transform_text(input_sms)
        
        # Display Preprocessing Steps (optional for user feedback)
        st.markdown("### 🔄 Preprocessed SMS")
        st.code(transformed_sms)

        # Vectorize the input
        vector_input = vectorizer.transform([transformed_sms])

        # Predict using the model
        with st.spinner("Analyzing..."):
            result = model.predict(vector_input)[0]

        # Display Results
        st.markdown("### 📊 Result:")
        if result == 1:
            st.success("🚨 This SMS is classified as **Spam**.")
        else:
            st.info("✅ This SMS is classified as **Not Spam**.")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About the Model")
    st.write(
        """
        This SMS Spam Detection model is built using:
        - **Natural Language Processing (NLP):** For text preprocessing and analysis.
        - **Machine Learning Model:** Naive Bayes, trained on a labeled dataset of spam and non-spam messages.
        - **Vectorization:** TF-IDF to convert text into numerical features.
        """
    )
    st.markdown("#### 💻 Technologies Used")
    st.write("- Python 🐍\n- Streamlit 🌐\n- Scikit-learn 🤖\n- NLTK 🗂️")
    st.markdown("---")
    st.markdown("**Made by Aman Sharma (MinCr) ❤️**")

# Batch Input Section (Advanced)
st.markdown("---")
st.header("📑 Batch SMS Classification")
uploaded_file = st.file_uploader(
    "Upload a CSV file with a column named `SMS` for batch classification:",
    type=["csv"],
)

if uploaded_file:
    # Process the uploaded file
    with st.spinner("Loading file..."):
        import pandas as pd
        data = pd.read_csv(uploaded_file)

    # Check for the required column
    if "SMS" not in data.columns:
        st.error("The uploaded file must contain a column named `SMS`.")
    else:
        # Preprocess and classify each SMS
        with st.spinner("Classifying messages..."):
            data["Transformed_SMS"] = data["SMS"].apply(transform_text)
            vectorized_data = vectorizer.transform(data["Transformed_SMS"])
            data["Prediction"] = model.predict(vectorized_data)
            data["Prediction"] = data["Prediction"].map({0: "Not Spam", 1: "Spam"})

        # Display results
        st.write("### Batch Classification Results")
        st.dataframe(data[["SMS", "Prediction"]])

        # Download results as CSV
        st.download_button(
            label="📥 Download Results",
            data=data.to_csv(index=False),
            file_name="sms_classification_results.csv",
            mime="text/csv",
        )
