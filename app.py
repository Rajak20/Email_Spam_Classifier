import streamlit as st
import pickle

# Page configuration
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered"
)

# Load the trained model and vectorizer
@st.cache_resource
def load_model():
    with open("spam_classifier.pkl", "rb") as file:
        model = pickle.load(file)

    with open("bow_vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


model, vectorizer = load_model()


# Title
st.title("📧 Email Spam Classifier")
st.write("Enter an email or message below to check whether it is **Spam** or **Not Spam**.")

# User input
email_text = st.text_area(
    "Enter your email/message:",
    height=200,
    placeholder="Example: Congratulations! You have won a free prize. Click here to claim now!"
)


# Prediction
if st.button("🔍 Check Email", use_container_width=True):

    if email_text.strip() == "":
        st.warning("Please enter an email or message first.")

    else:
        # Convert text into numerical features
        text_vector = vectorizer.transform([email_text])

        # Make prediction
        prediction = model.predict(text_vector)[0]

        # Display result
        st.divider()

        if prediction == 1 or str(prediction).lower() == "spam":
            st.error("🚨 SPAM EMAIL DETECTED")
            st.write("This email/message is predicted to be **Spam**.")
        else:
            st.success("✅ NOT SPAM (HAM)")
            st.write("This email/message is predicted to be **Safe / Not Spam**.")

        # Show probability if supported by the model
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(text_vector)[0]
            confidence = max(probability) * 100

            st.info(f"Prediction Confidence: **{confidence:.2f}%**")


# Sidebar
with st.sidebar:
    st.header("About")

    st.write("""
    This application uses a Machine Learning model
    trained to classify emails/messages as:

    - 🚨 Spam
    - ✅ Not Spam (Ham)

    **Model:** Multinomial Naive Bayes  
    **Text Processing:** Bag of Words
    """)