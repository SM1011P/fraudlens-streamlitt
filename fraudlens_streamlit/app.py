import streamlit as st
import joblib
import re
import cv2
import numpy as np
from PIL import Image

# =====================
# Page Config
# =====================
st.set_page_config(
    page_title="FraudLens",
    page_icon="🛡️",
    layout="wide"
)

# =====================
# Custom CSS (TEAL & WHITE)
# =====================
st.markdown("""
<style>

/* Main background */
.main {
    background-color: #E0F7FA;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #008080;
}
[data-testid="stSidebar"] * {
    color: white !important;
}

/* Headers */
h1, h2, h3, h4 {
    color: #004D4D !important;
}

/* Cards */
.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 25px;
    box-shadow: 0px 4px 15px rgba(0,128,128,0.2);
}

/* Buttons */
.stButton > button {
    background-color: #008080;
    color: white;
    border-radius: 10px;
    padding: 10px 25px;
    font-size: 16px;
    border: none;
}
.stButton > button:hover {
    background-color: #006666;
    color: white;
}

/* Text input */
textarea, input {
    background-color: #F1FFFF !important;
    color: black !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #004D4D;
    margin-top: 40px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =====================
# CENTER LOGO
# =====================
col1, col2, col3 = st.columns([2, 3, 2])

with col2:
    st.image("logo.jpg", width=260)



st.markdown("<h3 style='text-align:center;'>SMS, Link & QR Fraud Detection System</h3>", unsafe_allow_html=True)
st.markdown("---")

# =====================
# Load Models
# =====================
text_model = joblib.load("models/sms_email_model.pkl")
text_vectorizer = joblib.load("models/sms_email_vectorizer.pkl")

link_model = joblib.load("models/link_model.pkl")
link_vectorizer = joblib.load("models/link_vectorizer.pkl")

# =====================
# Helper Functions
# =====================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " url ", text)
    text = re.sub(r"\d+", " number ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

def clean_url(url):
    url = str(url).lower()
    url = re.sub(r"https?://", "", url)
    url = re.sub(r"www.", "", url)
    return url

# =====================
# Sidebar
# =====================
st.sidebar.title("⚙️ FraudLens Menu")
option = st.sidebar.radio(
    "Choose Feature:",
    ["📩 SMS / Email Detection", "🔗 Link & QR Detection"]
)

# =====================
# SMS / Email Detection
# =====================
if option == "📩 SMS / Email Detection":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📩 SMS & Email Spam Detection")

    msg = st.text_area("Enter SMS or Email message:")

    if st.button("🔍 Analyze Message"):
        if msg.strip() == "":
            st.warning("Please enter a message.")
        else:
            cleaned = clean_text(msg)
            vec = text_vectorizer.transform([cleaned])
            result = text_model.predict(vec)[0]

            if str(result).lower() == "spam":
                st.error("⚠️ This message is SPAM / FRAUD")
            else:
                st.success("✅ This message is SAFE")

    st.markdown("</div>", unsafe_allow_html=True)

# =====================
# Link & QR Detection
# =====================
elif option == "🔗 Link & QR Detection":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("🔗 Link Detection")

    url = st.text_input("Enter link:")

    if st.button("🔍 Check Link"):
        if url.strip() == "":
            st.warning("Please enter a link.")
        else:
            cleaned = clean_url(url)
            vec = link_vectorizer.transform([cleaned])
            result = link_model.predict(vec)[0]

            if str(result).lower() in ["phishing", "malicious"]:
                st.error("⚠️ Fraud / Phishing Link Detected")
            else:
                st.success("✅ Safe Link")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📷 QR Code Detection")

    uploaded_file = st.file_uploader("Upload QR Code Image", type=["png","jpg","jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded QR Code", use_column_width=True)

        img_np = np.array(image.convert("RGB"))
        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(img_np)

        if data:
            st.write("Decoded QR Link:")
            st.code(data)

            cleaned = clean_url(data)
            vec = link_vectorizer.transform([cleaned])
            result = link_model.predict(vec)[0]

            if str(result).lower() in ["phishing", "malicious"]:
                st.error("⚠️ Fraud QR Code Detected")
            else:
                st.success("✅ Safe QR Code")
        else:
            st.warning("No QR detected. Please upload a clearer image.")

    st.markdown("</div>", unsafe_allow_html=True)

# =====================
# Footer
# =====================
st.markdown("<div class='footer'>🚀 FraudLens | AI-Based Fraud Detection System</div>", unsafe_allow_html=True)
