
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF

# Sample dataset
data = {
    "Threshold_250Hz": [10, 20, 40, 70],
    "Threshold_500Hz": [10, 25, 45, 75],
    "Threshold_1000Hz": [15, 30, 50, 80],
    "Threshold_2000Hz": [20, 35, 55, 85],
    "Threshold_4000Hz": [25, 40, 60, 90],
    "Threshold_8000Hz": [30, 45, 65, 95],
    "Label": ["Normal", "Mild", "Moderate", "Severe"]
}

df = pd.DataFrame(data)
X = df.drop("Label", axis=1)
y = df["Label"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_encoded)

# Functions
def predict_hearing_loss(thresholds):
    input_df = pd.DataFrame([thresholds], columns=X.columns)
    pred = model.predict(input_df)[0]
    label = le.inverse_transform([pred])[0]
    return label


def recommend_hearing_aid(hearing_loss_type):
    if hearing_loss_type == "Normal":
        return "No hearing aid required"
    elif hearing_loss_type == "Mild":
        return "Use a basic behind-the-ear (BTE) or in-the-ear (ITE) hearing aid"
    elif hearing_loss_type == "Moderate":
        return "Use a digital hearing aid with volume control and noise reduction"
    elif hearing_loss_type == "Severe":
        return "Use a powerful BTE or cochlear implant system"
    else:
        return "Consult audiologist for advanced fitting options"

# Streamlit UI
st.title("Hearing Loss Detection & Hearing Aid Recommender")

st.write("### Enter Audiogram Thresholds (in dB HL)")

thresholds = []
for freq in ["250Hz", "500Hz", "1000Hz", "2000Hz", "4000Hz", "8000Hz"]:
    val = st.slider(f"Threshold at {freq}", 0, 100, 30)
    thresholds.append(val)

import matplotlib.pyplot as plt

# Plot Audiogram
fig, ax = plt.subplots()
frequencies = [250, 500, 1000, 2000, 4000, 8000]
ax.plot(frequencies, thresholds, marker='o')
ax.set_title("Audiogram")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Hearing Threshold (dB HL)")
ax.invert_yaxis()  # Audiograms are traditionally upside-down
st.pyplot(fig)

if st.button("Analyze"):
    prediction = predict_hearing_loss(thresholds)
    recommendation = recommend_hearing_aid(prediction)
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Hearing Assessment Report", ln=True, align='C')
    pdf.ln(10)
    for i, f in enumerate(["250Hz", "500Hz", "1000Hz", "2000Hz", "4000Hz", "8000Hz"]):
        pdf.cell(200, 10, txt=f"{f}: {thresholds[i]} dB HL", ln=True)
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Predicted Hearing Loss: {prediction}", ln=True)
        pdf.cell(200, 10, txt=f"Recommended Aid: {recommendation}", ln=True)

        pdf_file = "hearing_report.pdf"
        pdf.output(pdf_file)

        with open(pdf_file, "rb") as f:
           st.download_button("Download PDF Report", f, file_name=pdf_file, key="unique_pdf_btn")    

 st.markdown("---")
 st.markdown("### About")
 st.markdown("""
This app was developed as a real-time simulation project for the **Human Assist Devices** subject (CBM352),  
B.E. Electronics and Communication Engineering, Anna University - Regulation 2021.

**Features:**
- Predicts hearing loss using machine learning
- Recommends suitable hearing aids
- Visualizes audiogram data
- Generates downloadable reports

**Developer:** [Your Name]  
**Toolkits:** Google Colab, Streamlit, Scikit-learn, FPDF
""")
