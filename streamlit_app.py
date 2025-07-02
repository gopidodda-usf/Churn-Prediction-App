import streamlit as st
import joblib
import numpy as np


scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")


st.title("CHURN PREDICTION APP")

st.divider()

st.write("Please enter the values and hit the predict button for getting a prediction.")

st.divider()

age = st.number_input("Enter Age", min_value = 10, max_value = 80, value = 30)

gender = st.selectbox("Enter Gender", ["Male", "Female"])

tenure = st.number_input("Enter Tenure", min_value = 1, max_value = 130, value = 18)

usagefrequency = st.number_input("Enter Usage Frequency", min_value = 1, max_value = 30, value = 7)

supportcalls = st.number_input("Enter Support Calls", min_value = 0, max_value = 100, value = 3)

paymentdelay = st.number_input("Enter Payment Delay", min_value = 0, max_value = 30, value = 10)

subscriptiontype = st.selectbox("Enter Subscription Type", ["Basic", "Standard", "Premium"])

contractlength = st.selectbox("Enter Contract Length", ["Annual", "Quarterly", "Monthly"])

totalspend = st.number_input("Enter Total Spend", min_value = 100, max_value = 1000, value = 300)

lastinteraction = st.number_input("Enter Last Interaction", min_value = 1, max_value = 30, value = 7)

st.divider()

predictbutton = st.button("Predict!")


if predictbutton:
    gender_map = {"Female": 0, "Male": 1}
    gender_ = gender_map[gender]

    subscription_map = {"Basic": 0, "Premium": 1, "Standard": 2}
    subscriptiontype_ = subscription_map[subscriptiontype]

    contract_map = {"Annual": 0, "Monthly": 1, "Quarterly": 2}
    contractlength_ = contract_map[contractlength]
    
    X = [age, gender_, tenure, usagefrequency, supportcalls, paymentdelay, subscriptiontype_, contractlength_, totalspend, lastinteraction]

    X1 = np.array(X)
    
    X_array = scaler.transform([X1])
    
    prediction = model.predict(X_array)[0]

    predicted = "YES" if prediction == 1 else "NO"
    
    if prediction == 0:
        st.markdown("<span style='color:green; font-weight:bold;'>✅ GREAT NEWS! This customer is likely to stay.</span>", unsafe_allow_html = True)
        st.balloons()

    else:
        st.markdown("<span style='color:red; font-weight:bold;'>⚠️ WARNING! This customer is likely to churn.</span>", unsafe_allow_html = True)

else:
    st.write("Please enter all the values and try again!")



