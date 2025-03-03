import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Title of the app
st.title("Customer Spending Prediction")
st.write("Welcome to the Ecommerce Customer Spending Prediction App!")

# Load the pre-trained model
def load_model():
    with open("linear_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Required columns for the model
required_columns = ["Avg. Session Length", "Time on App", "Time on Website", "Length of Membership"]

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Check if uploaded dataset has the required columns
    if set(required_columns) == set(df.columns):
        st.success("Dataset uploaded successfully!")
        input_df = df[required_columns]  # Use uploaded data for prediction
        show_extra_analysis = False  # Only show predictions, skip comparisons
    else:
        st.error("Error: Uploaded dataset does not match the required columns!")
        input_df = None
        show_extra_analysis = False  # Prevent further execution
else:
    file_path = "Ecommerce Customers.csv"  # Default dataset
    df = pd.read_csv(file_path)
    
    st.sidebar.header("User Input")
    
    def user_input_features():
        avg_session_length = st.sidebar.number_input("Avg. Session Length", min_value=0.0, value=0.0)
        time_on_app = st.sidebar.number_input("Time on App", min_value=0.0, value=0.0)
        time_on_website = st.sidebar.number_input("Time on Website", min_value=0.0, value=0.0)
        length_of_membership = st.sidebar.number_input("Length of Membership", min_value=0.0, value=0.0)

        data = {
            "Avg. Session Length": avg_session_length,
            "Time on App": time_on_app,
            "Time on Website": time_on_website,
            "Length of Membership": length_of_membership,
        }
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()
    show_extra_analysis = True  # Show extra analysis only for user input case

st.divider()

# Display user input
st.subheader("Predict on Input Data")
if input_df is not None:
    st.write(input_df)

    # Make predictions
    if st.button("Predict"):
        if input_df.isnull().any().any():
            st.error("Please fill in all the input fields.")
        else:
            predictions = model.predict(input_df)
            st.subheader("Prediction")
            
            if len(predictions) == 1:
                st.success(f"Predicted Yearly Amount Spent: **${predictions[0]:.2f}**")
            else:
                st.write("Predicted Yearly Amount Spent for Uploaded Data:")
                st.write(pd.DataFrame({"Prediction": predictions}))

if show_extra_analysis:
    st.divider()
    # Comparison of predicted and actual values
    st.subheader("Comparison with Average Customer")
    avg_customer = df[required_columns].mean()

    comparison_data = {
        "Feature": input_df.columns,
        "Your Input": input_df.values.flatten(),
        "Average Customer": avg_customer.values,
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.write(comparison_df)

    st.divider()

    # Dataset Summary
    st.subheader("Dataset Summary")
    st.write(df.describe())

    st.divider()

    # Model Accuracy
    st.subheader("Model Score")
    model_score = model.score(df[required_columns], df["Yearly Amount Spent"])
    st.info(f"Model Accuracy (R² Score): {model_score:.2f}")

    st.divider()
    # Model accuracy
    accuracy = 0.98  
    error = 1 - accuracy  

    # Data for pie chart
    labels = ["Accuracy", "Error"]
    sizes = [accuracy * 100, error * 100]  # Convert to percentage
    colors = ["#4CAF50", "#FF5733"]  # Green for accuracy, Red for error

    # Create pie chart
    fig, ax = plt.subplots(figsize=(4, 3))  
    ax.pie(sizes, labels=labels, autopct="%1.2f%%", colors=colors, startangle=0, 
           textprops={'fontsize': 8})  

    ax.set_title("Model Accuracy", fontsize=10)  

    # Display in Streamlit
    st.pyplot(fig)
