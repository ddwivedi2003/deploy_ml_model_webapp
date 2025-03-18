import streamlit as st
import pandas as pd
import l_reg as lr
import dec_tree as dt
import matplotlib.pyplot as plt

st.title("Machine Learning Web App")
st.write("Upload CSV file for training")
training = st.file_uploader("Select the CSV file")
st.write("Select the model for prediction")
model = st.selectbox("Select the model", ["None","Decision Tree", "Linear Regression"])


    


# Initialize session state variables
if "verify" not in st.session_state:
    st.session_state.verify = 0
if "lr_model" not in st.session_state:
    st.session_state.lr_model = None

if training is not None:
    fl = pd.read_csv(training)
    
    if model == "None":
        pass

    if model == "Linear Regression":
        st.header("Linear Regression Model")
        st.write("Train the model")
        button_train_lr = st.button("Train the model(Linear Regression)")
        if button_train_lr:
            x, y, y_pred_train, lr_model = lr.lr(fl)
            st.session_state.lr_model = lr_model  # Save the trained model in session state
            st.write("Trained Model Graph")
            fig_train, ax_train = plt.subplots()
            ax_train.scatter(x, y, color='blue', label='Actual')
            ax_train.plot(x, y_pred_train, color='red', label='Predicted')
            ax_train.set_title("Linear Regression Trained Model Graph")
            ax_train.set_xlabel("X")
            ax_train.set_ylabel("Y")
            ax_train.legend(["Predicted", "Actual"])
            st.pyplot(fig_train)
            st.write("Model Trained Successfully")
            st.session_state.verify = 1  # Update session state

        if st.session_state.verify == 1:
            st.write("Choose the Option for Prediction")
            lr_pred = st.selectbox("Select the option for prediction", ["Upload CSV for Prediction", "Enter the value for Prediction"])
            if lr_pred == "Enter the value for Prediction":
                st.write("Enter the value for prediction")
                x_in = st.number_input("Enter the value for X")
                button_pred = st.button("Predict")
                if button_pred and st.session_state.lr_model is not None:
                    import numpy as np
                    y_pred = st.session_state.lr_model.predict(np.array([[x_in]]))
                    st.write("The predicted value is", y_pred)

            if lr_pred == "Upload CSV for Prediction":
                st.write("Upload the CSV file for prediction")
                pred = st.file_uploader("Select the CSV for prediction")
                button_pred = st.button("Predict")
                if button_pred and pred is not None and st.session_state.lr_model is not None:
                    f2 = pd.read_csv(pred)
                    x_1 = f2.iloc[:, :-1]
                    y_pred = st.session_state.lr_model.predict(x_1)

                    x_out_flat = x_1.values.flatten()
                    y_pred_out_flat = y_pred.flatten()

                    x_out_flat = pd.Series(x_out_flat).reset_index(drop=True)
                    y_pred_out_flat = pd.Series(y_pred_out_flat).reset_index(drop=True)

                    pred_table = pd.DataFrame({
                        "X": x_out_flat,
                        "Y": y_pred_out_flat
                    })

                    st.write("Prediction Table")
                    st.dataframe(pred_table)   
    if model =="Decision Tree":
        pass

st.markdown("Developed by: [Dewashish Dwivedi] for more info visit [Linkedin](https://www.linkedin.com/in/dewashish-dwivedi-806a92206/)")

