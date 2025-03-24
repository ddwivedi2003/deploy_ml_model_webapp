import streamlit as st
import pandas as pd
import l_reg as lr
import dec_tree as dt
import matplotlib.pyplot as plt

st.title("Machine Learning Web App")
st.write("Upload CSV file for training")
training = st.file_uploader("Select the CSV file")
st.write("Select the model for prediction")
model = st.selectbox("Select the model", ["None", "Decision Tree", "Linear Regression"])

# Initialize session state variables
if "verify" not in st.session_state:
    st.session_state.verify = 0
if "lr_model" not in st.session_state:
    st.session_state.lr_model = None
if "dt_model" not in st.session_state:
    st.session_state.dt_model = None

if training is not None:
    fl = pd.read_csv(training)

    if model == "None":
        st.write("Please select a model to proceed.")

    if model == "Linear Regression":
        st.header("Linear Regression Model")
        st.write("Train the model")
        button_train_lr = st.button("Train the model (Linear Regression)")
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

                    pred_table = pd.DataFrame({
                        "X": x_1.values.flatten(),
                        "Y": y_pred.flatten()
                    })

                    st.write("Prediction Table")
                    st.dataframe(pred_table)

    if model == "Decision Tree":
        st.header("Decision Tree Model")
        st.write("Train the model")

        # Input for optional parameters
        target_column = st.text_input("Enter the target column (leave blank for auto-detection):")
        numerical_columns = st.text_input("Enter numerical columns to scale (comma-separated, optional):")
        test_size = st.slider("Select test size (proportion of test data):", 0.1, 0.5, 0.2)
        random_state = st.number_input("Enter random state (default: 42):", value=42, step=1)

        # Convert numerical columns input to a list
        numerical_columns = [col.strip() for col in numerical_columns.split(",")] if numerical_columns else None

        button_train_dt = st.button("Train the model (Decision Tree)")
        if button_train_dt:
            try:
                # Train the Decision Tree model
                dt_model = dt.dt(
                    fl,
                    target_column=target_column if target_column else None,
                    numerical_columns=numerical_columns,
                    test_size=test_size,
                    random_state=random_state,
                )
                st.session_state.dt_model = dt_model  # Save the trained model in session state
                st.write("Model trained successfully!")

            except ValueError as e:
                st.error(f"Error: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

        if st.session_state.dt_model is not None:
            st.write("Choose the option for prediction")
            dt_pred = st.selectbox("Select the option for prediction", ["Upload CSV for Prediction", "Enter the value for Prediction"])

            if dt_pred == "Enter the value for Prediction":
                st.write("Enter the value for prediction")
                x_in = st.text_input("Enter the feature values (comma-separated):")
                button_pred = st.button("Predict")
                if button_pred:
                    try:
                        import numpy as np
                        x_in_array = np.array([float(val) for val in x_in.split(",")]).reshape(1, -1)
                        y_pred = st.session_state.dt_model.predict(x_in_array)
                        st.write("The predicted value is:", y_pred[0])
                    except Exception as e:
                        st.error(f"Error in prediction: {str(e)}")

            if dt_pred == "Upload CSV for Prediction":
                st.write("Upload the CSV file for prediction")
                pred = st.file_uploader("Select the CSV for prediction")
                button_pred_csv = st.button("Predict from CSV")
                if button_pred_csv and pred is not None:
                    try:
                        f2 = pd.read_csv(pred)
                        y_pred = st.session_state.dt_model.predict(f2)

                        pred_table = pd.DataFrame({
                            "Input Features": f2.values.tolist(),
                            "Predicted Output": y_pred,
                        })
                        st.write("Prediction Table")
                        st.dataframe(pred_table)
                    except Exception as e:
                        st.error(f"Error in prediction: {str(e)}")

st.markdown("Developed by: [Dewashish Dwivedi] for more info visit [LinkedIn](https://www.linkedin.com/in/dewashish-dwivedi-806a92206/)")