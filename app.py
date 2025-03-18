import streamlit as st
import pandas as pd
import l_reg as lr
import dec_tree as dt
import matplotlib.pyplot as plt

st.header("Machine Learning Web App")
st.write("Upload CSV file for training")
training = st.file_uploader("Select the CSV file")
st.write("upload the csv file for prediction")
pred = st.file_uploader("select the csv for prediction")
st.write("Select the model for prediction")
model = st.selectbox("Select the model", ["None","Decision Tree", "Linear Regression"])

if training is not None:
    
    fl = pd.read_csv(training)
    f2 = pd.read_csv(pred)
    

    if model == "None":
        pass


    if model == "Linear Regression":
        x, y, y_pred_train, lr_model,y_pred_out,x_out= lr.lr(fl,f2)

        st.write("Trained Model Graph")
        fig_train, ax_train = plt.subplots()
        ax_train.scatter(x, y, color='blue', label='Actual')
        ax_train.plot(x, y_pred_train, color='red', label='Predicted')
        ax_train.set_title("Linear Regression Trained Model Graph")
        ax_train.set_xlabel("X")
        ax_train.set_ylabel("Y")
        ax_train.legend(["Predicted", "Actual"])
        st.pyplot(fig_train)

        st.write("Prediction Model Graph")
        fig_pred , ax_pred = plt.subplots()
        ax_pred.scatter(x_out,y_pred_out,color='blue',label = 'Predicted')
        ax_pred.plot(x_out,y_pred_out,color = "Red",label="Prediction")
        ax_pred.set_title("Linear Regression Salary Prediction")
        ax_pred.set_xlabel("X")
        ax_pred.set_ylabel("Y")
        st.pyplot(fig_pred)

        x_out_flat = x_out.values.flatten()
        y_pred_out_flat = y_pred_out.flatten()

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