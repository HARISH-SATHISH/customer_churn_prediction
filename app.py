import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from flask import Flask, request, render_template
from dash import Dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px
import shap

app = Flask(__name__)

# Initialize the Dash app
dash_app = Dash(__name__, server=app, url_base_pathname='/dashboard/')

# Load your CSV data
df = pd.read_csv('selected feaures.csv')

# Create example figures using Plotly Express
fig1 = px.bar(df, x='tenure', y='MonthlyCharges', title='Monthly Charges by Tenure')
fig2 = px.pie(df, names='Dependents', title='Dependents Distribution')
fig3 = px.histogram(df, x='TotalCharges', nbins=50, title='Total Charges Distribution')

# Define the layout of the Dash app
dash_app.layout = html.Div(children=[
    html.H1(children='Customer Churn Dashboard'),

    dcc.Graph(
        id='bar-graph',
        figure=fig1
    ),

    dcc.Graph(
        id='pie-chart',
        figure=fig2
    ),

    dcc.Graph(
        id='histogram',
        figure=fig3
    )
])

@app.route("/")
def home_page():
    return render_template('home.html')

@app.route("/", methods=['POST'])
def predict():
    """ Selected feature are Dependents, tenure, OnlineSecurity,
        OnlineBackup, DeviceProtection, TechSupport, Contract,
        PaperlessBilling, MonthlyCharges, TotalCharges """

    Dependents = request.form['Dependents']
    tenure = float(request.form['tenure'])
    OnlineSecurity = request.form['OnlineSecurity']
    OnlineBackup = request.form['OnlineBackup']
    DeviceProtection = request.form['DeviceProtection']
    TechSupport = request.form['TechSupport']
    Contract = request.form['Contract']
    PaperlessBilling = request.form['PaperlessBilling']
    MonthlyCharges = float(request.form['MonthlyCharges'])
    TotalCharges = float(request.form['TotalCharges'])

    # Load the model using joblib
    model = joblib.load('Model.sav')
    data = [[Dependents, tenure, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, Contract, PaperlessBilling, MonthlyCharges, TotalCharges]]
    df_input = pd.DataFrame(data, columns=['Dependents', 'tenure', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'Contract',
        'PaperlessBilling', 'MonthlyCharges', 'TotalCharges'])

    categorical_feature = {feature for feature in df_input.columns if df_input[feature].dtypes == 'O'}

    encoder = LabelEncoder()
    for feature in categorical_feature:
        df_input[feature] = encoder.fit_transform(df_input[feature])

    single = model.predict(df_input)
    probability = model.predict_proba(df_input)[:, 1]
    probability = probability * 100

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_input)
    feature_importance = pd.DataFrame(list(zip(df_input.columns, np.abs(shap_values).mean(axis=0))),
                                      columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)

    if single == 1:
        op1 = "This Customer is likely to be Churned!"
        op2 = f"Confidence level is {np.round(probability[0], 2)}"
    else:
        op1 = "This Customer is likely to Continue!"
        op2 = f"Confidence level is {np.round(probability[0], 2)}"

    return render_template("home.html", op1=op1, op2=op2,
                           feature_importance=feature_importance.to_dict(orient='records'),
                           Dependents=request.form['Dependents'],
                           tenure=request.form['tenure'],
                           OnlineSecurity=request.form['OnlineSecurity'],
                           OnlineBackup=request.form['OnlineBackup'],
                           DeviceProtection=request.form['DeviceProtection'],
                           TechSupport=request.form['TechSupport'],
                           Contract=request.form['Contract'],
                           PaperlessBilling=request.form['PaperlessBilling'],
                           MonthlyCharges=request.form['MonthlyCharges'],
                           TotalCharges=request.form['TotalCharges'])

if __name__ == '__main__':
    app.run(debug=True)
