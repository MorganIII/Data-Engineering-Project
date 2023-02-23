import plotly.express as px 
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import math
from plotly.subplots import make_subplots
from milestone_1 import handling_missing_values ,handling_outliers,handling_unclean_data
import pyarrow.parquet as pq

def plot_missing_values(df):
    # Create a list of tuples containing the column names and the number of missing values
    data = [(col, df[col].isnull().sum()) for col in df.columns]
    # Sort the list of tuples by the number of missing values, in descending order
    data.sort(key=lambda x: x[1], reverse=True)

    # Extract the column names and the number of missing values into separate lists
    columns, values = zip(*data)

    # Create a bar chart using plotly
    fig = go.Figure([go.Bar(x=columns, y=values)])
    fig.update_layout(title='Missing Values in Dataset', xaxis_title='Columns', yaxis_title='Missing Values')
    return fig

def histogram(df, column_name):
    fig = px.histogram(df, x=column_name, nbins=50, title=f' {column_name}', 
                       labels={column_name: column_name})
    return fig
    
def create_dashboard(filename1,filename2):
    na_vals = ['NA', 'Missing','None']
    df1 = pd.read_parquet(filename2,engine='pyarrow')
    df_accident = pd.DataFrame(df1)
    df_accident.replace(na_vals, np.nan, inplace=True)
    df_accident.set_index('accident_index', inplace=True)
    df2 = pd.read_csv(filename1)
    app = Dash()
    app.layout = html.Div([
        html.H1('Web App for UK Accidents', style={'textAlign': 'center'}),
        html.Br(),
        html.Div(),
        html.H1('The distribution of Missing Values in the UK Accidents dataset', style={'textAlign': 'center'}),
        dcc.Graph(figure=plot_missing_values(df_accident)),
        html.Br(),
        html.Div(),
        html.H1('the distribution of the accident severity', style={'textAlign': 'center'}),
        dcc.Graph(figure=histogram(df2, 'accident_severity')),
        html.Br(),
        html.Div(),
        html.H1('The frequent weather conditions happened at the time of the accident ', style={'textAlign': 'center'}),
        dcc.Graph(figure=histogram(df2, 'weather_conditions')),
        html.Br(),
        html.Div(),
        html.H1('The road type that accident happened on it', style={'textAlign': 'center'}),
        dcc.Graph(figure=histogram(df2, 'road_type')),
        html.Br(),
        html.Div(),
        html.H1('The most frequent day of the week that accidents happened on it', style={'textAlign': 'center'}),
        dcc.Graph(figure=histogram(df2, 'day_of_week')),
    ])
    app.run_server(host='0.0.0.0')