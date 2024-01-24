import json
from datetime import datetime, timedelta
from pytz import timezone
from time import sleep
import pandas as pd
from plotly.subplots import make_subplots
import dash
import os
import csv
from dash import dcc
from dash import html
from dash.dependencies import Output, Input, State
import streamlit as st
import plotly.graph_objects as go
from pya3 import *
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import dash_table
import pymongo
from pymongo import MongoClient
import ta
from ta.utils import dropna



# Replace these with your actual MongoDB connection details
MONGO_CONNECTION_STRING = "mongodb://localhost:27017/"
DB_NAME = "banknifty"
COLLECTION_NAME = "26009CE"

client = MongoClient(MONGO_CONNECTION_STRING)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Define your AliceBlue user ID and API key
user_id = 'AB093838'
api_key = 'cy5uYssgegMaUOoyWy0VGLBA6FsmbxYd0jNkajvBVJuEV9McAM3o0o2yG6Z4fEFYUGtTggJYGu5lgK89HumH3nBLbxsLjgplbodFHDLYeXX0jGQ5CUuGtDvYKSEzWSMk'

# Initialize AliceBlue connection
alice = Aliceblue(user_id=user_id, api_key=api_key)

# Print AliceBlue session ID
print(alice.get_session_id())

# Initialize variables for WebSocket communication
lp = 0
socket_opened = False
subscribe_flag = False
subscribe_list = []
unsubscribe_list = []
data_list = []  # List to store the received data
df = pd.DataFrame(columns=["timestamp", "lp"])  # Initialize an empty DataFrame for storing the data
# File paths for saving data and graph
data_file_path = "26009CE.csv"

graph_file_path = "26009CE.html"

# Check if the data file exists
if os.path.exists(data_file_path):
    # Load existing data from the CSV file
    df = pd.read_csv(data_file_path, index_col="timestamp", parse_dates=True)
else:
    df = pd.DataFrame(columns=["timestamp", "lp"])  # Initialize an empty DataFrame for storing the data


all_trend_lines = []
trend_line_visibility = []


# Callback functions for WebSocket connection
def socket_open():
    print("Connected")
    global socket_opened
    socket_opened = True
    if subscribe_flag:
        alice.subscribe(subscribe_list)


def socket_close():
    global socket_opened, lp
    socket_opened = False
    lp = 0
    print("Closed")


def socket_error(message):
    global lp
    lp = 0
    print("Error:", message)

consecutive_green_candles = 0
previous_candle_green = False
label_data = []

# Callback function for receiving data from WebSocket
def feed_data(message):
    global lp, subscribe_flag, data_list, consecutive_green_candles
    feed_message = json.loads(message)
    if feed_message["t"] == "ck":
        print("Connection Acknowledgement status: %s (Websocket Connected)" % feed_message["s"])
        subscribe_flag = True
        print("subscribe_flag:", subscribe_flag)
        print("-------------------------------------------------------------------------------")
        pass
    elif feed_message["t"] == "tk":
        print("Token Acknowledgement status: %s" % feed_message)
        print("-------------------------------------------------------------------------------")
        pass
    else:
        print("Feed:", feed_message)
        if 'lp' in feed_message:
            timestamp = datetime.now(timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
            feed_message['timestamp'] = timestamp
            data_list.append(feed_message)  # Append the received data to the list
            # Insert the data into MongoDB
            collection.insert_one(feed_message)

            # Update marking information only for Heikin Ashi candles
            if len(df) >= 2 and df['mark'].iloc[-1] == '' and feed_message['t'] == 'c':
                if (df['close'].iloc[-2] > df['open'].iloc[-2] and df['close'].iloc[-1] > df['open'].iloc[-1]):
                    if previous_candle_green:  # Check if the previous candle was green
                        consecutive_green_candles += 1
                        if consecutive_green_candles == 2:  # Mark "YES" only on the second consecutive green candle
                            df.at[df.index[-1], 'mark'] = 'YES'
                    else:
                        consecutive_green_candles = 1
                    previous_candle_green = True
                elif (df['close'].iloc[-2] > df['open'].iloc[-2] and df['close'].iloc[-1] < df['open'].iloc[-1]):
                    consecutive_green_candles = 0
                    previous_candle_green = False
                    df.at[df.index[-1], 'mark'] = 'NO'

        else:
            print("'lp' key not found in feed message.")


# Connect to AliceBlue

# Socket Connection Request
alice.start_websocket(socket_open_callback=socket_open, socket_close_callback=socket_close,
                      socket_error_callback=socket_error, subscription_callback=feed_data, run_in_background=True,
                      market_depth=False)

while not socket_opened:
    pass

# Subscribe to Tata Motors
subscribe_list = [alice.get_instrument_by_token('NSE', 26009)]
alice.subscribe(subscribe_list)
print(datetime.now())
sleep(10)
print(datetime.now())

def calculate_adx(df):
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14, fillna=True).adx()
    df['di_plus'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14, fillna=True).adx_pos()
    df['di_minus'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14, fillna=True).adx_neg()

    return df

# def generate_signals(df):
#     df['buy_signal'] = ((df['di_plus'] > df['di_minus']) & (df['adx'] > 25))
#     df['sell_signal'] = ((df['di_plus'] < df['di_minus']) & (df['adx'] > 25))

#     return df

app = dash.Dash(__name__)
server = app.server
# Define layout of the app
app.layout = html.Div([
    dcc.Graph(id='candlestick-graph'),
    dcc.Graph(id='adx-graph'),
    dcc.Dropdown(
        id='timeframe-dropdown',
        options=[
            {'label': '1 min', 'value': '1T'},
            {'label': '3 min', 'value': '3T'},
            {'label': '5 min', 'value': '5T'},
            {'label': '10 min', 'value': '10T'},
            {'label': '15 min', 'value': '15T'},
            {'label': '30 min', 'value': '30T'},
            {'label': '1 hr', 'value': '1H'},
        ],
        value='5T',  # Default to 5 min
        style={'width': '50%'},
    ),
    dcc.Interval(id='update-interval', interval=5*1000, n_intervals=0)
])

# Callback to update the candlestick graph and ADX graph
@app.callback(
    [Output('candlestick-graph', 'figure'),
     Output('adx-graph', 'figure')],
    [Input('update-interval', 'n_intervals')],
    [State('timeframe-dropdown', 'value')]
)
def update_graph_callback(n, selected_timeframe):
    data = collection.find({}, {'_id': 0}).sort('timestamp')
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    df.set_index('timestamp', inplace=True)

    # Drop non-numeric values in the 'lp' column
    df['lp'] = pd.to_numeric(df['lp'], errors='coerce')
    df = df.dropna(subset=['lp'])

    # Resample data for candlestick graph based on the selected timeframe
    resampled_data = df["lp"].resample(selected_timeframe).ohlc()
    resampled_data = resampled_data.dropna()

    # Calculate ADX
    resampled_data = calculate_adx(resampled_data)

    # Generate buy/sell signals
    # resampled_data = generate_signals(resampled_data)

    # Create candlestick figure
    candlestick_fig = go.Figure(data=[go.Candlestick(x=resampled_data.index,
                                                      open=resampled_data['open'],
                                                      high=resampled_data['high'],
                                                      low=resampled_data['low'],
                                                      close=resampled_data['close'])])

    candlestick_fig.update_xaxes(type='category', tickformat='%H:%M')
    candlestick_fig.update_layout(title=f'Real-Time Candlestick Chart with ADX ({selected_timeframe})',
                                  xaxis_title='Time',
                                  yaxis_title='Price',
                                  xaxis_rangeslider_visible=False,
                                  template='plotly')

    # Create ADX figure
    adx_fig = go.Figure()

    adx_fig.add_trace(go.Scatter(x=resampled_data.index, y=resampled_data['adx'], mode='lines', name='ADX',
                                 line=dict(color='purple', width=2)))

    adx_fig.add_trace(go.Scatter(x=resampled_data.index, y=resampled_data['di_plus'], mode='lines', name='+DI',
                                 line=dict(color='green', width=2)))

    adx_fig.add_trace(go.Scatter(x=resampled_data.index, y=resampled_data['di_minus'], mode='lines', name='-DI',
                                 line=dict(color='red', width=2)))

    # # Add buy signals
    # buy_signals = resampled_data[resampled_data['buy_signal']]
    # adx_fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['adx'], mode='markers', name='Buy Signal',
    #                              marker=dict(color='green', size=8), showlegend=False))

    # # Add sell signals
    # sell_signals = resampled_data[resampled_data['sell_signal']]
    # adx_fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['adx'], mode='markers', name='Sell Signal',
    #                              marker=dict(color='red', size=8), showlegend=False))

    adx_fig.update_layout(title=f'Real-Time ADX Chart ({selected_timeframe})',
                          xaxis_title='Time',
                          yaxis_title='Value',
                          xaxis_rangeslider_visible=False,
                          template='plotly')

    # Set the x-axis range of the ADX graph to match the candlestick graph
    adx_fig.update_xaxes(type='category', tickformat='%H:%M', range=[resampled_data.index[0], resampled_data.index[-1]])

    return candlestick_fig, adx_fig

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload=False)

    # Save the graphs locally
    candlestick_fig, adx_fig = update_graph_callback(0, '5T')  # Manually trigger the callback to get the latest graphs

    candlestick_fig.write_html("candlestick_graph.html")
    adx_fig.write_html("adx_graph.html")