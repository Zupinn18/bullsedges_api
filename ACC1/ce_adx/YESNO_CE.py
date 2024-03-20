import json
from datetime import datetime, timedelta
from pytz import timezone
from time import sleep
import pandas as pd
import dash
import os
import csv
# from telegram import Bot  # Import the Bot class from the telegram module
# from telegram.error import TelegramError
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
COLLECTION_NAME = "66690_ce"

client = MongoClient(MONGO_CONNECTION_STRING)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
# data_df = pd.read_csv('data.csv')
# user_id = data_df['User ID'].iloc[-1]
# api_key = data_df['API Key'].iloc[-1]
# instrument_type = data_df['Instrument Type'].iloc[-1]
# token_number = data_df['Token Number'].iloc[-1]


# Define your AliceBlue user ID and API key
user_id = 'AB093838'
api_key = 'cy5uYssgegMaUOoyWy0VGLBA6FsmbxYd0jNkajvBVJuEV9McAM3o0o2yG6Z4fEFYUGtTggJYGu5lgK89HumH3nBLbxsLjgplbodFHDLYeXX0jGQ5CUuGtDvYKSEzWSMk'

# Initialize AliceBlue connection
alice = Aliceblue(user_id=user_id, api_key=api_key)

# Print AliceBlue session ID
print(alice.get_session_id())

# Initialize variables for WebSocket communicatdion
lp = 0
socket_opened = False
subscribe_flag = False
subscribe_list = []
unsubscribe_list = []
data_list = []  # List to store the received data
df = pd.DataFrame(columns=["timestamp", "lp"])  # Initialize an empty DataFrame for storing the data
# File paths for saving data and graph
data_file_path = "66690PE.csv"

graph_file_path = "66690PE.html"

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
subscribe_list = [alice.get_instrument_by_token("NFO", 66690)]
alice.subscribe(subscribe_list)
print(datetime.now())
sleep(10)
print(datetime.now())



def calculate_heikin_ashi(data):
    ha_open = 0.5 * (data['open'].shift() + data['close'].shift())
    ha_close = 0.25 * (data['open'] + data['high'] + data['low'] + data['close'])
    ha_high = data[['high', 'open', 'close']].max(axis=1)
    ha_low = data[['low', 'open', 'close']].min(axis=1)

    ha_data = pd.DataFrame({'open': ha_open, 'high': ha_high, 'low': ha_low, 'close': ha_close})
    
    for i in range(len(ha_data)):
        if i == 0:
            ha_data.iat[0, 0] = round(((data['open'].iloc[0] + data['close'].iloc[0]) / 2), 2)
        else:
            ha_data.iat[i, 0] = round(((ha_data.iat[i-1, 0] + ha_data.iat[i-1, 3]) / 2), 2)

    ha_data['high'] = ha_data[['high', 'open', 'close']].max(axis=1)
    ha_data['low'] = ha_data[['low', 'open', 'close']].min(axis=1)
    ha_data['close'] = round(0.25 * (data['open'] + data['close'] + data['high'] + data['low']), 2)

    ha_data['mark'] = ''
    label_data = []

    # Initialize state
    consecutive_green_candles = 0
    prev_yes_open = None
    prev_green_low = None
    prev_green_high = None 
    no_confirmed = True
    last_yes_high = None
    last_updated_price = None  # Initialize last_updated_price here

    for i in range(1, len(ha_data)):
        seven_updated = False

        if ha_data['close'].iloc[i - 1] > ha_data['open'].iloc[i - 1] and ha_data['close'].iloc[i] > ha_data['open'].iloc[i]:
            if consecutive_green_candles == 0:
                consecutive_green_candles = 1
                prev_yes_open = data['open'].iloc[i]
                prev_green_high = ha_data['high'].iloc[i]

                ha_data.at[ha_data.index[i], 'mark'] = 'YES'
                label_data.append(('YES', ha_data.index[i], data['open'].iloc[i], None))
                last_yes_high = data['open'].iloc[i]
              
                trade_book_data = alice.get_trade_book()
                if trade_book_data and not isinstance(trade_book_data, dict):
                    trade_book_data = {'Price': [trade_book_data]}  # Convert scalar value to list inside a dictionary
                trade_book_df = pd.DataFrame(trade_book_data, index=[0])
                trade_book_df.to_csv('trade_book_data.csv', index=False)
                trade_book = pd.read_csv('trade_book_data.csv')  # Assuming 'trade_book.csv' contains the trade book data
                
                if not trade_book.empty:
                    if 'Price' in trade_book.columns:
                        last_updated_price = float(trade_book['Price'].iloc[-1])
                        if data['high'].iloc[i] > last_updated_price + 7:
                            ha_data.at[ha_data.index[i], 'mark'] = 'seven'
                            label_data.append(('seven', ha_data.index[i], data['high'].iloc[i], None))
                            seven_updated = True
                        else:
                            seven_updated = False
                    else:
                        print("Column 'Price' not found in trade book data.")
                else:
                    print("Trade book data is empty or not in the expected format.")

        elif ha_data['close'].iloc[i - 1] > ha_data['open'].iloc[i - 1] and ha_data['close'].iloc[i] < ha_data['open'].iloc[i]:
            if consecutive_green_candles > 0:
                if no_confirmed:
                    ha_data.at[ha_data.index[i], 'mark'] = 'NO'
                    if prev_yes_open is not None:
                        confirmed_no_closing = ha_data['close'].iloc[i]
                        diff = prev_yes_open - confirmed_no_closing
                        label_data.append(('NO', ha_data.index[i], confirmed_no_closing, diff))
                    no_confirmed = True
                    consecutive_green_candles = 0
                    last_yes_high = None

        if not seven_updated and ha_data.at[ha_data.index[i], 'mark'] != 'seven':
            if last_yes_high is not None and last_updated_price is not None:
                if data['high'].iloc[i] > last_updated_price + 7:
                    ha_data.at[ha_data.index[i], 'mark'] = 'seven'
                    label_data.append(('seven', ha_data.index[i], data['high'].iloc[i], None))

    ha_data['Difference'] = ha_data['open'] - ha_data['close']

    label_csv_filename = 'label_66690_CE.csv'
    try:
        with open(label_csv_filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Label', 'Timestamp', 'Value', 'Difference'])
            csv_writer.writerows(label_data)
        print(f'Labels saved to {label_csv_filename}')
    except Exception as e:
        print(f'Error saving labels: {e}')

    return ha_data
# def update_label_trade_book(trade_book_data):
#     label_trade_book_filename = 'trade_book_ce.csv'

#     # Check if the file already exists
#     file_exists = os.path.isfile(label_trade_book_filename)
#     if not file_exists:
#         with open(label_trade_book_filename, 'w', newline='') as new_csv_file:
#             csv_writer = csv.DictWriter(new_csv_file, fieldnames=trade_book_data[0].keys())
#             csv_writer.writeheader()

#     try:
#         with open(label_trade_book_filename, 'a', newline='') as csv_file:
#             csv_writer = csv.DictWriter(csv_file, fieldnames=trade_book_data[0].keys())
#             existing_data = set()
#             if file_exists:
#                 with open(label_trade_book_filename, 'r') as existing_file:
#                     existing_reader = csv.DictReader(existing_file)
#                     for row in existing_reader:
#                         existing_data.add((row['Timestamp'], row['Price']))  # Assuming Timestamp and Price are the unique identifiers

#             # Append new data if it doesn't already exist
#             for entry in trade_book_data:
#                 if 'Price' in entry:
#                     updated_price = float(entry['Price'])
#                     entry['Price'] = str(updated_price)
#                 if ('Timestamp' in entry) and ('Price' in entry) and (entry['Timestamp'], entry['Price']) not in existing_data:
#                     print(f'Writing to CSV: {entry}')  # Debugging statement
#                     csv_writer.writerow(entry)

#         print(f'Label trade book updated and saved to {label_trade_book_filename}')
#     except FileNotFoundError as file_err:
#         print(f'Error: {file_err}. File not found.')
#     except Exception as e:
#         print(f'Error updating label trade book: {e}')


app = dash.Dash(__name__)
server = app.server

# Add a new dropdown for selecting candle type
candle_type_dropdown = dcc.Dropdown(
    id='candle-type-dropdown',
    options=[
        {'label': 'Normal Candlestick', 'value': 'normal'},
        {'label': 'Heikin Ashi', 'value': 'heikin_ashi'},
    ],
    value='normal',
    style={'width': '50%'},
)

# Add a new graph for Heikin Ashi candlestick
heikin_ashi_graph = dcc.Graph(id='heikin-ashi-graph')
# adx_graph = dcc.Graph(id='adx-graph')
# heikin_ashi_adx_graph = dcc.Graph(id='heikin-ashi-adx-graph')

# Update layout to include the new graph component
app.layout = html.Div([
    candle_type_dropdown,
    dcc.Graph(id='candlestick-graph'),
    heikin_ashi_graph,
    # adx_graph,
    # heikin_ashi_adx_graph,  # Add the new graph component
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

# Callback to update the candlestick graphs based on selected time frame and candle type
# Callback to update the candlestick graphs based on selected time frame and candle type
@app.callback(
    [Output('candlestick-graph', 'figure'),
     Output('heikin-ashi-graph', 'figure')],
    #  Output('adx-graph', 'figure'),
    #  Output('heikin-ashi-adx-graph', 'figure')],  # Add output for the new graph
    [Input('update-interval', 'n_intervals'),
     Input('candlestick-graph', 'relayoutData')],
    [State('timeframe-dropdown', 'value'),
     State('candle-type-dropdown', 'value')]
)
def update_graph_callback(n, relayoutData, selected_timeframe, selected_candle_type):
    data = collection.find({}, {'_id': 0}).sort('timestamp')
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    df.set_index('timestamp', inplace=True)

    # Drop non-numeric values in the 'lp' column
    df['lp'] = pd.to_numeric(df['lp'], errors='coerce')
    df = df.dropna(subset=['lp'])

    # Handle xaxis range if available in relayoutData
    # Handle xaxis range if available in relayoutData
    if 'xaxis.range' in relayoutData:
        xaxis_range = relayoutData['xaxis.range']
    else:
        # Adjust xaxis_range to include the last 4320 minutes
        end_time = df.index[-1]
        start_time = end_time - pd.Timedelta(hours=72)
        xaxis_range = [start_time, end_time]

    # Filter data based on xaxis range
    filtered_data = df[(df.index >= xaxis_range[0]) & (df.index <= xaxis_range[1])]


    # Resample data for candlestick graph based on the selected timeframe
    resampled_data = filtered_data["lp"].resample(selected_timeframe).ohlc()
    resampled_data = resampled_data.dropna()

    # Calculate ADX for the resampled data
    # adx_data = calculate_adx(resampled_data)

    # Create normal candlestick figure
    normal_candlestick_fig = go.Figure(data=[go.Candlestick(x=resampled_data.index,
                                                             open=resampled_data['open'],
                                                             high=resampled_data['high'],
                                                             low=resampled_data['low'],
                                                             close=resampled_data['close'])])
    normal_candlestick_fig.update_xaxes(type='category', tickformat='%H:%M')
    normal_candlestick_fig.update_layout(title=f'Real-Time Candlestick Chart ({selected_timeframe})',
                                  xaxis_title='Time',
                                  yaxis_title='Price',
                                  xaxis_rangeslider_visible=False,
                                  template='plotly')

    # Create Heikin Ashi candlestick figure
    if selected_candle_type == 'heikin_ashi':
        heikin_ashi_data = calculate_heikin_ashi(resampled_data)
        heikin_ashi_fig = go.Figure(data=[go.Candlestick(x=heikin_ashi_data.index,
                                                         open=heikin_ashi_data['open'],
                                                         high=heikin_ashi_data['high'],
                                                         low=heikin_ashi_data['low'],
                                                         close=heikin_ashi_data['close'])])
        heikin_ashi_fig.update_xaxes(type='category', tickformat='%H:%M')
        heikin_ashi_fig.update_layout(title=f'Real-Time Heikin Ashi Candlestick Chart ({selected_timeframe})',
                                  xaxis_title='Time',
                                  yaxis_title='Price',
                                  xaxis_rangeslider_visible=False,
                                  template='plotly')
    else:
        heikin_ashi_fig = go.Figure()

    return normal_candlestick_fig, heikin_ashi_fig
''', adx_fig , heikin_ashi_adx_fig'''


if __name__ == '__main__':
    app.run_server(debug=True, port=8004)
