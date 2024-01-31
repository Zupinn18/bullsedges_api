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
import plotly.graph_objects as go



# Replace these with your actual MongoDB connection details
MONGO_CONNECTION_STRING = "mongodb://localhost:27017/"
DB_NAME = "banknifty"
COLLECTION_NAME = "35507PE_15"

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
data_file_path = "35507PE.csv"

graph_file_path = "35507PE.html"

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
subscribe_list = [alice.get_instrument_by_token('NFO', 35507)]
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
    prev_green_low = None  # Track the low of the previous green candle
    prev_green_high = None 
    no_confirmed = True  # Flag to track if "NO" has been confirmed
    # yes_updated = False
    label_data = []  # Create an empty list to store label data

    for i in range(1, len(ha_data)):
        if (ha_data['close'].iloc[i - 1] > ha_data['open'].iloc[i - 1] and
                ha_data['close'].iloc[i] > ha_data['open'].iloc[i]):

            consecutive_green_candles += 1
            prev_yes_open = data['open'].iloc[i]  # Update previous "YES" open value
            prev_green_high = ha_data['high'].iloc[i]  # Update the high of the previous green candle

            if consecutive_green_candles == 1:
                # Check if the current candle or the next candles satisfy the condition
                for j in range(i, len(ha_data)):
                    if ha_data['high'].iloc[j] > ha_data['high'].iloc[j - 1]:
                        ha_data.at[ha_data.index[j], 'mark'] = 'YES'
                        label_data.append(('YES', ha_data.index[j], data['open'].iloc[j], None))
                        # Check if the current closing price is 7 points higher than the previous "YES" high
                        if prev_green_high is not None and data['high'].iloc[j] > ha_data['high'].iloc[j - 1] + 7:
                            label_data.append(('seven', ha_data.index[j], data['high'].iloc[j], None))
                        break  # Exit the loop once the condition is satisfied
                else:
                # The condition was not met for consecutive candles, move on to the next candle
                    consecutive_green_candles = 0
        elif (ha_data['close'].iloc[i - 1] > ha_data['open'].iloc[i - 1] and
            ha_data['close'].iloc[i] < ha_data['open'].iloc[i]):

            # Check if the previous candle was green
            if consecutive_green_candles > 0:
                ha_data.at[ha_data.index[i], 'mark'] = 'NO'
                if prev_yes_open is not None:
                    if no_confirmed:  # Calculate difference only if "NO" is confirmed
                        confirmed_no_closing = ha_data['close'].iloc[i]  # Store confirmed "NO" closing value
                        diff = prev_yes_open - confirmed_no_closing  # Corrected difference calculation
                        label_data.append(('NO', ha_data.index[i], confirmed_no_closing, diff))

                        if ha_data['close'].iloc[i] < ha_data['open'].iloc[i]:
                            prev_green_low = ha_data['low'].iloc[i]  # Update the low of the previous green candle

                        if i + 1 < len(ha_data) and ha_data['close'].iloc[i + 1] < ha_data['open'].iloc[i + 1]:
                            label_data.append(('RED', ha_data.index[i + 1], ha_data['close'].iloc[i + 1], None))
                        # yes_updated = False  # Reset 'YES' update flag
                    else:
                        ha_data.at[ha_data.index[i], 'mark'] = ''
                        print("Warning: NO not confirmed yet, skipping difference calculation")
                else:
                    ha_data.at[ha_data.index[i], 'mark'] = ''
                    print("Warning: prev_yes_open is None, skipping difference calculation")

                no_confirmed = True  # "NO" is confirmed
                consecutive_green_candles = 0  # Reset consecutive_green_candles

        

    # Calculate the difference and add it to the DataFrame
    ha_data['Difference'] = ha_data['open'] - ha_data['close']

    label_csv_filename = 'label_35507PE_15.csv'
    try:
        with open(label_csv_filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Label', 'Timestamp', 'Value', 'Difference'])
            csv_writer.writerows(label_data)
        print(f'Labels saved to {label_csv_filename}')
    except Exception as e:
        print(f'Error saving labels: {e}')

    return ha_data






def calculate_supertrend(data, atr_period=12, factor=2.0, multiplier=2.0):
    data = data.copy()  # Create a copy of the data DataFrame

    close = data['close']
    high = data['high']
    low = data['low']

    tr = pd.DataFrame()
    tr['h-l'] = high - low
    tr['h-pc'] = abs(high - close.shift())
    tr['l-pc'] = abs(low - close.shift())
    tr['tr'] = tr.max(axis=1)

    atr = tr['tr'].rolling(atr_period).mean()

    median_price = (high + low) / 2
    data['upper_band'] = median_price + (multiplier * atr)
    data['lower_band'] = median_price - (multiplier * atr)

    supertrend = pd.Series(index=data.index)
    direction = pd.Series(index=data.index)

    supertrend.iloc[0] = data['upper_band'].iloc[0]
    direction.iloc[0] = 1

    for i in range(1, len(data)):
        if close.iloc[i] > supertrend.iloc[i - 1]:
            supertrend.iloc[i] = max(data['lower_band'].iloc[i], supertrend.iloc[i - 1])
            direction.iloc[i] = 1
        else:
            supertrend.iloc[i] = min(data['upper_band'].iloc[i], supertrend.iloc[i - 1])
            direction.iloc[i] = -1

        # Start uptrend calculation anew whenever a new uptrend begins
        if direction.iloc[i] == 1 and direction.iloc[i - 1] != 1:
            supertrend.iloc[i] = data['lower_band'].iloc[i]

        # Start downtrend calculation anew whenever a new downtrend begins
        if direction.iloc[i] == -1 and direction.iloc[i - 1] != -1:
            supertrend.iloc[i] = data['upper_band'].iloc[i]

    data['supertrend'] = supertrend  # Add the 'supertrend' column to the data DataFrame
    data['direction'] = direction  # Add the 'direction' column to the data DataFrame

    return data[['open', 'high', 'low', 'close', 'supertrend', 'direction', 'lower_band', 'upper_band']]

def calculate_trend_lines(data):
    current_trend = None
    trend_start = None
    trend_lines = []

    for i in range(len(data)):
        current_signal = data.iloc[i]

        if current_trend is None:
            current_trend = current_signal['direction']
            trend_start = current_signal.name

        if current_trend != current_signal['direction']:
            if trend_start is not None:
                trend_data = data.loc[trend_start:data.index[i - 1]]
                if len(trend_data) > 1:
                    trend_lines.append((current_trend, trend_data))

            current_trend = current_signal['direction']
            trend_start = current_signal.name

    # Handle the last trend if it's still ongoing
    if trend_start is not None and trend_start != data.index[-1]:
        trend_data = data.loc[trend_start:]
        if len(trend_data) > 1:
            trend_lines.append((current_trend, trend_data))

    return trend_lines


all_trend_lines = []



# Function to update the graph

def calculate_current_trend_lines(data):
    current_trend = None
    in_trend = False
    trend_start = None
    trend_lines = []
    buy_signals = pd.DataFrame(columns=['supertrend', 'direction'])
    sell_signals = pd.DataFrame(columns=['supertrend', 'direction'])
    # band_data = pd.DataFrame(columns=['Timestamp', 'band_type', 'band_value'])

    for i in range(len(data)):
        current_signal = data.iloc[i]

        if current_trend is None:
            current_trend = current_signal['direction']
            in_trend = True
            trend_start = current_signal.name

        if current_trend != current_signal['direction']:
            if current_signal['direction'] == 1:
                sell_signals = pd.concat([sell_signals, current_signal])
                # band_data = pd.concat([band_data, pd.DataFrame({'Timestamp': [current_signal.name], 'band_type': ['lower_band'], 'band_value': [current_signal['lower_band']]}), pd.DataFrame({'Timestamp': [current_signal.name], 'band_type': ['upper_band'], 'band_value': [np.nan]})])
            else:
                buy_signals = pd.concat([buy_signals, current_signal])
                # band_data = pd.concat([band_data, pd.DataFrame({'Timestamp': [current_signal.name], 'band_type': ['upper_band'], 'band_value': [current_signal['upper_band']]}), pd.DataFrame({'Timestamp': [current_signal.name], 'band_type': ['lower_band'], 'band_value': [np.nan]})])

            if in_trend:
                trend_data = data.loc[trend_start:data.index[i - 1]]

                if len(trend_data) > 1:
                    trend_lines.append((current_trend, trend_data))

            else:
                if current_signal['direction'] == 1 and current_trend == -1:
                    updated_trend_data = data.loc[trend_start:data.index[i]]
                    updated_supertrend_data = calculate_supertrend(updated_trend_data, factor=2.0)
                    current_signal['supertrend'] = updated_supertrend_data['supertrend'].iloc[-1]
                    current_signal['direction'] = updated_supertrend_data['direction'].iloc[-1]
                elif current_signal['direction'] == -1 and current_trend == 1:
                    updated_trend_data = data.loc[trend_start:data.index[i]]
                    updated_supertrend_data = calculate_supertrend(updated_trend_data, factor=2.0)
                    current_signal['supertrend'] = updated_supertrend_data['supertrend'].iloc[-1]
                    current_signal['direction'] = updated_supertrend_data['direction'].iloc[-1]

            current_trend = current_signal['direction']
            in_trend = False

        if not in_trend:
            if current_trend == 1:
                if not np.isnan(current_signal['upper_band']):
                    trend_start = current_signal.name
                    in_trend = True
            else:
                if not np.isnan(current_signal['lower_band']):
                    trend_start = current_signal.name
                    in_trend = True

    if in_trend:
        trend_data = data.loc[trend_start:]

        if len(trend_data) > 1:
            first_high = trend_data['high'].iloc[0]
            last_close = trend_data['close'].iloc[-1]

    # Handle the continuation of uptrend without a change in direction
    if len(trend_lines) > 0 and data.index[-1] not in trend_lines[-1][1].index and trend_lines[-1][0] == 1:
        last_trend_type, last_trend_data = trend_lines[-1]
        continuation_data = data.loc[data.index > last_trend_data.index[-1]]
        if len(continuation_data) > 1:
            updated_trend_data = pd.concat([last_trend_data.iloc[:-1], continuation_data])
            updated_supertrend_data = calculate_supertrend(updated_trend_data, factor=2.0)
            continuation_data['supertrend'] = updated_supertrend_data['supertrend'].values[-len(continuation_data):]
            continuation_data['direction'] = updated_supertrend_data['direction'].values[-len(continuation_data):]
            trend_lines[-1] = (last_trend_type, updated_trend_data)

    # Save band_data to a single CSV file
    # band_data.to_csv('band_data_CE.csv', index=False)

    return trend_lines, buy_signals, sell_signals



# Function to update the graph
def update_graph(n, interval, chart_type):
    global df, data_list, all_trend_lines

    # start_time = datetime.now()
    # while (datetime.now() - start_time).seconds < 1:
    #     pass

    data = collection.find({}, {'_id': 0}).sort('timestamp')
    df = pd.DataFrame(data)

    # Convert 'timestamp' column to datetime and set it as the index
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    df.set_index('timestamp', inplace=True)

    # Check if there is new data
    if len(data_list) > 0:
        new_df = pd.DataFrame(data_list)
        new_df['lp'] = pd.to_numeric(new_df['lp'], errors='coerce')
        new_df = new_df.dropna(subset=['lp'])
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], format='%Y-%m-%d %H:%M:%S')
        new_df.set_index('timestamp', inplace=True)
        df = pd.concat([df, new_df])

        df.to_csv(data_file_path)
        data_list = []

    df["lp"] = pd.to_numeric(df["lp"], errors="coerce")
    trading_start_time = pd.Timestamp(df.index[0].date()) + pd.Timedelta(hours=9)
    trading_end_time = pd.Timestamp(df.index[0].date()) + pd.Timedelta(hours=23)
    trading_hours_mask = (df.index.time >= trading_start_time.time()) & (df.index.time <= trading_end_time.time())
    df = df[trading_hours_mask]


    # Resample the data for the desired interval
    resampled_data = df["lp"].resample(f'{interval}T').ohlc()
    resampled_data = resampled_data.dropna()

    # Create a datetime index for the x-axis, starting from the first data point and ending at the last data point
    x = pd.date_range(start=df.index[0], end=df.index[-1], freq=f'{interval}T')

    # Plot the data using plotly
    fig = go.Figure(data=[go.Candlestick(x=x,
                                      open=resampled_data['open'],
                                      high=resampled_data['high'],
                                      low=resampled_data['low'],
                                      close=resampled_data['close'])])

    # Set x-axis label to show only the time
    fig.update_xaxes(type='category', tickformat='%H:%M')

    # Add live closing price ticker
    last_close_price = resampled_data['close'].iloc[-1]
    fig.add_trace(go.Scatter(x=[resampled_data.index[-1]],
                            y=[last_close_price],
                            mode='markers+text',
                            marker=dict(color='red', size=8),
                            text=[f'Last Close: {last_close_price:.2f}'],
                            textposition='bottom right',
                            name='Last Close'))


    # Update the layout and display the figure
    fig.update_layout(title=f'Real-Time {chart_type} Chart',
                      xaxis_title='Time',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False,
                      yaxis2=dict(overlaying='y', side='left', showgrid=False),
                      template='plotly_dark')


    if chart_type == 'heikin_ashi':
        resampled_data = calculate_heikin_ashi(resampled_data)

        fig = go.Figure()

        # Add Heikin Ashi candlesticks to the figure
        for i, candle in enumerate(resampled_data.itertuples()):
            candle_color = 'green' if candle.close > candle.open else 'red'

            # Adjust timestamp
            timestamp = candle.Index

            fig.add_trace(go.Candlestick(x=[timestamp],
                                        open=[candle.open],
                                        high=[candle.high],
                                        low=[candle.low],
                                        close=[candle.close],
                                        increasing_line_color='green',
                                        decreasing_line_color='red',
                                        name=f'Candle {i + 1}'))

            # Add "yes" or "no" label above the candle
            label_y = None
            label_text = None
            if candle.mark == 'YES':
                label_y = candle.high + 5  # Adjust this value for proper positioning
                label_text = 'Yes'
            elif candle.mark == 'NO':
                label_y = candle.low - 15  # Adjust this value for proper positioning
                label_text = 'No'

            if label_y is not None:
                fig.add_annotation(
                    go.layout.Annotation(
                        x=timestamp,
                        y=label_y,
                        text=label_text,
                        showarrow=False,
                        font=dict(color='black', size=12),
                    )
                )


    # Calculate the Supertrend and get the direction from the result
    supertrend_data = calculate_supertrend(resampled_data, factor=2.0)  # Use the new factor parameter
    resampled_data = supertrend_data  # Update resampled_data with the DataFrame returned from calculate_supertrend

    # Add 'volume' column with default value if it doesn't exist in resampled_data
    if 'volume' not in resampled_data:
        resampled_data['volume'] = 0

    # Create a new figure (initialize or update existing figure)
    if 'fig' not in globals():
        fig = plot_candlestick(resampled_data)
        all_trend_lines = []  # Initialize the list of trend lines for the new figure
    else:
        fig = go.Figure()
        all_trend_lines = []

    # Calculate the current trend lines using the updated Supertrend data
    trend_lines, buy_signals, sell_signals = calculate_current_trend_lines(resampled_data)
    trend_lines = calculate_trend_lines(resampled_data)

    for i, (trend_type, trend_data) in enumerate(trend_lines):
        color = 'green' if trend_type == 1 else 'red'
        trend_trace = go.Scatter(
            x=trend_data.index,
            y=trend_data['supertrend'],
            mode='lines',
            name=f'{"Uptrend" if trend_type == 1 else "Downtrend"} Line',
            line=dict(color=color, width=2),
        )

        fig.add_trace(trend_trace)
        st.plotly_chart(fig, theme='streamlit', use_container_width=True)

    # Initialize trend_start and current_trend
    trend_start = None
    current_trend = None

    # Add the sell signals above the candlesticks
    fig.add_trace(go.Scatter(x=sell_signals.index,
                             y=sell_signals['supertrend'],
                             mode='markers',
                             name='Sell Signal',
                             marker=dict(color='green', symbol='triangle-up', size=10)))

    # Add the buy signals below the candlesticks
    fig.add_trace(go.Scatter(x=buy_signals.index,
                             y=buy_signals['supertrend'],
                             mode='markers',
                             name='Buy Signal',
                             marker=dict(color='red', symbol='triangle-down', size=10)))

    # Add live closing price ticker on the right side
    last_close_price = resampled_data['close'].iloc[-1]

    # Adjust the right margin to make room for the live price annotation
    fig.update_layout(margin=dict(r=100))

    # Define the x and y coordinates for the annotation
    annotation_x = resampled_data.index[-1]
    annotation_y = last_close_price

    # Add live price annotation
    fig.add_annotation(
        go.layout.Annotation(
            x=annotation_x,
            y=annotation_y,
            text=f'lp:{last_close_price:.2f}',
            showarrow=False,
            arrowhead=2,
            arrowcolor='red',
            arrowwidth=2,
            arrowsize=1,
            font=dict(color='red', size=20),
            xshift=5,  # Adjust this value to control horizontal position
            yshift=-10,  # Adjust this value to control vertical position
        )
    )

    # Shift y-axis to the right
    fig.update_layout(yaxis=dict(overlaying='y', side='right', showgrid=False))

    # Update y-axis tick text to include live price
    if fig.layout.yaxis.tickvals:
        yaxis_tick_text = [f'{tick:.2f}' for tick in fig.layout.yaxis.tickvals]
        yaxis_tick_text.append(f'{last_close_price:.2f}')
        fig.update_yaxes(ticktext=yaxis_tick_text)

    fig.write_html(graph_file_path)

    return fig, resampled_data.to_dict('records')

# Function to plot candlestick graph with custom colors
def plot_candlestick(data):
    fig = go.Figure(data=[
        go.Candlestick(x=data.index,
                       open=data['open'],
                       high=data['high'],
                       low=data['low'],
                       close=data['close'],
                       increasing_line_color='green',
                       decreasing_line_color='red',
                       line=dict(width=1))
    ])

    # Add traces for 'lp' values as horizontal dotted lines
    for _, row in data.iterrows():
        if 'lp' in row.index:
            lp_trace = go.Scatter(
                x=[row.name, row.name],
                y=[row['lp'], row['lp']],
                mode='lines',
                line=dict(color='blue', width=1, dash='dot'),
                name=f'lp {row["lp"]}'
            )
            fig.add_trace(lp_trace)
    # Customizing the layout of the graph
    fig.update_layout(
        title="Live Candlestick Graph PE",
        title_x=0.5,
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        font=dict(family="Arial, sans-serif", size=14),
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot area
    )

    # Add secondary y-axis for price
    fig.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False))

    return fig
trend_line_visibility = [False] * len(all_trend_lines)

external_stylesheets = [
    {
        "href": (
            "https://fonts.googleapis.com/css2"
            "font-family: 'Qwitcher Grypen', cursive;"
        ),
        "rel": "stylesheet",
    },
]
# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets= external_stylesheets)
server = app.server
# MongoDB setup
client = MongoClient(MONGO_CONNECTION_STRING)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

app.layout = html.Div([
    dcc.Interval(id='graph-update-interval', interval=2000, n_intervals=0),
    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='Tab 1', value='tab-1'),
        dcc.Tab(label='Tab 2', value='tab-2'),
    ]),

    html.Div(id='tabs-content')
])


# Define the callback to display the appropriate page content based on the URL pathname
@app.callback(Output('tabs-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/page-2':
        return generate_page_2_content()
    else:
        return generate_page_1_content()
    
def generate_page_1_content():
    return html.Div([
        html.Div([
            dcc.Graph(id='live-candlestick-graph', config={'displayModeBar': True, 'scrollZoom': True}),
            
            html.Div([
                html.Label('Chart Type:', className='dropdown-label'),
                dcc.Dropdown(
                    id='chart-type-dropdown',
                    options=[
                        {'label': 'Normal', 'value': 'normal'},
                        {'label': 'Heikin Ashi', 'value': 'heikin_ashi'},
                    ],
                    value='normal',
                    clearable=False,
                    className='dropdown'
                ),
            ], className='dropdown-container'),
            
            html.Div([
                html.Label('Interval:', className='dropdown-label'),
                dcc.Dropdown(
                    id='interval-dropdown',
                    options=[
                        {'label': '1 Min', 'value': 1},
                        {'label': '3 Min', 'value': 3},
                        {'label': '5 Min', 'value': 5},
                        {'label': '10 Min', 'value': 10},
                        {'label': '30 Min', 'value': 30},
                        {'label': '60 Min', 'value': 60},
                        {'label': '1 Day', 'value': 1440}
                    ],
                    value=1,
                    clearable=False,
                    className='dropdown'
                ),
            ], className='dropdown-container'),
            dcc.Interval(id='graph-update-interval', interval=2000, n_intervals=0),
            html.Button('Show/Hide Trend Lines', id='toggle-trend-lines-button', n_clicks=0),
        ], className='content-section'),
    ], className='content')

def generate_page_2_content():
    # Fetch data from MongoDB
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    data = collection.find({}, {'_id': 0}).sort('timestamp')
    df = pd.DataFrame(data)
    # Convert and process data as needed
    
    return  html.Div([
            html.H3('Market Depth Table'),
            dash_table.DataTable(
                id='data-table',
                columns=[{'name': col, 'id': col} for col in df.columns],
                data=df.to_dict('records'),
                style_table={'height': '1000px', 'overflowY': 'auto'}
            ),
            dcc.Interval(id='table-update-interval', interval=1000, n_intervals=0)
        ])

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        html.H2(["BullsEdges"], className='header-title'),
        html.Nav([
            dcc.Link('Candlestick Chart', href='/', className='nav-link'),
            dcc.Link('Market Depth Table', href='/page-2', className='nav-link'),
            # Add more navigation links as needed
        ], className='nav'),
    ], className='header'),
    
    html.Div(id='tabs-content'),
    
    html.Div([
        html.P("Your Footer Information", style={'textAlign': 'center'}),
    ], className='footer'),
])

# Layout of the app

visible_trend_lines = []

# Define the callback to update the data for the market depth table on page two
@app.callback(
    Output('live-candlestick-graph', 'figure'),
    [
        Input('interval-dropdown', 'value'),
        Input('chart-type-dropdown', 'value'),
        Input('live-candlestick-graph', 'relayoutData'),
        Input('toggle-trend-lines-button', 'n_clicks'),
        Input('graph-update-interval', 'n_intervals')
    ],
    [
        State('graph-update-interval', 'n_intervals')
    ]
)
def update_graph_callback(interval, chart_type, relayoutData, n_clicks, n, _):
    fig = go.Figure()
    global all_trend_lines, trend_line_visibility
    fig, _ = update_graph(n, interval, chart_type)

    if 'xaxis.range' in relayoutData:
        xaxis_range = relayoutData['xaxis.range']
    else:
        xaxis_range = [df.index[-1] - pd.Timedelta(minutes=60), df.index[-1]]

    filtered_data = df[(df.index >= xaxis_range[0]) & (df.index <= xaxis_range[1])]

    # Toggle visibility of trend lines based on the button click count
    show_trend_lines = n_clicks % 2 == 1

    # Update layout with range selector
    fig.update_layout(
        xaxis=dict(
            range=xaxis_range,
            type='date',
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1D", step="day", stepmode="backward"),
                    dict(count=7, label="1W", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all")
                ]
            )
        )
    )
    return fig


# Define the callback to update the data for the data table on page two
@app.callback(
    Output('data-table', 'data'),
    [Input('interval-dropdown', 'value')],
    [dash.dependencies.State('graph-update-interval', 'n_intervals')]
)
def update_data_table(interval, n):
    global df, data_list

    # Append new data to DataFrame
    if len(data_list) > 0:
        new_df = pd.DataFrame(data_list)
        new_df['lp'] = pd.to_numeric(new_df['lp'], errors='coerce')
        new_df = new_df.dropna(subset=['lp'])
        new_df = new_df[["timestamp", "lp"]]
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], format='%Y-%m-%d %H:%M:%S')
        new_df.set_index("timestamp", inplace=True)
        df = pd.concat([df, new_df])

        df.to_csv(data_file_path)
        data_list = []

    # Fetch data from MongoDB
    data = collection.find({}, {'_id': 0}).sort('timestamp')
    df = pd.DataFrame(data)

    # Convert 'timestamp' column to datetime and set it as the index
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    df.set_index('timestamp', inplace=True)

    # Convert DataFrame to dictionary format for DataTable
    data_table_data = df.to_dict('records')

    return data_table_data
# Run the Dash app
if __name__ == '__main__':
    df = pd.read_csv(data_file_path, index_col="timestamp", parse_dates=True)
    app.run_server(debug=True)
