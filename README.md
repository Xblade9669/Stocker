# Stocker
Stock Market Predictor Using LSTM
This project is a stock market price predictor that utilizes historical stock price data, processes it using machine learning techniques, and employs a Long Short-Term Memory (LSTM) neural network to predict future stock prices.

Project Overview
The program fetches real-time stock price data using the Yahoo Finance (yfinance) API, processes it, and then trains an LSTM deep learning model to make predictions. It is designed to predict the next closing price of a given stock based on past trends.

How It Works
The project follows these key steps:

1. Fetching Stock Data (Real-Time Data Retrieval)
We use yfinance, a Python wrapper for Yahoo Finance, to fetch real-time stock prices.
The function get_stock_data(ticker, period='1y', interval='1d') fetches the stock's historical price data.
The user inputs a stock ticker (e.g., AAPL for Apple, AMZN for Amazon, TSLA for Tesla), and the program retrieves the last 1 year of daily price data.

2. Data Preprocessing (Scaling and Feature Engineering)
Stock prices have large variations, so we normalize the data using MinMaxScaler from sklearn.preprocessing.
We only consider the closing price for prediction.
The function preprocess_data(df) transforms the closing price data into a scaled range (0 to 1), which helps in stabilizing the model training process.

3. Creating Input Data for Training the LSTM Model
Since stock prices follow a time series, we need to structure our dataset properly.
The function create_dataset(data, time_step=60) splits the dataset into training sequences:
It uses a 60-day lookback period, meaning it takes the last 60 days' prices as input (X) to predict the next day's closing price (Y).
The dataset is then reshaped into a 3D array, which is required for LSTM models (samples, time steps, features).

4. Building the LSTM Neural Network Model
We use a stacked LSTM model with the following architecture:
LSTM Layer 1: 50 neurons, returns sequences (return_sequences=True).
Dropout Layer: 20% dropout to prevent overfitting.
LSTM Layer 2: 50 neurons, does not return sequences.
Dropout Layer: Another 20% dropout.
Dense Layer: 25 neurons.
Output Layer: 1 neuron (predicts the next day's price).
The model uses the Adam optimizer and Mean Squared Error (MSE) as the loss function.

5. Training the Model and Predicting Future Stock Prices
We first train the LSTM model on past stock prices.
Once trained, we take the last 60 days of real-time stock price data, feed it into the model, and predict the next closing price.
The predicted price is then inverse-transformed (converted back to the original scale).

Limitations & Future Improvements
🔴 Limitations:

The model only uses closing prices; incorporating technical indicators (e.g., Moving Averages, RSI) could improve accuracy.
It predicts only the next day's price; extending it to predict multiple future days would require modifications.
Market factors like news events, earnings reports, and macroeconomic changes are not considered.
✅ Future Improvements:

Use additional technical indicators (Moving Averages, MACD, Bollinger Bands).
Extend the model to predict next week/month's stock prices.
Improve accuracy with hyperparameter tuning and more layers in the LSTM model.
Implement Sentiment Analysis on financial news to factor in market trends.
Conclusion
This stock market predictor provides a basic LSTM model for forecasting future stock prices using historical data. While it offers a simple approach, real-world stock predictions require more complex models, additional features, and a deep understanding of market trends. 🚀
