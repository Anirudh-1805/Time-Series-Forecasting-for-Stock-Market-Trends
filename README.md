# **Stock Price Prediction using LSTM in PyTorch**

This repository contains code to predict stock prices using a Long Short-Term Memory (LSTM) neural network in PyTorch. The model is trained on historical stock price data and predicts future stock closing prices based on past values.

## **Project Overview**

The project uses a dataset of stock prices (CSV format) and applies a time series forecasting method using an LSTM model. The model learns patterns from historical data and generates predictions for stock prices.

### **Key Features**

- **Data Preprocessing**:
  - Extracts the closing price of the stock from the dataset.
  - Transforms the data into sequences with a sliding window of past values (`lookback` period).
  - Scales the data using `MinMaxScaler` to normalize the input features.
  
- **Model Architecture**:
  - A simple LSTM neural network with a fully connected output layer.
  - The model predicts the closing price of a stock given a sequence of past closing prices.

- **Model Training**:
  - Trained using the Mean Squared Error (MSE) loss function and the Adam optimizer.
  - Data split into training and testing sets for evaluation.
  
- **Evaluation**:
  - Plots of actual vs. predicted stock prices.
  - Calculation of the R² score for model performance.

## **Dataset**

The project uses a CSV file named `AMZN.csv` that contains historical stock price data of Amazon. The dataset must have the following columns:

- `Date`: The date of the record.
- `Close`: The closing price of the stock on that date.

## **Dependencies**

The following Python libraries are required to run the code:

- `numpy`
- `pandas`
- `matplotlib`
- `torch`
- `scikit-learn`

To install PyTorch, run:

```bash
!pip install torch
