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
```
## **Instructions**

1. **Load the dataset:** Make sure you have the `AMZN.csv` file in your working directory.
2. **Run the code:** The script reads the dataset, preprocesses the data, builds and trains the LSTM model, and outputs predictions along with evaluation metrics and plots.
3. **Training the model:** The model is trained for a default of 10 epochs using a batch size of 16. You can adjust these hyperparameters in the code.
4. **Evaluate the model:** The model generates plots of actual vs. predicted stock prices for both training and testing data, and computes the R² score.

## **Model Structure**

- **Input**: Sequences of stock closing prices with a lookback period of 7 days.
- **Hidden Layers**: LSTM layers with a hidden size of 4 and one stacked layer.
- **Output**: One fully connected layer predicting the closing price for the next day.

## **Example Results**

The following plots are generated:

- **Training Data**: A comparison of the actual vs. predicted closing prices for the training set.
- **Testing Data**: A comparison of the actual vs. predicted closing prices for the testing set.

You can also find the R² score, which measures how well the model fits the test data.

## **Usage**

- Clone the repository and ensure all dependencies are installed.
- Upload your dataset (`AMZN.csv`) and run the script.


