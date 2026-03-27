# Jane Street Real-Time Market Data Forecasting

This repository contains the **bronze-winning solution (ranked 321/3757)** for the [Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting) competition on Kaggle.

The project includes:

- Data analysis
- Feature engineering and categorical encoding
- Convolutional GRU-based time series modeling (offline training)
- Online incremental learning and inference

---

## 📁 Project Structure

```text
├── data/                          # All training & test datasets
│   ├── train.parquet              # Initial full training data
│   ├── test.parquet               # Official simulated test data
│   ├── lags.parquet               # Simulated lag features provided by organizers
│   ├── train_data.parquet         # Filtered training data (date_id > 1100)
│   ├── valid_data.parquet         # Filtered validation data (date_id > 1640)
│   ├── synthetic_test.parquet     # Simulated test data synthesized from train.parquet (for online learning)
│   └── synthetic_lags.parquet     # Simulated lag features (for online learning)
├── kaggle_evaluation/             # Scripts to simulate online prediction via official interface
├── model/
│   └── GRU.ckpt                   # Checkpoint of GRU model after offline training
├── notebooks/
│   ├── EDA.ipynb                  # Exploratory data analysis
│   └── feature-engineering.ipynb  # Attempts at feature engineering
├── scripts/
│   └── run_all.sh                 # One-click script for data generation, training, and inference
├── src/
│   ├── synthetic_data.py          # Generate synthetic data for training (49 dates' test and lags data)
│   ├── train.py                   # Training entry point, includes validation logic
│   ├── inference.py               # Local inference entry point, starts inference server
│   ├── model_gru.py               # GRU model architecture and parameters
│   ├── online_predictor.py        # JsGruOnlinePredictor class for online learning + inference
│   ├── utils.py                   # Utility functions: encoding, R², etc.
│   └── __init__.py                # Marks src as a Python module
├── requirements.txt               # Python dependencies
├── setup.py                       # Project installation script
├── submission.parquet             # Submission file generated after running inference
└── README.md                      # This file
