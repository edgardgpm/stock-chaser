# stock-chaser
#### Video Demo:  <URL HERE - PENDING>

#### Description: The StockChaser is a Python project that employs a machine learning pipeline to predict whether a stock's closing price will be higher in N amount of days, using historical market data and technical indicators. The goal of the project is to build a clean, reproducible pipeline for financial prediction while following good ML and software engineering practices.

## Features
- Moving averages
- Multi-period returns
- Rolling volatility
- Automatic historical data download via Yahoo Finance
- Time-series safe train/test split
- Configurable probability threshold
- Model evaluation & artifact generation

## Model
Random Forest Classifier (scikit-learn)

## How it works
Data Load → Feature Engineering → Dataset Preparation → Time Split → Training → Evaluation → Reporting

## How to run
python main.py --symbol AAPL --threshold 0.6

## Optional Arguments 
--start       Start date for historical data  
--end         End date  
--threshold   Probability cutoff for UP prediction

## Technologies Used
- Python
- yfinance
- pandas
- numpy
- scikit-learn
- matplotlib

## Outputs
/results folder:
predictions (.csv)
metrics (.txt)
comparison chart (.png)
probability chart (.png)

## Project Status
🚧 In development — feature engineering and model training in progress.

## Author
Edgard González