#!/usr/bin/env python3

"""Read CSV file with stock data for ONE stock."""

# Standard library

# 3rd party packages
import pandas as pd

# Local source


def read_time_series(input_file: str) -> pd.DataFrame:
    """Assumes first column contains time series for stock data."""
    dataframe = pd.read_csv(input_file, header=0, parse_dates=[0])
    dataframe.columns = dataframe.columns.str.lower()
    return dataframe


def write_time_series(dataframe: pd.DataFrame, output_file: str) -> pd.DataFrame:
    """Output stock price as a time series function.
    USES OHLC (open, high, low, close) AVERAGE AS THE STOCK PRICE.
    """
    # OHLC_data = dataframe[["open", "high", "low", "close"]].mean(axis=1)
    # new_dataframe = pd.concat([dataframe["date"], OHLC_data], axis=1)
    new_dataframe = pd.concat([dataframe["date"], dataframe["adj close"]])
    new_dataframe.columns = ["date", "price"]
    new_dataframe.to_csv(output_file)
