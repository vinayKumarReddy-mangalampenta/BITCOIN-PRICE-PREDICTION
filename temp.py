import yfinance as yf
import pandas as pd

# Download Bitcoin historical data
btc = yf.download("BTC-USD", period="8y")

# Reset index to include the Date column
btc.reset_index(inplace=True)

# Add "Adj Close" column (same as "Close" for cryptocurrencies)
btc["Adj Close"] = btc["Close"]

# Reorder columns to match the required format
btc = btc[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]

# Save data to a CSV file
btc.to_csv("BTC-USD.csv", index=False)

print("Bitcoin price data saved to bit.csv successfully!")
