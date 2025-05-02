import requests
import pandas as pd
import time
import os
from openpyxl import load_workbook


API_KEY = "6934cd92fc76e2dbd01a9085"
BASE_URL = "https://v6.exchangerate-api.com/v6/{api_key}/latest/{base}"
REQUEST_SLEEP = 1 

top_20_currencies = [
    "USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "CNY",
    "HKD", "NZD", "SEK", "KRW", "SGD", "INR", "RUB", "MXN",
    "BRL", "ZAR", "TRY", "DKK"
]

top_50_currencies = [
    "USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "CNY", "HKD", "NZD",
    "SEK", "KRW", "SGD", "NOK", "MXN", "INR", "RUB", "ZAR", "TRY", "BRL",
    "TWD", "DKK", "PLN", "THB", "IDR", "HUF", "CZK", "ILS", "CLP", "PHP",
    "AED", "COP", "SAR", "MYR", "RON", "VND", "EGP", "PKR", "BDT", "LKR",
    "KWD", "QAR", "OMR", "BHD", "JOD", "DZD", "MAD", "LBP", "NGN", "CRC"
]

def fetch_rates_for_base(base_currency):
    """
    Fetches the conversion rates for the given base currency.
    Returns:
        A dictionary mapping quote currencies to rates if successful,
        or None if there is an error.
    """
    url = BASE_URL.format(api_key=API_KEY, base=base_currency)
    response = requests.get(url)
    data = response.json()
    
    if data.get("result") != "success":
        print(f"Error fetching rates for {base_currency}: {data.get('error-type')}")
        return None
    return data.get("conversion_rates")

def main():
    # Use USD as the initial base to determine supported currencies.
    base_for_list = "USD"
    initial_rates = fetch_rates_for_base(base_for_list)
    if initial_rates is None:
        raise Exception("Failed to fetch rates for USD to determine supported currencies.")
    
    # Filter the supported currencies to only those in the top 20 list.
    supported_currencies = sorted(set(initial_rates.keys()).intersection(set(top_20_currencies)))
    print("Supported major currencies (top 20):", supported_currencies)
    
    # Create a DataFrame to hold the full exchange rate matrix.
    full_matrix = pd.DataFrame(index=supported_currencies, columns=supported_currencies, dtype=float)
    
    # For each currency in our filtered list, fetch its conversion rates.
    for base in supported_currencies:
        print(f"Fetching rates for base currency: {base}")
        rates = fetch_rates_for_base(base)
        if rates is not None:
            for quote in supported_currencies:
                full_matrix.loc[base, quote] = rates.get(quote)
        else:
            print(f"Skipping {base} due to an error.")
        time.sleep(REQUEST_SLEEP)  # Pause to respect API rate limits.
    
    # Save the full matrix to a CSV file.
    csv_filename = "full_exchange_rates_matrix_top20.csv"
    full_matrix.to_csv(csv_filename)
    print(f"Full exchange rate matrix for top 20 currencies saved to {csv_filename}")

if __name__ == "__main__":
    main()