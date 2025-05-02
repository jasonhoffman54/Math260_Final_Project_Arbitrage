# Math260_Final_Project_Arbitrage

## Currency Arbitrage Project

This repository contains two Python scripts that together allow you to:

Fetch up-to-date exchange rate data for a set of major currencies and save it as a CSV matrix.

Detect and visualize the most profitable currency arbitrage cycle using the Bellman–Ford algorithm on the logarithmically transformed exchange rates.

## Contents

- `Get_Exchange_Rates.py`  
  Fetches current exchange rates from ExchangeRate-API for a specified list of currencies and writes a *.csv matrix.

- `Arbitrage_BFord.py`  
  Reads the CSV matrix, transforms rates with `-log`, builds a directed graph, enumerates negative cycles across all currencies, selects the most profitable arbitrage cycle, and visualizes it.

- `full_exchange_rates_matrix_top20.csv` (generated)  
  The exchange-rate matrix for the top 20 currencies.



### 1. Fetching Exchange Rates

The script `Get_Exchange_Rates.py`:

Defines a list of top 20 (or top 50) currency codes.

Calls **ExchangeRate-API** endpoints with your API key to fetch all conversion rates for each base currency.

Populates a Pandas DataFrame where rows and columns are the same currency list.

Saves the complete matrix as `full_exchange_rates_matrix_top20.csv`.

#### Usage

Edit the list at the top if you wish to change which currencies to include, then run:

**python Get_Exchange_Rates.py**

After completion, you will see:

A printed list of supported currencies.

A file `full_exchange_rates_matrix_top20.csv` in the working directory.

### 2. Arbitrage Detection

The script `Arbitrage_BFord.py`:

Reads the CSV matrix via parseRates().

Transforms exchange rates to edge weights .

Builds an edge list for every pair of currencies.

Uses an extended Bellman–Ford (bellman_ford_all) from each currency as the source to find all negative-cycle vertices.

Reconstructs each cycle, normalizes it (rotates to a canonical form), and computes its profit ratio .

Caches cycles to avoid duplicates and selects the cycle with the highest profit ratio.

Visualizes the graph with NetworkX:

Highlights the best arbitrage cycle in red.

Displays the profit multiplier in the title.

#### Usage

Ensure you have generated `full_exchange_rates_matrix_top20.csv`. Then run:

**python Arbitrage_BFord.py**

On success, you'll see:

The best arbitrage cycle printed (e.g., ['NZD', 'SEK', 'NZD']).

The profit ratio (e.g., 1.0012).

A window displaying the graph, with the cycle edges highlighted.
