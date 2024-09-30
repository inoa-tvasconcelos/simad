import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Function to get the opening prices
def get_prices(symbol, start_date, end_date):
    return yf.download(symbol, start=start_date, end=end_date)["Open"]

# Function to calculate daily returns
def get_returns(prices):
    return prices.pct_change().dropna()

# Function to calculate exponential smoothed volatility
def get_eased_vols(returns, beta, valid_period):
    volatilidade_suavizada = [np.std(returns[:valid_period])]  # Initialize with std of the first 'valid_period' returns
    
    for i in range(1, len(returns)):
        vol_atual = np.std(returns[max(0, i-10):i+1])  # Local standard deviation
        vol_suavizada = beta * volatilidade_suavizada[-1] + (1 - beta) * vol_atual  # Exponential smoothing
        volatilidade_suavizada.append(vol_suavizada)
    
    return np.array(volatilidade_suavizada)

# Function to calculate exponential smoothed returns
def get_eased_returns(returns, vols, alfa):
    retorno_medio_suavizado = [returns[0]]  # Initialize with the first return as the starting point
    
    for i in range(1, len(returns)):
        a_i = alfa / (1 + vols[i])  # Dynamic adjustment of 'a' based on volatility
        R_i = a_i * retorno_medio_suavizado[-1] + (1 - a_i) * returns[i]  # Exponential smoothing
        retorno_medio_suavizado.append(R_i)
    
    return np.array(retorno_medio_suavizado)

# Function to generate the DataFrame with smoothed returns and volatility
def get_return_vol_historic_data(prices, alfa, beta, period):
    retornos = get_returns(prices)
    vols = get_eased_vols(retornos, beta, period)
    eased_returns = get_eased_returns(retornos, vols, alfa)
    return pd.DataFrame({
        "vol": vols,
        "ret": eased_returns
    }, index=retornos.index)

# Function to simulate GBM
def simulate_gbm(precos, mu, sigma, epsilon):
    exponent = (mu - 0.5 * sigma**2) + sigma * epsilon
    return precos * np.exp(exponent)

# Function to simulate Monte Carlo from temp_date to end_date
def simulate_monte_carlo(preco_inicial, mu, sigma, start_date, end_date, N_simulations):
    period = len(pd.date_range(start=start_date, end=end_date, freq='B'))  # Ensure business days
    simulations = np.zeros((N_simulations, period))
    simulations[:, 0] = preco_inicial

    for t in range(1, period):
        epsilon = np.random.standard_normal(N_simulations)
        simulations[:, t] = simulate_gbm(simulations[:, t - 1], mu, sigma, epsilon)

    return simulations

if __name__ == "__main__":
    # Parameters
    symbol = 'PETR4.SA'
    start_date = pd.to_datetime('2023-01-01')
    end_date = pd.to_datetime('2023-12-31')
    temp_date = pd.to_datetime('2023-09-01')  # Temp date where simulation starts
    beta = 0.8
    alfa = 1
    periodo_volatilidade = 7
    concurrent_simulations = 10000
    
    # Fetch opening prices up to end date
    prices = get_prices(symbol, start_date, end_date)
    
    # Ensure temp_date exists in the prices index
    if temp_date not in prices.index:
        temp_date = prices.index[prices.index.get_loc(temp_date, method='ffill')]
    
    # Fetch prices up to the temp_date for historical data calculations
    prices_temp = prices.loc[:temp_date]
    
    # Calculate smoothed return and volatility based on historical data until temp_date
    result = get_return_vol_historic_data(prices_temp, alfa, beta, periodo_volatilidade)
    
    # Get the last smoothed return (mu) and volatility (sigma) at temp_date
    last_mu = result["ret"].iloc[-1]
    last_sigma = result["vol"].iloc[-1]
    last_price = prices_temp.iloc[-1]
    
    # Simulate Monte Carlo from temp_date to end_date
    simulations = simulate_monte_carlo(last_price, last_mu, last_sigma, temp_date, end_date, concurrent_simulations)
    
    # Create a date range for the simulation period
    simulated_dates = pd.date_range(start=temp_date, end=end_date, freq='B')
    mean_simulated_prices = np.mean(simulations, axis=0)

    assert simulations.shape[1] == len(simulated_dates), "Mismatch between simulation length and date range."
    plt.figure(figsize=(12, 6))
    
    plt.plot(simulated_dates, simulations.T, color='blue', alpha=0.1)
    plt.plot(prices.index, prices.values, label='Preços Reais', color='black', linewidth=2)
    plt.plot(simulated_dates, mean_simulated_prices, label='Preço Médio Simulado', color='green', linewidth=2)
    # Title and labels
    plt.title('Comparação entre Preços Reais e Preço Médio Simulado de Monte Carlo')
    plt.xlabel('Data')
    plt.ylabel(f'Preço de {symbol} (R$)')
    plt.legend()
    
    # Show the plot
    plt.show()
