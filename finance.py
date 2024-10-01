import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def get_prices(symbol, start_date, end_date):
    return yf.download(symbol, start=start_date, end=end_date)["Open"]

def get_returns(prices):
    return prices.pct_change().dropna()

def get_eased_vols(returns, beta, valid_period):
    volatilidade_suavizada = [np.std(returns[:valid_period])]  
    
    for i in range(1, len(returns)):
        vol_atual = np.std(returns[max(0, i-10):i+1])
        vol_suavizada = beta * volatilidade_suavizada[-1] + (1 - beta) * vol_atual 
        volatilidade_suavizada.append(vol_suavizada)
    
    return np.array(volatilidade_suavizada)

def get_eased_returns(returns, vols, alfa):
    retorno_medio_suavizado = [returns[0]] 
    
    for i in range(1, len(returns)):
        a_i = alfa / (1 + vols[i])
        R_i = a_i * retorno_medio_suavizado[-1] + (1 - a_i) * returns[i] 
        retorno_medio_suavizado.append(R_i)
    
    return np.array(retorno_medio_suavizado)

def get_return_vol_historic_data(prices, alfa, beta, period):
    retornos = get_returns(prices)
    vols = get_eased_vols(retornos, beta, period)
    eased_returns = get_eased_returns(retornos, vols, alfa)
    return pd.DataFrame({
        "vol": vols,
        "ret": eased_returns
    }, index=retornos.index)

def simulate_gbm(precos, mu, sigma, epsilon):
    exponent = (mu - 0.5 * sigma**2) + sigma * epsilon
    return precos * np.exp(exponent)

def simulate_monte_carlo(preco_inicial, mu, sigma, start_date, end_date, N_simulations):
    period = len(pd.date_range(start=start_date, end=end_date, freq='B'))
    simulations = np.zeros((N_simulations, period))
    simulations[:, 0] = preco_inicial

    for t in range(1, period):
        epsilon = np.random.standard_normal(N_simulations)
        simulations[:, t] = simulate_gbm(simulations[:, t - 1], mu, sigma, epsilon)

    return simulations

if __name__ == "__main__":
    symbol = 'PETR4.SA'
    start_date = pd.to_datetime('2023-01-01')
    end_date = pd.to_datetime('2023-12-31')
    temp_date = pd.to_datetime('2023-09-01')
    beta = 0.8
    alfa = 1
    periodo_volatilidade = 64
    concurrent_simulations = 10000
    prices = get_prices(symbol, start_date, end_date)
    
    prices_temp = prices.loc[:temp_date]
    
    result = get_return_vol_historic_data(prices_temp, alfa, beta, periodo_volatilidade)
    
    last_mu = result["ret"].iloc[-1]
    last_sigma = result["vol"].iloc[-1]
    last_price = prices_temp.iloc[-1]
    
    simulations = simulate_monte_carlo(last_price, last_mu, last_sigma, temp_date, end_date, concurrent_simulations)
    
    simulated_dates = pd.date_range(start=temp_date, end=end_date, freq='B')
    
    mean_simulated_prices = np.mean(simulations, axis=0)

    if True:
      plt.figure(figsize=(12, 6))
      plt.plot(simulated_dates, simulations.T, color='blue', alpha=0.1)
      plt.plot(prices.index, prices.values, label='Preços Reais', color='black', linewidth=2)
      plt.plot(simulated_dates, mean_simulated_prices, label='Preço Médio Simulado', color='green', linewidth=2)
      plt.title('Comparação entre Preços Reais e Preço Médio Simulado de Monte Carlo')
      plt.xlabel('Data')
      plt.ylabel(f'Preço de {symbol} (R$)')
      plt.legend()
      
    
    if True:
      plt.figure(figsize=(12, 6))
      
      plt.plot(simulated_dates, simulations.T, color='blue', alpha=0.1)
      plt.title('Diferentes simulações de preço')
      plt.xlabel('Data')
      plt.ylabel(f'Preço de {symbol} (R$)')
      plt.legend()
    
    final_prices = simulations[:, -1]
    if True:
      plt.figure(figsize=(10, 6))
      plt.hist(final_prices, bins=50, color='blue')
      plt.title('Distribuição dos Preços Finais após Simulações')
      plt.xlabel('Preço Final')
      plt.ylabel('Frequência')
    
    mean_final_price = np.mean(final_prices)
    std_final_price = np.std(final_prices)

    confidence_interval_95 = np.percentile(final_prices, [2.5, 97.5])

    print(f"Média dos Preços Finais: {mean_final_price:.2f}")
    print(f"Desvio Padrão dos Preços Finais: {std_final_price:.2f}")
    print(f"Intervalo de Confiança de 95% dos Preços Finais: {confidence_interval_95[0]:.2f} - {confidence_interval_95[1]:.2f}")
    
    sorted_final_prices = np.sort(final_prices)
    cdf_empirical = np.arange(1, len(sorted_final_prices) + 1) / len(sorted_final_prices)

    mu = np.log(mean_final_price) - 0.5 * np.log(1 + (std_final_price / mean_final_price)**2)
    sigma = np.sqrt(np.log(1 + (std_final_price / mean_final_price)**2))
    cdf_theoretical = norm.cdf(np.log(sorted_final_prices), loc=mu, scale=sigma)

    # Plotar as CDFs empírica e teórica
    if True:
      plt.figure(figsize=(10, 6))
      plt.plot(sorted_final_prices, cdf_empirical, label="CDF Empírica", color="blue")
      plt.plot(sorted_final_prices, cdf_theoretical, label="CDF Teórica (Log-normal)", color="red", linestyle='--')
      plt.title('Comparação entre a CDF Empírica e a CDF Teórica')
      plt.xlabel('Preço Final')
      plt.ylabel('CDF')
      plt.legend()
    
    valid_dates = simulated_dates.intersection(prices.index)
    valid_indices = simulated_dates.get_indexer(valid_dates)
    valid_simulations = simulations[:, valid_indices]
    real_prices = prices.loc[valid_dates]
    mean_simulated_prices = np.mean(valid_simulations, axis=0)
    predicted_prices = mean_simulated_prices[:len(valid_dates)]  
    
    mae = mean_absolute_error(real_prices, predicted_prices)
    rmse = np.sqrt(mean_squared_error(real_prices, predicted_prices))

    r2 = r2_score(real_prices, predicted_prices)

    print(f"Erro Médio Absoluto (MAE): {mae:.2f}")
    print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f}")
    print(f"Coeficiente de Determinação (R²): {r2:.2f}")
    
    real_final_price = prices.loc[simulated_dates[-1]] if simulated_dates[-1] in prices.index else prices.iloc[-1]

    print(f"Preço Real Final em {simulated_dates[-1].date()}: {real_final_price:.2f}")
    residuals = real_prices - predicted_prices
    std_residuals = np.std(residuals)

    residual_confidence_upper = predicted_prices + 1.96 * std_residuals
    residual_confidence_lower = predicted_prices - 1.96 * std_residuals
    if True:
      plt.figure(figsize=(12, 6))
      plt.plot(prices.index, prices.values, label='Real Prices', color='black', linewidth=2)
      plt.plot(valid_dates, predicted_prices, label='Predicted Prices', color='green', linewidth=2)
      plt.fill_between(valid_dates, residual_confidence_lower, residual_confidence_upper, color='green', alpha=0.3, label='95% Confidence Interval')
      plt.title(f'Previsão de preço com 95% confiança')
      plt.xlabel('Date')
      plt.ylabel(f'Price of {symbol} (R$)')
      plt.legend()
    
    within_confidence_interval = np.mean((real_prices >= residual_confidence_lower) & (real_prices <= residual_confidence_upper)) * 100
    print(f"{within_confidence_interval:.2f}% of real prices are within the confidence interval")
    print(f"interval of confidence {(2*1.96 * std_residuals)}")
    
    sim_confidence_lower = np.percentile(valid_simulations, 2.5, axis=0)
    sim_confidence_upper = np.percentile(valid_simulations, 97.5, axis=0)
    if True:
      plt.figure(figsize=(12, 6))
      plt.plot(prices.index, prices.values, label='Real Prices', color='black', linewidth=2)
      plt.plot(valid_dates, mean_simulated_prices, label='Predicted Mean Prices', color='green', linewidth=2)
      plt.fill_between(valid_dates, sim_confidence_lower, sim_confidence_upper, color='green', alpha=0.3, label='95% Confidence Interval')
      plt.title(f'Previsão de preço com dinamica 95% CONFIANCA')
      plt.xlabel('Date')
      plt.ylabel(f'Price of {symbol} (R$)')
      plt.legend()
    
    plt.show()