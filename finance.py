import random
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

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def play(symbols, start_date, end_date, init_date):
  prices = {}
  for symbol in symbols:
    try:
      price = get_prices(symbol, start_date, end_date)
      if not price.empty:
        price.loc[init_date] # try to see if it has price in that date
        prices[symbol] = price
      else:
        print(f"Não foi possível obter preços do {symbol}")
    except:
      print(f"{symbol} não tem preço para o dia {str(init_date)}")
  initial_qty = 500
  symbols = list(prices.keys())
  simulated_dates = pd.date_range(start=init_date, end=end_date, freq='B')
  valid_dates = simulated_dates.intersection(prices[symbols[0]].index) 
  carteira = {symbol: initial_qty for symbol in symbols}  
  carteira_pato = {symbol: initial_qty for symbol in symbols}  
  carteira_per_month = {}
  carteira_pato_per_month = {}
  total_value = sum([val * prices[symb].loc[init_date] for (symb, val) in carteira.items()])
  total_value_pato = sum([val * prices[symb].loc[init_date] for (symb, val) in carteira_pato.items()])

  total_per_month = [{'date': init_date, 'total_value': total_value}]
  total_per_month_pato = [{'date': init_date, 'total_value': total_value_pato}]
  stock_qty_per_month = {symbol: [{ "value": carteira[symbol], "date": init_date }] for symbol in symbols}  
  stock_qty_per_month_pato = {symbol: [{ "value": carteira_pato[symbol], "date": init_date }] for symbol in symbols}  
  total_qty_per_month = [sum(carteira.values())] 
  total_qty_per_month_pato = [sum(carteira_pato.values())] 
  
  sell_limit = 20000
  buy_limit = 10000
  
  valid_dates_per_month = valid_dates.to_series().groupby(valid_dates.to_period("M")).first()
  if init_date not in valid_dates_per_month.values:
    valid_dates_per_month = valid_dates_per_month.append(pd.Series(init_date)).sort_values()
  valid_dates_per_month = valid_dates_per_month[valid_dates_per_month >= init_date]
  
  dates_size = len(valid_dates_per_month)
  printProgressBar(0, dates_size, prefix = 'Progress:', suffix = 'Complete', length = 50)
  for i, current_date in enumerate(valid_dates_per_month):
    if current_date <= init_date:
      continue
    printProgressBar(i + 1, dates_size, prefix = 'Progress:', suffix = 'Complete', length = 50)
    retornos = []
    retornos_pato = []
    for symbol in symbols:
      ret = expected_return(current_date, prices[symbol], end_date)
      retornos.append({'symbol': symbol, 'ret': ret})
      ret_pato = random.choice([0, 1/len(symbols)])
      retornos_pato.append({'symbol': symbol, 'ret': ret_pato})
    carteira_desejada_alloc = carteira_desejada(retornos)
    carteira_desejada_pato = carteira_desejada(retornos_pato)
    price_by_symbol = {symbol: prices[symbol].loc[current_date] for symbol in symbols}
    carteira_diff = decisoes_do_dia(carteira, carteira_desejada_alloc, price_by_symbol, sell_limit, buy_limit)
    carteira_diff_pato = decisoes_do_dia(carteira_pato, carteira_desejada_pato, price_by_symbol, sell_limit, buy_limit)
    for diff in carteira_diff.values():
      symbol = diff['symbol']
      carteira[symbol] += diff['ret']
    for diff in carteira_diff_pato.values():
      symbol = diff['symbol']
      carteira_pato[symbol] += diff['ret']
    for symbol in symbols:
      stock_qty_per_month[symbol].append({ "value": carteira[symbol], "date": current_date })
      stock_qty_per_month_pato[symbol].append({ "value": carteira_pato[symbol], "date": current_date })
    
    total_qty = sum(carteira[symbol] for symbol in symbols)
    total_qty_pato = sum(carteira_pato[symbol] for symbol in symbols)
    total_qty_per_month.append(total_qty)
    total_qty_per_month_pato.append(total_qty_pato)
    
    carteira_per_month[current_date.month] = {symbol: carteira[symbol] for symbol in symbols}
    carteira_pato_per_month[current_date.month] = {symbol: carteira_pato[symbol] for symbol in symbols}
    total_value = sum([carteira[symbol] * price_by_symbol[symbol] for symbol in symbols])
    total_value_pato = sum([carteira_pato[symbol] * price_by_symbol[symbol] for symbol in symbols])
    total_per_month.append({'date': current_date, 'total_value': total_value})
    total_per_month_pato.append({'date': current_date, 'total_value': total_value_pato})
  
  plot_stock_quantities(stock_qty_per_month, stock_qty_per_month_pato)
  plot_total_quantity(total_qty_per_month, total_qty_per_month_pato, valid_dates_per_month)
  plot_total_carteira_value(total_per_month, total_per_month_pato)
  return carteira_per_month, total_per_month, stock_qty_per_month, total_qty_per_month

def plot_stock_quantities(stock_qty_per_month, stock_qty_per_month_pato):
    plt.figure(figsize=(10, 6))
    
    for symbol, values in stock_qty_per_month.items():
        dates = [entry['date'] for entry in values]
        values = [entry['value'] for entry in values]
        plt.plot(dates, values, label=f'Humano {symbol} Qty', color="blue")
    for symbol, values in stock_qty_per_month_pato.items():
        dates = [entry['date'] for entry in values]
        values = [entry['value'] for entry in values]
        plt.plot(dates, values, label=f'Humano {symbol} Qty', color="red")
    
    plt.title('Quantidade total por simbolo')
    plt.xlabel('Data')
    plt.ylabel('Qty')
    plt.legend()
    
    

def plot_total_quantity(total_qty_per_month, total_qty_per_month_pato, valid_dates_per_month):
    plt.figure(figsize=(10, 6))
    
    plt.plot(valid_dates_per_month, total_qty_per_month, label='Humano Total Qty', color='blue')
    plt.plot(valid_dates_per_month, total_qty_per_month_pato, label='Peixe Total Qty', color='red')
    
    plt.title('Total Quantity Across All Symbols Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Stock Quantity')
    plt.legend()

def plot_total_carteira_value(total_per_month, total_per_month_pato):
    plt.figure(figsize=(10, 6))
    
    # Extract dates and total values from total_per_month
    dates = [entry['date'] for entry in total_per_month]
    dates_pato = [entry['date'] for entry in total_per_month_pato]
    total_values = [entry['total_value'] for entry in total_per_month]
    total_values_pato = [entry['total_value'] for entry in total_per_month_pato]
    
    # Plot total carteira value over time
    plt.plot(dates, total_values, label='Humano Total Carteira Valor', color='green', linewidth=2)
    plt.plot(dates_pato, total_values_pato, label='Peixe Total Carteira Valor', color='red', linewidth=2)
    
    plt.title('Total Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Portfolio Value (R$)')
    plt.legend()

def carteira_desejada(retornos):
    retornos_positivos = [data for data in retornos if data['ret'] > 0]
    total_retorno = sum([data['ret'] for data in retornos_positivos])
    if total_retorno == 0:
        return {data['symbol']: 0 for data in retornos}
    carteira_resultante = {
        data['symbol']: data['ret'] / total_retorno if data['ret'] > 0 else 0 for data in retornos
    }
    
    return carteira_resultante

def decisoes_do_dia(carteira_atual, carteira_desejada, price_by_symbol, sell_limit, buy_limit):
    carteira_diff = {}
    total_sell = 0
    total_shares = sum([v for v in carteira_atual.values()])
    for simbolo, atual_value_in_shares in carteira_atual.items():
        desired_value_in_shares = carteira_desejada[simbolo]*total_shares
        price = price_by_symbol[simbolo]
        diff_in_shares = desired_value_in_shares - atual_value_in_shares
        if diff_in_shares < 0:
            total_sell += abs(diff_in_shares) * price
    
    sell_multiplier = 1
    if total_sell > sell_limit:
        sell_multiplier = sell_limit / total_sell
        total_sell = sell_limit
    buy_limit += total_sell
    
    for simbolo, atual_value_in_shares in carteira_atual.items():
        desired_value_in_shares = carteira_desejada[simbolo]*total_shares
        price = price_by_symbol[simbolo]
        diff_in_shares = desired_value_in_shares - atual_value_in_shares
        if diff_in_shares < 0:
            carteira_diff[simbolo] = {
              'ret': round(diff_in_shares * sell_multiplier),
              'symbol': simbolo
            }
    
    total_buy = 0
    for simbolo, atual_value_in_shares in carteira_atual.items():
        desired_value_in_shares = carteira_desejada[simbolo] * (total_buy + buy_limit)
        price = price_by_symbol[simbolo]
        diff_in_shares = desired_value_in_shares - atual_value_in_shares
        if diff_in_shares > 0:
            total_buy += diff_in_shares * price
    
    buy_multiplier = 1
    if total_buy > buy_limit:
        buy_multiplier = buy_limit / total_buy
        total_buy = buy_limit    
    for simbolo, atual_value_in_shares in carteira_atual.items():
        desired_value_in_shares = carteira_desejada[simbolo] * (total_buy + buy_limit)
        diff_in_shares = desired_value_in_shares - atual_value_in_shares
        if diff_in_shares > 0:
          carteira_diff[simbolo] = {
              'ret': round(diff_in_shares * buy_multiplier),
              'symbol': simbolo
          }
    
    return carteira_diff
  
def expected_return(temp_date, prices, end_date):
  beta = 0.8
  alfa = 0.95
  periodo_volatilidade = 16
  concurrent_simulations = 10000
  prices_temp = prices.loc[:temp_date]
  if len(prices_temp) < periodo_volatilidade:
      return 0  
  result = get_return_vol_historic_data(prices_temp, alfa, beta, periodo_volatilidade)
  last_mu = result["ret"].iloc[-1]
  last_sigma = result["vol"].iloc[-1]
  last_price = prices_temp.iloc[-1]
  if temp_date >= end_date:
      return 0  
  simulations = simulate_monte_carlo(last_price, last_mu, last_sigma, temp_date, end_date, concurrent_simulations)
  final_prices = simulations[:, -1]
  mean_final_price = np.mean(final_prices)
  return (mean_final_price - last_price) / last_price
  
def show_example():
    symbol = 'IRBR3.SA'
    start_date = pd.to_datetime('2023-01-01')
    end_date = pd.to_datetime('2023-12-31')
    temp_date = pd.to_datetime('2023-09-21')
    beta = 0.8
    alfa = 0.95
    periodo_volatilidade = 32
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

    render_images = False
    if render_images:
      plt.figure(figsize=(12, 6))
      plt.plot(simulated_dates, simulations.T, color='blue', alpha=0.1)
      plt.plot(prices.index, prices.values, label='Preços Reais', color='black', linewidth=2)
      plt.plot(simulated_dates, mean_simulated_prices, label='Preço Médio Simulado', color='green', linewidth=2)
      plt.title('Comparação entre Preços Reais e Preço Médio Simulado de Monte Carlo')
      plt.xlabel('Data')
      plt.ylabel(f'Preço de {symbol} (R$)')
      plt.legend()
      
    
    if render_images:
      plt.figure(figsize=(12, 6))
      
      plt.plot(simulated_dates, simulations.T, color='blue', alpha=0.1)
      plt.title('Diferentes simulações de preço')
      plt.xlabel('Data')
      plt.ylabel(f'Preço de {symbol} (R$)')
      plt.legend()
    
    final_prices = simulations[:, -1]
    if render_images:
      from scipy.stats import norm
      plt.figure(figsize=(10, 6))
      plt.hist(final_prices, bins=50, density=True, color='blue', alpha=0.6, label='Simulated Prices')

      # Fit and plot the normal distribution
      mean, std = np.mean(final_prices), np.std(final_prices)
      xmin, xmax = plt.xlim()
      x = np.linspace(xmin, xmax, 100)
      p = norm.pdf(x, mean, std)
      plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')

      # Add titles and labels
      plt.title('Distribuição dos Preços Finais após Simulações com Curva Normal')
      plt.xlabel('Preço Final')
      plt.ylabel('Densidade de Probabilidade')
      plt.legend()
    
    mean_final_price = np.mean(final_prices)
    std_final_price = np.std(final_prices)

    print(f"Média dos Preços Finais: {mean_final_price:.2f}")
    print(f"Desvio Padrão dos Preços Finais: {std_final_price:.2f}")
    
    sorted_final_prices = np.sort(final_prices)
    cdf_empirical = np.arange(1, len(sorted_final_prices) + 1) / len(sorted_final_prices)

    mu = np.log(mean_final_price) - 0.5 * np.log(1 + (std_final_price / mean_final_price)**2)
    sigma = np.sqrt(np.log(1 + (std_final_price / mean_final_price)**2))
    cdf_theoretical = norm.cdf(np.log(sorted_final_prices), loc=mu, scale=sigma)

    # Plotar as CDFs empírica e teórica
    if render_images:
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
    absolute_differences = np.abs(mean_simulated_prices - real_prices)
    mean_absolute_difference = np.mean(absolute_differences)
    print(f"Média das Diferenças Absolutas: {mean_absolute_difference:.2f}")
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
    
    print(f"95% Confidence Interval for the final price: {mean_final_price - 1.96 * std_residuals } - {mean_final_price + 1.96 * std_residuals}")

    
    if render_images:
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
    
    from sklearn.linear_model import LinearRegression
    
    valid_dates_numeric = np.arange(len(valid_dates)).reshape(-1, 1)
    initial_price = real_prices.iloc[0]
    real_prices_adjusted = real_prices - initial_price
    model = LinearRegression(fit_intercept=False)
    model.fit(valid_dates_numeric, real_prices_adjusted)
    regression_line = model.predict(valid_dates_numeric) + initial_price
    
    mean_simulated_prices = np.mean(valid_simulations, axis=0)
    std_simulated_prices = np.std(valid_simulations, axis=0)
    std_error_simulated_prices = std_simulated_prices / np.sqrt(concurrent_simulations)
    upper_bound = mean_simulated_prices + std_error_simulated_prices
    lower_bound = mean_simulated_prices - std_error_simulated_prices
    
    if True:
      plt.figure(figsize=(12, 6))
      plt.plot(prices.index, prices.values, label='Real Prices', color='black', linewidth=2)
      plt.plot(valid_dates, mean_simulated_prices, label='Predicted Mean Prices', color='green', linewidth=2)
      plt.plot(valid_dates, regression_line, label='Regressão Linear com os Preços Reais', color='red', linestyle='--', linewidth=2)
      plt.title(f'Previsão de preço com dinâmica e Regressão Linear')
      plt.xlabel('Date')
      plt.ylabel(f'Price of {symbol} (R$)')
      plt.legend()
    
    plt.show()

def get_random_symbols(file_path, qty):
    data = pd.read_csv(file_path)
    symbols = data['Ticker'].tolist()
    return random.sample([f"{symbol}.SA" for symbol in symbols], qty)

if __name__ == "__main__":
  symbols = get_random_symbols("acoes-listadas-b3.csv", 5)
  start_date = pd.to_datetime('2021-10-01')
  end_date = pd.to_datetime('2023-12-31')
  init_date = pd.to_datetime('2021-11-01')
  play(symbols, start_date, end_date, init_date)
  plt.show(block=True)
  pass
  