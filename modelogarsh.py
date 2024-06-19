import numpy as np
import scipy.optimize as spop
import matplotlib.pyplot as plt
import yfinance as yf



tickers = ['MTRE3.SA', 'LWSA3.SA', 'MDNE3.SA', 'PRNR3.SA', 'ALPK3.SA', 'AMBP3.SA', 'SOMA3.SA', 'DMVF3.SA',
         'LJQQ3.SA', 'LAVV3.SA', 'PGMN3.SA', 'PETZ3.SA', 'PLPL3.SA', 'CURY3.SA', 'HBSA3.SA', 'MELK3.SA',
         'SEQL3.SA', 'GMAT3.SA', 'TFCO4.SA', 'CASH3.SA', 'ENJU3.SA', 'AERI3.SA', 'RRRP3.SA', 'RDOR3.SA', 'NGRD3.SA']

historicos = []

for ticker in tickers:
    historico = yf.Ticker(ticker).history(period='max')
    historicos.append(historico)



for i, ticker in enumerate(tickers):
    print(f"Dados históricos para {ticker}:")
    print(historicos[i])
    print()


# ----- maximização da máxima verossimilhança (MLE) do GARCH -----
def garch_mle(returns, params):
    mu = params[0]
    omega = params[1]
    alpha = params[2]
    beta = params[3]
    
    # -----volatilidade de longo prazo-----
    long_run = (omega/(1 - alpha - beta))**(1/2)
    
    # -----volatilidade realizada e condicional-----
    resid = returns - mu
    realised = abs(resid)
    conditional = np.zeros(len(returns))
    conditional[0] = long_run
    
    for t in range(1,len(returns)):
        conditional[t] = (omega + alpha*resid[t-1]**2 + beta*conditional[t-1]**2)**(1/2)
    
    # Calculando a log-verossimilhança
    likelihood = 1/((2*np.pi)**(1/2)*conditional)*np.exp(-realised**2/(2*conditional**2))
    log_likelihood = np.sum(np.log(likelihood))
    
    return -log_likelihood

# Lista de parâmetros iniciais para a otimização (média, omega, alpha, beta)
initial_params = [0, 0.01, 0.1, 0.8]

optimal_params = []


for historico in historicos:
  
    returns = historico['Close'].pct_change().dropna().values
    
   
    res = spop.minimize(garch_mle, initial_params, args=(returns,), method='Nelder-Mead')
    
    
    optimal_params.append(res.x)


for i, ticker in enumerate(tickers):
    print(f"Parâmetros ótimos para {ticker}: {optimal_params[i]}")


for historico, ticker in zip(historicos, tickers):
    plt.plot(historico.index, historico['Close'], label=ticker)



params = optimal_params[0]
mu = params[0]
omega = params[1]
alpha = params[2]
beta = params[3]

# -----volatilidade realizada e condicional para os parâmetros ótimos-----
long_run = (omega/(1 - alpha - beta))**(1/2)
resid = returns - mu
realised = abs(resid)
conditional = np.zeros(len(returns))
conditional[0] = long_run

for t in range(1,len(returns)):
    conditional[t] = (omega + alpha*resid[t-1]**2 + beta*conditional[t-1]**2)**(1/2)

# -----parâmetros ótimos-----
print('Parâmetros ótimos para a primeira ação:')
print('')
print('média: ', round(mu, 6))
print('omega: ', round(omega, 6))
print('alpha: ', round(alpha, 4))
print('beta: ', round(beta, 4))
print('volatilidade de longo prazo: ', round(long_run, 4))

# -----Resultados-----
plt.title('Preço de Fechamento Ajustado das Ações')
plt.xlabel('Data')
plt.ylabel('Preço de Fechamento Ajustado')
plt.legend()  
plt.grid(True)  
plt.xticks(rotation=45)  


# -----Resultados-----
plt.figure(figsize=(10, 6))
plt.plot(realised, label='Volatilidade Realizada')
plt.plot(conditional, label='Volatilidade Condicional')
plt.xlabel('Tempo')
plt.ylabel('Volatilidade')
plt.title('Volatilidade Realizada vs. Volatilidade Condicional')
plt.legend()
plt.grid(True)
plt.show()
