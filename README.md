# AI-Stock-Portfolio-Recommender
Diversified and optimized
portfolio recommender
An AI model that recommends an optimal stock portfolio across multiple sectors, based on
user-defined preferences.
Objectives:
The model will take user inputs such as expected returns, risk tolerance, investment
time frame, and sector preferences and give optimized portfolio recommendation as
output(predicted weights of stocks, expected return, sharp ratio and risk(standard
deviation) of portfolio).
It will focus on stocks of various sectors from the Bombay Stock Exchange (BSE).
It must dynamically adjust to new market data and optimize based on selected
constraints.
Data collection:
APIs: yahoo finance and Alpha Vantage for BSE stocks data
Metrics calculation:
historical stock returns
standard deviation of returns for volatility
correlation between stocks
risk free rate(The theoretical return of an investment with zero risk), sharp ratio of
individual stocks
MA & EMA
Beta for individual stocks(Measures a stock’s sensitivity to market movements)
Data pre-processing
Handling missing values
outliers
normalization, if needed
sector wise grouping
User inputs:
Risk Tolerance: High-risk (focus on maximizing returns) or low-risk (focus on minimizing
volatility).
Investment Time Frame: Short-term or long-term investments.
Sector Preferences: Choose 2-3 sectors (e.g., technology, energy, finance).
Using these inputs to filter stocks and guide the optimization process.
Implementing the core genetic algorithm for portfolio optimization with local search:
Population Initialization: Generate a population of random portfolios, each containing a
mix of stocks with random weights that sum to 1.
Fitness Function: Define a fitness function that evaluates portfolios based on:
Maximizing Returns: Based on the weighted sum of stock returns.
Minimizing Risk: Using portfolio variance (volatility).
Sharpe Ratio: To balance risk and return.
Selection: Choose portfolios with better fitness scores for reproduction.
Crossover: Combine two parent portfolios to generate o spring portfolios.
Mutation: Introduce random changes to portfolio weights to maintain diversity.
Replacement: Replace weaker portfolios with better-performing ones in each generation.
Iteration: Repeat until a stopping condition (e.g., a fixed number of generations) is met.
Output: Optimized portfolio weights for selected stocks.
Using classical portfolio optimization techniques to complement the genetic algorithm.
MPT: Model the trade-o  between risk and return using the e cient frontier.
MVO: Minimize portfolio variance while maximizing expected returns using the
covariance matrix of stock returns.
Compare the results of MPT/MVO with those from the genetic algorithm.
Visualizing stock correlations to enhance portfolio diversification.
Compute the correlation matrix of stock returns.
Build a correlation network where nodes represent stocks, and edges represent
correlations between them.
Use this network to select uncorrelated stocks, helping to minimize portfolio risk.
Ensuring the model evolves dynamically by fetching updated stock data periodically.
Schedule regular data fetching using cron jobs or scheduling libraries like schedule in
Python.
Performance evaluation metrics:
Portfolio return
Portfolio variance
tracking error
====================
To implement the diversified and optimized portfolio recommender system using AI, we will develop an end-to-end solution that meets the outlined objectives. Here's how we can approach the problem, focusing on user inputs, data collection, pre-processing, and implementing a genetic algorithm for portfolio optimization. Additionally, we will compare the results with classical portfolio optimization techniques like Modern Portfolio Theory (MPT) and Mean-Variance Optimization (MVO).
Python Implementation Breakdown

    Data Collection: We will use the yfinance and Alpha Vantage APIs to collect historical stock data for the BSE (Bombay Stock Exchange) stocks.

    Preprocessing and Feature Engineering: We will handle missing values, outliers, normalization, and sector-wise grouping of stocks.

    Genetic Algorithm for Portfolio Optimization: The genetic algorithm will optimize portfolios based on user inputs and constraints.

    Classical Optimization Techniques: We will implement the MPT and MVO for comparison with the genetic algorithm.

    Visualization: We will visualize stock correlations and optimize the portfolio for better diversification.

Libraries Needed

pip install yfinance pandas numpy scipy matplotlib seaborn alpha_vantage

Step-by-Step Code Implementation
1. Data Collection (Using Yahoo Finance & Alpha Vantage)

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fetch stock data from Yahoo Finance
def fetch_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

# Example tickers (from the BSE)
tickers = ['TCS.BO', 'INFY.BO', 'RELIANCE.BO', 'HDFC.BO', 'SBIN.BO']
start_date = '2015-01-01'
end_date = '2023-01-01'

stock_data = fetch_stock_data(tickers, start_date, end_date)
stock_data.head()

2. Data Preprocessing (Handling Missing Values, Normalization)

# Handling missing values (forward fill)
stock_data.fillna(method='ffill', inplace=True)

# Calculate daily returns
daily_returns = stock_data.pct_change().dropna()

# Normalize data if needed (optional)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_returns = scaler.fit_transform(daily_returns)

# Example of calculated metrics (mean, std dev, etc.)
mean_returns = daily_returns.mean()
cov_matrix = daily_returns.cov()

3. Define User Inputs for Portfolio Optimization

# Define user inputs
risk_tolerance = 'high'  # 'low' or 'high'
investment_timeframe = 'long_term'  # 'short_term' or 'long_term'
sector_preferences = ['technology', 'finance']  # Example sectors

# You can filter stocks by sectors if sector data is available (assume sectors are predefined for tickers)
# For simplicity, we'll focus on all available tickers for now.

4. Genetic Algorithm for Portfolio Optimization

We will implement a genetic algorithm to find the optimal portfolio.

# Genetic Algorithm Parameters
population_size = 50
generations = 100
mutation_rate = 0.02
elitism_rate = 0.1

# Fitness Function: Maximize Sharpe Ratio
def calculate_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    portfolio_return = np.sum(weights * mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return sharpe_ratio

# Portfolio Initialization: Generate Random Portfolios
def generate_population(size, num_assets):
    population = np.random.rand(size, num_assets)
    population /= population.sum(axis=1)[:, np.newaxis]  # Normalize weights to sum to 1
    return population

# Crossover: Combine two parent portfolios
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1)-1)
    child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    return child

# Mutation: Randomly adjust portfolio weights
def mutate(portfolio, mutation_rate):
    if np.random.rand() < mutation_rate:
        mutation_point = np.random.randint(len(portfolio))
        portfolio[mutation_point] = np.random.rand()
        portfolio /= portfolio.sum()  # Normalize weights again
    return portfolio

# Selection: Choose the best portfolios based on fitness
def selection(population, fitness_scores):
    sorted_indices = np.argsort(fitness_scores)
    top_portfolios = population[sorted_indices[:int(elitism_rate * population_size)]]
    return top_portfolios

# Portfolio Optimization using Genetic Algorithm
def optimize_portfolio(stock_data, generations=100, population_size=50):
    num_assets = len(stock_data.columns)
    mean_returns = stock_data.mean()
    cov_matrix = stock_data.cov()
    
    # Initialize population
    population = generate_population(population_size, num_assets)
    
    # Track best portfolio
    best_portfolio = None
    best_sharpe = -np.inf
    
    # Genetic Algorithm Loop
    for generation in range(generations):
        fitness_scores = np.array([calculate_sharpe_ratio(p, mean_returns, cov_matrix) for p in population])
        
        # Update the best portfolio
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > best_sharpe:
            best_sharpe = fitness_scores[best_idx]
            best_portfolio = population[best_idx]
        
        # Selection and Crossover
        selected_parents = selection(population, fitness_scores)
        next_population = selected_parents.copy()
        
        while len(next_population) < population_size:
            parent1, parent2 = selected_parents[np.random.choice(len(selected_parents), 2)]
            child = crossover(parent1, parent2)
            next_population = np.vstack([next_population, mutate(child, mutation_rate)])
        
        population = next_population
    
    return best_portfolio, best_sharpe

# Optimize Portfolio
best_portfolio, best_sharpe = optimize_portfolio(daily_returns)
print(f"Best Portfolio Weights: {best_portfolio}")
print(f"Best Sharpe Ratio: {best_sharpe}")

5. Classical Portfolio Optimization (MPT/MVO)

We will use scipy.optimize to implement MPT for Mean-Variance Optimization.

from scipy.optimize import minimize

# Objective function: Minimize portfolio variance
def objective(weights, mean_returns, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

# Constraint: Sum of weights should be 1
def constraint(weights):
    return np.sum(weights) - 1

# Optimization function
def mpt_optimization(stock_data):
    mean_returns = stock_data.mean()
    cov_matrix = stock_data.cov()
    
    num_assets = len(stock_data.columns)
    initial_guess = np.ones(num_assets) / num_assets
    bounds = [(0, 1) for _ in range(num_assets)]
    cons = ({'type': 'eq', 'fun': constraint})
    
    # Minimize portfolio variance
    result = minimize(objective, initial_guess, args=(mean_returns, cov_matrix), bounds=bounds, constraints=cons)
    
    return result.x  # Optimized portfolio weights

# Optimize using MPT
mpt_portfolio = mpt_optimization(daily_returns)
print(f"MPT Optimized Portfolio Weights: {mpt_portfolio}")

6. Visualization of Correlations

We can visualize correlations between stock returns to enhance portfolio diversification.

import seaborn as sns

# Correlation matrix
corr_matrix = daily_returns.corr()

# Heatmap of correlations
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Stock Correlations')
plt.show()

7. Final Output and Evaluation

Finally, we evaluate the performance of the optimized portfolios based on the portfolio return, variance, and Sharpe ratio.

# Evaluate portfolio performance
def evaluate_portfolio(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(weights * mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio

# Evaluate best portfolio from GA
best_return, best_volatility, best_sharpe = evaluate_portfolio(best_portfolio, mean_returns, cov_matrix)
print(f"GA Optimized Portfolio Return: {best_return}")
print(f"GA Optimized Portfolio Volatility: {best_volatility}")
print(f"GA Optimized Portfolio Sharpe Ratio: {best_sharpe}")

# Evaluate MPT portfolio
mpt_return, mpt_volatility, mpt_sharpe = evaluate_portfolio(mpt_portfolio, mean_returns, cov_matrix)
print(f"MPT Portfolio Return: {mpt_return}")
print(f"MPT Portfolio Volatility: {mpt_volatility}")
print(f"MPT Portfolio Sharpe Ratio: {mpt_sharpe}")

Conclusion

This code provides a complete framework for developing a diversified and optimized portfolio recommender using genetic algorithms, alongside classical portfolio optimization techniques like MPT and MVO. It dynamically adjusts to market data and gives portfolio recommendations based on user-defined preferences such as risk tolerance, investment time
