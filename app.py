import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

# App Title
st.title("Stock Market Portfolio Optimization")

# Sidebar Input for Stock Selection
st.sidebar.header("Portfolio Settings")
tickers = st.sidebar.text_input("Enter stock tickers (comma-separated)", "AAPL, MSFT, GOOG, AMZN")
start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.Timestamp("2023-01-01"))

# Fetch Stock Data
if st.sidebar.button("Fetch Data"):
    tickers_list = [ticker.strip() for ticker in tickers.split(",")]

    if not tickers_list:
        st.error("Please enter at least one stock ticker.")
    else:
        try:
            # Download stock data
            data = yf.download(tickers_list, start=start_date, end=end_date)

            # Use 'Close' column
            if "Close" in data.columns:
                data = data["Close"]
            else:
                st.write("Columns available in data:", data.columns)
                st.error("The 'Close' column is not available. Please verify the stock tickers.")
                st.stop()

            st.write("### Stock Price Data")
            st.write("The stock price trends over time are shown in the graph below. You can observe how each stock has performed within the selected date range. Look for price movements, increases, and decreases over time.")
            st.write("### Insights:")
            st.write("- The X-axis represents the time (from the start date to the end date).")
            st.write("- The Y-axis shows the stock's closing price.")
            st.write("- You can compare multiple stocks to see which ones have performed better or worse.")
            st.write("- A rising stock suggests growth, while a falling stock shows decline.")
            st.write("- Volatility (frequent ups and downs) indicates higher risk.")
            st.write("### Stock Price Trends")
            fig, ax = plt.subplots(figsize=(10, 6))
            data.plot(ax=ax)
            st.pyplot(fig)

            # Calculate Daily Returns
            returns = data.pct_change().dropna()

            # Correlation Matrix
            correlation_matrix = returns.corr()

            # Plot Correlation Matrix using Seaborn Heatmap
            st.write("### Correlation Matrix")
            st.write("The correlation matrix shows the relationships between the stock returns. A value of +1 means that the stocks move perfectly in sync, while -1 indicates they move in opposite directions. A value close to 0 means there is no correlation.")
            st.write("### Insights:")
            st.write("- High correlation (close to +1) means the stocks move in similar patterns.")
            st.write("- Negative correlation (close to -1) means the stocks move in opposite directions.")
            st.write("- Diversification benefits come from stocks with low or negative correlation.")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True, ax=ax)
            st.pyplot(fig)

            # Portfolio Optimization
            st.write("### Portfolio Optimization Results")
            st.write("This section provides the optimal portfolio allocation based on maximizing the Sharpe Ratio. The goal is to find the portfolio with the best risk-adjusted return.")
            mean_returns = returns.mean()
            cov_matrix = returns.cov()

            # Risk-Return Functions
            def portfolio_performance(weights):
                port_return = np.dot(weights, mean_returns)
                port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe = port_return / port_risk
                return -sharpe  # Negative Sharpe Ratio for minimization

            # Constraints and Bounds
            num_assets = len(mean_returns)
            bounds = tuple((0, 1) for _ in range(num_assets))
            constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}

            # Initial Guess
            initial_weights = num_assets * [1.0 / num_assets]

            # Optimization
            result = minimize(
                portfolio_performance,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            optimal_weights = result.x
            st.write("Optimal Portfolio Weights:")
            st.write(f"The optimal portfolio allocation is as follows (in percentage terms):")
            st.write(dict(zip(tickers_list, optimal_weights)))

            st.write("### Insights:")
            st.write("- The weights show how much of each stock should be included in your portfolio for the best risk-adjusted return.")
            st.write("- Stocks with higher weights are considered more optimal for maximizing returns given the portfolio's risk profile.")
            st.write("- If the weights are very different from each other, your portfolio is not well-diversified.")

            # Efficient Frontier
            st.write("### Efficient Frontier")
            st.write("The Efficient Frontier graph shows the optimal portfolios' risk-return trade-offs. The goal is to pick a portfolio on the frontier that aligns with your risk tolerance and return expectations.")
            target_returns = np.linspace(returns.mean().min(), returns.mean().max(), 100)
            risks = []
            valid_returns = []

            for ret in target_returns:
                try:
                    # Constraints: Sum of weights = 1 and target return = ret
                    constraints = [
                        {"type": "eq", "fun": lambda weights: np.sum(weights) - 1},
                        {"type": "eq", "fun": lambda weights: np.dot(weights, mean_returns) - ret},
                    ]

                    # Optimization for minimum risk
                    result = minimize(
                        lambda weights: np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))),
                        initial_weights,
                        method="SLSQP",
                        bounds=bounds,
                        constraints=constraints,
                    )

                    # Store risk and valid return if successful
                    if result.success:
                        risks.append(result.fun)
                        valid_returns.append(ret)
                except Exception as e:
                    st.write(f"Error for return {ret:.4f}: {e}")

            # Plot Efficient Frontier
            if risks and valid_returns:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(risks, valid_returns, label="Efficient Frontier", color="blue")
                ax.set_xlabel("Risk (Standard Deviation)")
                ax.set_ylabel("Return")
                ax.set_title("Efficient Frontier")
                ax.legend()
                st.pyplot(fig)

                st.write("### Insights:")
                st.write("- The efficient frontier helps you select a portfolio that balances the highest return for the least amount of risk.")
                st.write("- Portfolios below the frontier are suboptimal, as they do not offer the best risk-return combination.")
                st.write("- Higher positions on the frontier mean greater return for a given level of risk.")
            else:
                st.error("Efficient Frontier could not be generated. Check the input data or constraints.")
            
            # Additional Tips and Justifications
            st.write("### Investment Tips Based on Our Analysis:")
            st.write("Based on the analysis and calculations, here are some tips to improve your investment strategy:")

            # Diversification Tip
            if correlation_matrix.max().max() > 0.7:
                st.write("- **Consider diversifying your portfolio.** Based on the correlation matrix, some of your selected stocks have high correlations. Adding stocks that are less correlated with each other can reduce risk.")
            else:
                st.write("- **Good diversification already present.** The correlation matrix shows that your stocks are well-diversified.")

            # Risk and Return Balance Tip
            avg_sharpe_ratio = np.dot(optimal_weights, mean_returns) / np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
            if avg_sharpe_ratio < 1:
                st.write("- **Increase risk for higher return.** The Sharpe ratio is below 1, which suggests that you may consider accepting a bit more risk to increase potential returns.")
            else:
                st.write("- **Optimal risk-return balance.** Your portfolio offers a good trade-off between risk and return based on the Sharpe ratio.")

            # Portfolio Weights Tip
            if max(optimal_weights) > 0.5:
                st.write("- **Avoid concentration in one stock.** If any stock has a weight above 50%, you may want to reduce its proportion to avoid over-concentration and reduce risk.")
            else:
                st.write("- **Well-balanced portfolio.** Your portfolio has a good spread of investments, minimizing the risks associated with over-concentration in one stock.")
            
            # Efficient Frontier Tip
            st.write("- **Choose your position on the Efficient Frontier wisely.** Depending on your risk tolerance, pick a portfolio on the frontier that offers an optimal return for the level of risk you're willing to take.")
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
