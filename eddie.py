# Import Streamlit for creating site, Yahoo Finance for financial data, pandas, numpy, and matplotlib for working with data
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configure heading
st.set_page_config(layout="wide", page_title="Stockify Analysis Tool")

# Title & Introduction (Use HTML and CSS to structure and style the web page)
with st.container():
    st.markdown("""
        <style>
            h3 {
                margin-top: 20px;
            }
            [data-testid="stSidebar"] {
                background-color: rgb(1 20 11);
                color:white;
            }

            [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
                background-color: rgb(0 6 3);
                color:white;
            }
                 
            p {
                color: white;
            }
                
            button div p {
                color: black;
            }
            span {
                color: #2E8B57;
            }
        </style>
        <h1 style="text-align: center; color: #2E8B57; font-size: 50px; font-weight: bold; margin-bottom: 0;">Stockify Analysis</h1>
        <hr style="height: 4px; background-color: #2E8B57; margin: 25px 0;">
        <div style="background-color: rgb(1 20 11); padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px;">
            <p style="text-align: center; font-size: 22px; color: rgb(142 161 149); font-weight: bold;">
                Enter up to five different stocks to recieve actionable insights.
            </p>
            <p style="text-align: center; font-size: 22px; color: rgb(142 161 149); font-weight: bold;">
                Choose a time that best fits your needs and recieve your results.
            </p>
        </div>
    """, unsafe_allow_html=True)

# Function that accepts a list of stock symbols, start date, and end date. Returns price.
def get_stock_data(stock_symbols, start_date, end_date):
    all_data = yf.download(stock_symbols, start=start_date, end=end_date)
    price = all_data['Adj Close']
    return price

# Function that calculates risk, expected return, and CAGR for given stocks.
def analyze_stocks(price, stock_symbols, start_date, end_date):
    # Calculate Log Returns, a number in finance that allows us to generate statistics by comparing the value of a stock at two price points
    log_returns = np.log(price / price.shift(1))
    # Create dictionary to store statistics
    stock_stats = {}
    # Loop over the stocks and add the statistics to the dictionaries, creating a dictionary of key value pairs in which the value is another dictionary that holds the statistics of the key.
    for stock in stock_symbols:
        stock_stats[stock] = {
            # Calculate risk (standard deviation of of Log Returns). Round it to 5 decimals.
            "risk": round(log_returns[stock].std(), 5),
            # Calculate the expected return (mean of Log Returns)
            "expected_return": round(log_returns[stock].mean(), 5),
            # Calculate CAGR (Compund Annual Growth Rate)
            "cagr": round(((price[stock][-1] / price[stock][0]) ** (1 / ((end_date - start_date).days / 365)) - 1), 5)
        }
    # If there is more than one stock, take the price dataframe and calculate the correlation matrix and append it to our dictionary for later use.
    if len(stock_symbols) > 1:
        stock_stats["correlation"] = round(price.corr(), 3)
    # Return our dictionary
    return stock_stats

# Calculate returns, volatility, and Sharpe Ratios
def portfolio_analysis(price):
    # Log Returns but remove null values
    log_returns = np.log(price / price.shift(1)).dropna()
    
    # Get number of stocks
    num_stocks = len(price.columns)

    # Generate random weights for the portifolio. Create a 2D Array with size 50000 x num_stocks and allocate a random weight for each stock
    all_weights = np.random.rand(50000, num_stocks)
    # Since we are using randomization, the values could be above 100%, so to fix this, we normalize the values so that they all add up to 100%. This is done by dividing each value by the sum of values.
    # We can do this by creating a 2D column vector that contains the sum of each row, allowing for easier division.
    all_weights /= all_weights.sum(axis=1)[:, np.newaxis]
    
    # Calculate portifolio returns, volitility, and use those to calculate the sharpe ratio. Here we get the daily amount, and multiply by the number of trading days (252) to get our yearly returns
    ret_arr = np.dot(all_weights, log_returns.mean() * 252)
    # Calculate volatility. Use Einstein Summation instead of loops.  The formula for volatility is covariance between our weights, multiplied by the product of our weights. 
    vol_arr = np.sqrt(np.einsum('ij,jk,ik->i', all_weights, log_returns.cov() * 252, all_weights))
    # Use ternary operator to check if the volatility is zero. If it is, set the Sharpe ratio to 0, else, the Sharpe ratio should be return/volatility which we already dervied above.
    sharpe_arr = np.where(vol_arr != 0, ret_arr / vol_arr, 0) 

    # Find the index with the largest Sharpe Ratio
    max_sharpe_idx = sharpe_arr.argmax()

    # Return the metrics for the optimal portifolio as well as the data for every portifolio so we can plot it.
    return all_weights[max_sharpe_idx], ret_arr[max_sharpe_idx], vol_arr[max_sharpe_idx], sharpe_arr[max_sharpe_idx], ret_arr, vol_arr, sharpe_arr



# User Input for stock symbols
stock_symbol_1 = st.sidebar.text_input('Enter stock symbol 1', '').strip().upper()
stock_symbol_2 = st.sidebar.text_input('Enter stock symbol 2', '').strip().upper()
stock_symbol_3 = st.sidebar.text_input('Enter stock symbol 3', '').strip().upper()
stock_symbol_4 = st.sidebar.text_input('Enter stock symbol 4', '').strip().upper()
stock_symbol_5 = st.sidebar.text_input('Enter stock symbol 5', '').strip().upper()

# Save stock symbols to a variable for insertion into functions
stock_symbols = [stock_symbol_1, stock_symbol_2, stock_symbol_3, stock_symbol_4, stock_symbol_5]

# set date ranges, default is hardcoded below
date_range = st.sidebar.date_input(
    'Select date range',
    # Set to todays date
    [pd.to_datetime('2019-01-01'), pd.to_datetime('2023-12-5')]
)

# If the user enters a start and end date, set the variables, else throw an error
if len(date_range) == 2:
    start_date, end_date = date_range
else:
    st.error("Please select a valid date range.")
    start_date, end_date = None, None 

# Append a button and event listener to the side bar
calculate = st.sidebar.button('Analyze Stocks')

# If the button is pressed
if calculate:
    # Get key metrics for display
    price = get_stock_data(stock_symbols, start_date, end_date)
    stock_stats = analyze_stocks(price, stock_symbols, start_date, end_date)

    # For every stock, print the annual risk, annual expected return, and CAGR. Use python formatting (template literals)
    for stock in stock_symbols:
        stats = stock_stats.get(stock, {})
        if stats:
            st.markdown(f"### {stock} Metrics")
            st.write(f"{stock} annualized stock risk: {stats['risk'] * 100:.2f}%")
            st.write(f"{stock} annualized stock expected returns: {stats['expected_return'] * 100:.2f}%")
            st.write(f"{stock} stock CAGR: {stats['cagr'] * 100:.2f}%")

    # If there is a stock
    if len(stock_symbols) > 1:
        # Create heading for correlation matrix
        st.write("### Correlation Matrix")

        # Style the matrix for easier reading
        styled_corr = stock_stats["correlation"].style.background_gradient(cmap='RdYlGn',vmin=-1, vmax=1).format("{:.2f}")
        st.dataframe(styled_corr)

        # UI Explanation for the Correlation Matrix
        st.write("""
        ### Understanding the Correlation Matrix
        The correlation matrix measures how changes in one stock's prices are associated with changes in another stock's prices. Values range between -1 and 1:
        
        - **1**: Perfect positive correlation. Both stocks tend to move in the same direction.
        - **-1**: Perfect negative correlation. When one stock goes up, the other tends to go down.
        - **0**: No correlation. The movements of the stocks are not related.
        
        Correlations can help in diversifying a portfolio. For instance, mixing stocks with low or negative correlations can reduce the portfolio's overall risk.
        """)

 

    # Get our data from the portofolio analysis function
        optimized_weights, optimized_ret, optimized_vol, optimized_sr, ret_arr, vol_arr, sharpe_arr = portfolio_analysis(price)

        # Get a list of our optimized weights by using zip to iterate through our symbols and weights wil joining them all into a string
        optimized_weights_str = ", ".join([f"{stock}: {weight:.2f}" for stock, weight in zip(stock_symbols, optimized_weights)])
        st.write("### Optimized Weights")
        st.write(optimized_weights_str)
        st.write("""
These represent the proportion of the total investment that should be allocated to each stock in order to optimize the portfolio based on the Sharpe Ratio. The values sum up to 1 or 100%.
""")
        
    # Create a pie chart of the weights. First create a matplotbib figure and axis with a height and length of 2.5 inches
        fig, ax = plt.subplots(figsize=(2.5, 2.5))  
        # Move chart over so we can put the legend on the side
        fig.subplots_adjust(left=0.1, right=0.75)  
        # Mtplotlib color options
        colors = plt.cm.Set1.colors
        # Create a pie chart with the optimized weights using the colors above
        wedges, _ = ax.pie(optimized_weights, startangle=90, colors=colors, shadow=False, wedgeprops=dict(width=0.3, edgecolor='white'))
        # Make sure the pie chart stays as a circle
        ax.axis('equal') 

        # Configure Title
        title_text = 'Portfolio Weights'
        plt.title(title_text, fontsize=12, color='black', fontweight='bold', pad=10)

        # Create a list of percentages from our optimized weights (ie. 0.5 -> 50%)
        percentages = [f"{weight*100:.1f}%" for weight in optimized_weights]
        # For each stock, create a string in the form of [Stock - Percentage]
        legend_labels = [f"{label} - {percent}" for label, percent in zip(stock_symbols, percentages)]
        # Adds legend to the pie chart. Sets wedges as the key, and legend_lables as the value. Move it to the side.
        plt.legend(wedges, legend_labels, title="Allocation", loc="center left", fontsize=8, title_fontsize=10, bbox_to_anchor=(1, 0, 0.5, 1))

        # Border Styling
        fig.patch.set_linewidth(2)  
        fig.patch.set_edgecolor('grey')  
        fig.patch.set_facecolor('black')

        # Set the text color to white
        ax.set_facecolor('black')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.tick_params(axis='both', colors='white')

        # Push Figure
        st.pyplot(fig)


    # Display the Returns, Volatility, and Sharpe Ratio with a nice little explanation
        st.write("### Optimized Portfolio Return")
        st.write(f"{optimized_ret * 100:.2f}%") 
        st.write("""
        This is the expected annual return of the optimized portfolio, given the weights. 
        """)
        st.write("### Optimized Portfolio Volatility")
        st.write(f"{optimized_vol * 100:.2f}%")
        st.write("""
        This represents the standard deviation of the optimized portfolio's return, which is a measure of its risk. A higher volatility indicates a wider range of potential outcomes for the portfolio's return, implying more risk.
        """)
        st.write("### Optimized Portfolio Sharpe Ratio")
        st.write(f"{optimized_sr:.4f}")
        st.write("""
        The Sharpe Ratio is a measure of risk-adjusted return. It's calculated as the portfolio's excess return (over the risk-free rate) divided by its volatility.In simpler terms, a higher Sharpe Ratio indicates that the portfolio is delivering better returns for the level of risk taken. Generally, a ratio above 1 is favorable, with higher values being increasingly desirable. The 'optimized' portfolio showcased here has the highest Sharpe Ratio.
        """)

    # Display all the portifolios we randomly calculated in an efficient frontier
        # Create new figure of 12 x 8 inches
        fig, ax = plt.subplots(figsize=(12, 8))
        # Create a scatterplot in x,y form with vol_arr (volatility) as x and ret_arr (return) as y. set the color of the dot to depend on the sharpe ratio. Color map as viridis, transparency .6.
        sc = ax.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis', alpha=0.6, edgecolors="w", linewidth=0.5)
        # Add a color bar to the right that automatically adjusts to encapsulate all Sharpe Factors in our dataset
        # LABEL MISSING
        plt.colorbar(sc, label='Sharpe Ratio', ax=ax).set_label('Sharpe Ratio', color='white')
        
        # Set x, y, and graph labels
        ax.set_title('Efficient Frontier For Your Stocks', fontsize=18, )
        ax.set_xlabel('Volatility (Standard Deviation)', fontsize=14)
        ax.set_ylabel('Expected Return', fontsize=14)

        # Add x and y limits of the plot (similar to calculator window)
        if np.isfinite(vol_arr.min()) and np.isfinite(vol_arr.max()):
            ax.set_xlim([vol_arr.min() - 0.05, vol_arr.max() + 0.05])
        if np.isfinite(ret_arr.min()) and np.isfinite(ret_arr.max()):
            ax.set_ylim([ret_arr.min() - 0.05, ret_arr.max() + 0.05])

        # Create another scatter plot with a singular point that is the color red, representing our optimized portofolio. Increase its size so it pops. Add label for legend
        ax.scatter(optimized_vol, optimized_ret, c='red', s=100, edgecolors="k", label='Optimized Portfolio')

        # Add legend to explain the optimized porifolio
        ax.legend(fontsize=12)

        # Style Border
        fig.patch.set_linewidth(2)  
        fig.patch.set_edgecolor('grey') 

        # Border Styling
        fig.patch.set_linewidth(2)  
        fig.patch.set_edgecolor('grey')  
        fig.patch.set_facecolor('black')

        # Set the text color to white
        ax.set_facecolor('black')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.tick_params(axis='both', colors='white')
        
        # Append Figure
        st.pyplot(fig)

    # Explain Figure
    st.write("""
        ## Efficient Frontier
        The scatter plot above represents various portfolio combinations and their expected returns against their volatilities.
        
        - **Points**: Each point signifies a portfolio with a specific combination of the stocks you've entered.
        - **Color**: The color of the points indicates the Sharpe Ratio. Yellow points have a higher Sharpe ratio, implying better risk-adjusted returns.
        - **Red Dot**: Represents the 'Optimized Portfolio', the best combination of stocks that offers the maximum expected return for a given level of risk.
        
        In general, you'd want a portfolio that's towards the top left of the graph, where you get the best returns for best risk.
    """)

