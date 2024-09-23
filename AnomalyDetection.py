import numpy as np
import pandas as pd
from DataStream import generate_price_series

'''
Reason for Choosing Exponential Moving Average (EMA) for Anomaly Detection:

I chose the Exponential Moving Average (EMA) for this analysis due to its numerous advantages in handling time series data characterized by concept drift and seasonality.
Firstly, EMA is highly sensitive to recent data, allowing it to quickly adapt to sudden changes in trends, which is essential in dynamic environments.
It also effectively smooths variability, helping to reveal underlying seasonal patterns without being overly influenced by noise.
Moreover, EMA's adaptability ensures it can track changing conditions, making it suitable for processes that evolve over time.
Its reduced lag compared to simple moving averages allows for more timely forecasts and anomaly detections.
Finally, the flexibility in adjusting the smoothing factor (alpha) enables optimization based on the specific characteristics of the dataset, enhancing overall performance.
These features make EMA a powerful tool for detecting anomalies in time series data, particularly in scenarios with evolving patterns and concept drift.
'''

# Function to apply exponential decay to the threshold value
def exponential_decay_threshold(threshold_factor, anomaly_detected, time_step, reset_value=2.0, min_value=1.75, decay_rate=0.00001):
    """
    Calculate the threshold value with exponential decay.

    Parameters:
    threshold_factor (float): The initial threshold factor.
    anomaly_detected (bool): Flag indicating if an anomaly has been detected.
    time_step (int): The current time step.
    reset_value (float, optional): The value to reset the threshold to when an anomaly is detected. Default is 2.0.
    min_value (float, optional): The minimum value the threshold can decay to. Default is 1.75.
    decay_rate (float, optional): The rate at which the threshold decays. Default is 0.00001.

    Returns:
    float: The updated threshold value.
    """
    if anomaly_detected:
        return reset_value  # Reset to 2.0 when an anomaly is detected
    else:
        decayed_value = threshold_factor * np.exp(-decay_rate * time_step)  # Apply decay
        return max(decayed_value, min_value)  # Ensure it doesn't go below 1.75

# Anomaly detection using EMA and dynamic thresholds with exponential decay
def detect_anomalies_with_ema(span=20, initial_threshold_factor=2.0, decay_rate=0.00001):
    """
    Detect anomalies in a price series using Exponential Moving Average (EMA) and dynamic thresholds.
    Parameters:
    span (int): The span for calculating the EMA. Default is 20.
    initial_threshold_factor (float): The initial factor to determine the threshold for anomaly detection. Default is 2.0.
    decay_rate (float): The rate at which the threshold factor decays over time. Default is 0.00001.
    Returns:
    pd.DataFrame: A DataFrame containing the original price series, time, and a boolean column 'Anomaly' indicating whether each point is an anomaly.
    Notes:
    - The function assumes the existence of a `generate_price_series` function that returns a tuple of (price, time).
    - The function also assumes the existence of an `exponential_decay_threshold` function to update the threshold factor.
    - Anomalies are detected based on whether the price deviates significantly from the EMA, considering a dynamically adjusted threshold.
    """
    [price,time] = generate_price_series()
    # Calculate EMA and EMA standard deviation
    df = pd.DataFrame(price, columns=['Price'])
    df['Time'] = time
    ema = df['Price'][0]  # Start with the first price
    ema_std = 0.05*ema  # Initial standard deviation
    
    threshold_factor = initial_threshold_factor  # Start with initial threshold factor
    recent_anomaly = 0
    
    # Loop over data and detect anomalies
    for t in range(1,len(df)):
        ema_before  = ema
        ema_std_before = ema_std
        
        # Calculate alpha value based on the time step
        if t>span :
            alpha = 3/(span+1)
        else :
            alpha = 3/(t+2)
        
        ema = (df['Price'].iloc[t] * alpha) + (ema * (1 - alpha))
        diff = df['Price'].iloc[t] - ema
        ema_std = np.sqrt((alpha * diff**2) + ((1 - alpha) * ema_std**2))
        
        # Calculate dynamic thresholds
        upper_threshold = ema + threshold_factor * ema_std 
        lower_threshold = ema - threshold_factor * ema_std 
        
        # Check if current point is an anomaly
        is_anomaly = (df['Price'].iloc[t] > upper_threshold) or (df['Price'].iloc[t] < lower_threshold)
        df.loc[df.index[t], 'Anomaly'] = is_anomaly
        
        '''
        When an anomaly is detected, its value is adjusted by calculating the 
        difference between the current price and the previous EMA, scaled by 0.10 to reduce its impact. 
        This smoothed value is then used to update the EMA, ensuring it reflects the anomaly without overwhelming the underlying trend. A minimum alpha value is applied to balance responsiveness and stability. The EMA standard deviation is recalculated based on this adjusted value. 
        This approach allows for adaptive tracking of the data while mitigating the influence of outliers.
        '''
        if is_anomaly:
            recent_anomaly = t
            diff = (df['Price'].iloc[t] - ema_before)*(0.10)
            new_point = ema_before + diff
            min_alpha = alpha*2/3
            ema = (new_point * min_alpha) + (ema_before * (1 - min_alpha))
            ema_std = np.sqrt((min_alpha * (diff)**2) + ((1 - min_alpha) * ema_std_before**2))
            
        # Update the threshold factor: reset if anomaly is detected, else decay
        threshold_factor = exponential_decay_threshold(threshold_factor, is_anomaly, t-recent_anomaly, decay_rate=decay_rate)

    return df

