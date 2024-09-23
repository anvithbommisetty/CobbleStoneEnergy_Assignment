import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from AnomalyDetection import detect_anomalies_with_ema

"""
This script visualizes dynamic price data with anomaly detection using matplotlib and FuncAnimation.
Functions:
    init(): Initializes the plot by clearing the data.
    update(t): Updates the plot with data up to the current time step `t`.
Modules:
    numpy: Used for numerical operations.
    matplotlib.pyplot: Used for plotting.
    matplotlib.animation.FuncAnimation: Used for creating animations.
    AnomalyDetection: Custom module for detecting anomalies.
Variables:
    df: DataFrame containing the price data and anomaly detection results.
    fig: Figure object for the plot.
    ax: Axis object for the plot.
    line_normal: Line2D object for normal data points.
    scat_anomaly: Line2D object for anomaly data points.
    ani: FuncAnimation object for creating the animation.
Usage:
    Run the script to visualize the dynamic price plot with anomalies. The plot will update in real-time, showing normal data points in blue and anomalies in red.
"""

# Get the price data with anomalies from the anomaly detection function
df = detect_anomalies_with_ema(span=20, initial_threshold_factor=2.0, decay_rate=0.0001)

# figure setup and axis
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, df['Time'].iloc[-1])
ax.set_ylim(0, df['Price'].max() * 1.1)

# plot placeholders
line_normal, = ax.plot([], [], label='No Anomaly')  # Blue for no anomaly
scat_anomaly, = ax.plot([], [], 'ro',label='Anomaly')    # Red for anomaly

# Initialize function to clear the plot at the start
def init():
    line_normal.set_data([], [])
    scat_anomaly.set_data([], [])
    return line_normal, scat_anomaly

# Update function to dynamically plot data points
def update(t):
    # Get data up to current time step
    x = df['Time'][:t]
    y = df['Price'][:t]
    
    # Filter anomaly points
    normal_x = x
    normal_y = y
    anomaly_x = x[df['Anomaly'][:t] == 1]
    anomaly_y = y[df['Anomaly'][:t] == 1]
    
    # Update the data for both normal and anomaly points
    line_normal.set_data(normal_x, normal_y)
    scat_anomaly.set_data(anomaly_x, anomaly_y)
    
    return line_normal, scat_anomaly

# the animation part
# blit is used for optimization
ani = FuncAnimation(fig, update, frames=np.arange(1, len(df)+1), init_func=init,interval=50, blit=True, repeat=False)

# titles and labels
ax.set_title("Dynamic Price Plot with Anomalies")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()

# Display plot
plt.show()
