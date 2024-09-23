# Efficient Data Stream Anomaly Detection

## Overview
This project implements a Python script designed to detect anomalies in continuous data streams, with a focus on financial metrics and system metrics. It leverages statistical method of Exponential Moving Averages (EMA), to identify unusual patterns and facilitate robust anomaly detection.

## Features
- **Dynamic Thresholding**: Automatically adjusts thresholds based on recent observations, enhancing detection accuracy.
- **Anomaly Detection Algorithms**: Utilizes Exponential Moving Average (EMA) for effective anomaly identification.
- **Data Stream Simulation**: Simulates a continuous stream of floating-point numbers, incorporating concept drift and seasonal variations.
- **Real-time Visualization**: Provides a real-time display of the data stream alongside detected anomalies.

## Getting Started

### Prerequisites
- **Python**: Version 3.x
- **Required Libraries**: Refer to the `requirements.txt` file for a complete list of dependencies.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/anvithbommisetty/CobbleStoneEnergy_Assignment.git
   cd CobbleStoneEnergy_Assignment
2. Install Dependencies:
   ```bash
   pip install -r requirements.txt   
3. Usage:
   ```bash
   python main.py
