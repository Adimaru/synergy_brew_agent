import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

AGENT_STATE_DIR = 'agent_state'
PERFORMANCE_LOG_FILE = os.path.join(AGENT_STATE_DIR, 'performance_log.json')

def load_state(file_path, default_value):
    """Loads state from a JSON file. Returns default_value if file not found."""
    if not os.path.exists(file_path):
        print(f"State file not found: {file_path}. Returning default.")
        return default_value
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {file_path}. Returning default.")
        return default_value

if __name__ == "__main__":
    print("--- Synergy Brew Agent: Performance Analysis ---")

    performance_data = load_state(PERFORMANCE_LOG_FILE, [])

    if not performance_data:
        print("No performance data found. Run the agent for some days first.")
        exit()

    df_perf = pd.DataFrame(performance_data)
    df_perf['forecast_date'] = pd.to_datetime(df_perf['forecast_date'])
    df_perf = df_perf.sort_values('forecast_date')

    print(f"Loaded {len(df_perf)} performance records from {df_perf['forecast_date'].min().strftime('%Y-%m-%d')} to {df_perf['forecast_date'].max().strftime('%Y-%m-%d')}.")

    # --- Visualize MAE over time ---
    plt.figure(figsize=(14, 7))
    plt.plot(df_perf['forecast_date'], df_perf['mae'], marker='o', linestyle='-', color='blue', alpha=0.7)
    plt.title('Agent Performance: Mean Absolute Error (MAE) Over Time')
    plt.xlabel('Forecast Date')
    plt.ylabel('MAE (Units)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Visualize MAPE over time ---
    plt.figure(figsize=(14, 7))
    plt.plot(df_perf['forecast_date'], df_perf['mape'], marker='o', linestyle='-', color='red', alpha=0.7)
    plt.axhline(y=10, color='green', linestyle='--', label='Good Accuracy (MAPE < 10%)')
    plt.axhline(y=20, color='orange', linestyle='--', label='Acceptable Accuracy (MAPE < 20%)')
    plt.title('Agent Performance: Mean Absolute Percentage Error (MAPE) Over Time')
    plt.xlabel('Forecast Date')
    plt.ylabel('MAPE (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Visualize Actual vs. Forecasted Sales for a recent period ---
    # This gives a qualitative view of how well forecasts align with actuals
    if len(df_perf) > 30: # Only plot if we have enough data
        recent_data = df_perf.tail(30)
        plt.figure(figsize=(14, 7))
        plt.plot(recent_data['forecast_date'], recent_data['actual_sales'], label='Actual Sales', marker='x', color='blue')
        plt.plot(recent_data['forecast_date'], recent_data['forecasted_sales'], label='Forecasted Sales', marker='o', color='red', linestyle='--')
        plt.title('Actual vs. Forecasted Sales (Last 30 Days)')
        plt.xlabel('Date')
        plt.ylabel('Sales Units')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print("\nPerformance analysis complete.")