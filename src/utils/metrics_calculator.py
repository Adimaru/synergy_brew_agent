# src/utils/metrics_calculator.py
import pandas as pd

def calculate_financial_metrics(inventory_logs, financial_logs, sales_price, unit_cost):
    """
    Calculates key financial metrics from logs.

    Args:
        inventory_logs (list): List of inventory log entries.
        financial_logs (list): List of financial log entries.
        sales_price (float): Selling price per unit.
        unit_cost (float): Cost to buy one unit.

    Returns:
        tuple: (total_revenue, total_cost_of_products_sold, gross_profit,
                total_holding_cost, total_ordering_cost, total_stockout_cost, total_overall_cost)
    """
    total_revenue = 0
    total_cost_of_products_sold = 0
    if inventory_logs:
        for entry in inventory_logs:
            total_revenue += entry.get('actual_sales_today', 0) * sales_price
            total_cost_of_products_sold += entry.get('actual_sales_today', 0) * unit_cost
    gross_profit = total_revenue - total_cost_of_products_sold

    total_holding_cost = 0
    total_ordering_cost = 0
    total_stockout_cost = 0
    total_overall_cost = 0
    if financial_logs:
        financial_df = pd.DataFrame(financial_logs)
        total_holding_cost = financial_df['holding_cost'].sum()
        total_ordering_cost = financial_df['ordering_cost'].sum()
        total_stockout_cost = financial_df['stockout_cost'].sum()
        total_overall_cost = financial_df['total_daily_cost'].sum()

    return (total_revenue, total_cost_of_products_sold, gross_profit,
            total_holding_cost, total_ordering_cost, total_stockout_cost, total_overall_cost)

def calculate_prediction_accuracy_metrics(performance_logs):
    """
    Calculates sales prediction accuracy metrics (MAPE, MAE).

    Args:
        performance_logs (list): List of performance log entries.

    Returns:
        tuple: (avg_mape, avg_mae) or (0, 0) if no data.
    """
    if performance_logs:
        perf_df = pd.DataFrame(performance_logs)
        # Ensure 'mape' and 'mae' columns exist before trying to mean them
        avg_mape = perf_df['mape'].mean() if 'mape' in perf_df.columns else 0
        avg_mae = perf_df['mae'].mean() if 'mae' in perf_df.columns else 0
        return avg_mape, avg_mae
    return 0, 0