# src/analysis/financial.py

import pandas as pd
import numpy as np
import logging
from config.settings import (
    PRODUCT_PRICE_PER_UNIT,
    HOLDING_COST_PER_UNIT_PER_DAY,
    ORDERING_COST_PER_ORDER,
    STOCKOUT_COST_PER_UNIT_LOST_SALE
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def calculate_kpis(financial_log_df: pd.DataFrame, inventory_log_df: pd.DataFrame, lost_sales_total: float, total_sales: float, num_days: int) -> dict:
    """
    Calculates key performance indicators (KPIs) from the simulation logs.
    
    Args:
        financial_log_df (pd.DataFrame): DataFrame containing daily financial records.
        inventory_log_df (pd.DataFrame): DataFrame containing daily inventory records.
        lost_sales_total (float): The total number of lost sales units.
        total_sales (float): The total number of units sold.
        num_days (int): The number of days the simulation ran.
    
    Returns:
        dict: A dictionary of calculated KPIs.
    """
    if financial_log_df.empty or inventory_log_df.empty:
        return {}

    # Total Costs
    total_holding_cost = financial_log_df['holding_cost'].sum()
    total_ordering_cost = financial_log_df['ordering_cost'].sum()
    total_stockout_cost = financial_log_df['stockout_cost'].sum()
    total_operational_cost = total_holding_cost + total_ordering_cost + total_stockout_cost

    # Revenue & Profit
    total_revenue = total_sales * PRODUCT_PRICE_PER_UNIT
    total_profit = total_revenue - total_operational_cost

    # Service Level & Inventory Metrics
    total_demand = total_sales + lost_sales_total
    service_level_fill_rate = total_sales / total_demand if total_demand > 0 else 0
    average_inventory = inventory_log_df['ending_stock'].mean()
    
    # Orders
    total_orders_placed = (inventory_log_df['order_placed'] > 0).sum()

    kpis = {
        'total_revenue': total_revenue,
        'total_operational_cost': total_operational_cost,
        'total_profit': total_profit,
        'total_holding_cost': total_holding_cost,
        'total_ordering_cost': total_ordering_cost,
        'total_stockout_cost': total_stockout_cost,
        'total_sales_units': total_sales,
        'total_lost_sales_units': lost_sales_total,
        'service_level_fill_rate': service_level_fill_rate,
        'average_inventory': average_inventory,
        'total_orders_placed': total_orders_placed
    }

    return kpis

def format_kpis(kpis: dict) -> dict:
    """
    Formats the KPI values for better display (e.g., currency, percentage).
    
    Args:
        kpis (dict): The dictionary of raw KPI values.
        
    Returns:
        dict: A dictionary with formatted KPI strings.
    """
    if not kpis:
        return {}
        
    formatted_kpis = {
        'Total Revenue': f"${kpis['total_revenue']:.2f}",
        'Total Operational Cost': f"${kpis['total_operational_cost']:.2f}",
        'Total Profit': f"${kpis['total_profit']:.2f}",
        '---': '---',
        'Total Holding Cost': f"${kpis['total_holding_cost']:.2f}",
        'Total Ordering Cost': f"${kpis['total_ordering_cost']:.2f}",
        'Total Stockout Cost': f"${kpis['total_stockout_cost']:.2f}",
        '----': '----',
        'Units Sold': f"{int(kpis['total_sales_units'])}",
        'Units of Lost Sales': f"{int(kpis['total_lost_sales_units'])}",
        'Service Level (Fill Rate)': f"{kpis['service_level_fill_rate']:.2%}",
        'Average Daily Inventory': f"{kpis['average_inventory']:.2f} units",
        'Total Orders Placed': f"{kpis['total_orders_placed']} orders"
    }
    
    return formatted_kpis