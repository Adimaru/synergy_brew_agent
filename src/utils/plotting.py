# src/analysis/plotting.py

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from config.settings import PRODUCT_PRICE_PER_UNIT

def plot_inventory_levels(inventory_log_df: pd.DataFrame):
    """
    Generates a Plotly chart of inventory levels over time.
    
    Args:
        inventory_log_df (pd.DataFrame): DataFrame with daily inventory data.
    
    Returns:
        plotly.graph_objects.Figure: The Plotly figure object.
    """
    df = inventory_log_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'], 
        y=df['ending_stock'], 
        mode='lines+markers', 
        name='Ending Stock',
        line=dict(color='deepskyblue')
    ))

    fig.add_trace(go.Scatter(
        x=df['date'], 
        y=df['pending_orders'], 
        mode='lines', 
        name='Pending Orders',
        line=dict(color='lightcoral')
    ))

    fig.update_layout(
        title_text="Inventory Levels and Pending Orders Over Time",
        xaxis_title="Date",
        yaxis_title="Units",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

def plot_daily_costs(financial_log_df: pd.DataFrame):
    """
    Generates a stacked bar chart of daily costs over time.
    
    Args:
        financial_log_df (pd.DataFrame): DataFrame with daily financial data.
    
    Returns:
        plotly.graph_objects.Figure: The Plotly figure object.
    """
    df = financial_log_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate cumulative costs for a line chart
    df['cumulative_total_cost'] = df['total_daily_cost'].cumsum()

    # Create the stacked bar chart for daily costs
    fig_bars = go.Figure(data=[
        go.Bar(name='Holding Cost', x=df['date'], y=df['holding_cost'], marker_color='cadetblue'),
        go.Bar(name='Ordering Cost', x=df['date'], y=df['ordering_cost'], marker_color='skyblue'),
        go.Bar(name='Stockout Cost', x=df['date'], y=df['stockout_cost'], marker_color='indianred')
    ])

    # Add the cumulative cost line chart
    fig_bars.add_trace(go.Scatter(
        x=df['date'], 
        y=df['cumulative_total_cost'], 
        mode='lines', 
        name='Cumulative Cost',
        yaxis='y2', # Use a secondary y-axis
        line=dict(color='darkslateblue', width=3)
    ))

    # Update layout with a secondary y-axis for the cumulative cost line
    fig_bars.update_layout(
        barmode='stack', 
        title_text="Daily and Cumulative Operational Costs",
        xaxis_title="Date",
        yaxis=dict(title="Daily Cost ($)", titlefont=dict(color="cadetblue")),
        yaxis2=dict(title="Cumulative Cost ($)", titlefont=dict(color="darkslateblue"), overlaying='y', side='right'),
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig_bars

def plot_sales_and_lost_sales(inventory_log_df: pd.DataFrame):
    """
    Generates a stacked bar chart of daily sales and lost sales.
    
    Args:
        inventory_log_df (pd.DataFrame): DataFrame with daily inventory data.
    
    Returns:
        plotly.graph_objects.Figure: The Plotly figure object.
    """
    df = inventory_log_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    fig = go.Figure(data=[
        go.Bar(name='Actual Sales', x=df['date'], y=df['actual_sales_today'], marker_color='forestgreen'),
        go.Bar(name='Lost Sales', x=df['date'], y=df['lost_sales_today'], marker_color='firebrick')
    ])

    fig.update_layout(
        barmode='stack',
        title_text="Daily Sales and Lost Sales",
        xaxis_title="Date",
        yaxis_title="Units",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

def plot_forecast_vs_actual(inventory_log_df: pd.DataFrame):
    """
    Generates a Plotly chart comparing lead time forecast with actual demand.
    
    Args:
        inventory_log_df (pd.DataFrame): DataFrame with daily inventory data.
    
    Returns:
        plotly.graph_objects.Figure: The Plotly figure object.
    """
    df = inventory_log_df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Calculate actual demand over the lead time period
    # This requires a rolling sum of actual sales
    df['actual_demand_over_lt'] = df['actual_sales_today'].rolling(window=df['lead_time_forecast'].iloc[0], min_periods=1).sum()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['date'], 
        y=df['lead_time_forecast'], 
        mode='lines', 
        name='Forecasted Demand Over Lead Time',
        line=dict(color='darkorange', dash='dot')
    ))

    fig.add_trace(go.Scatter(
        x=df['date'], 
        y=df['actual_demand_over_lt'], 
        mode='lines', 
        name='Actual Demand Over Lead Time',
        line=dict(color='royalblue')
    ))

    fig.update_layout(
        title_text="Forecasted vs. Actual Demand Over Lead Time",
        xaxis_title="Date",
        yaxis_title="Units",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig