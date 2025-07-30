# src/utils/plot_generator.py
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def plot_inventory_levels(inventory_df, selected_product_name, selected_store_name):
    """
    Generates a Plotly figure for daily inventory levels, orders, and deliveries.

    Args:
        inventory_df (pd.DataFrame): DataFrame containing inventory log data.
        selected_product_name (str): The display name of the selected product.
        selected_store_name (str): The display name of the selected store.

    Returns:
        go.Figure: Plotly figure object.
    """
    fig_inv = go.Figure()

    # Ending Stock Line
    fig_inv.add_trace(go.Scatter(
        x=inventory_df['date'],
        y=inventory_df['ending_stock'],
        mode='lines+markers',
        name='Ending Stock Level',
        line=dict(color='orange', width=2),
        marker=dict(size=4, symbol='circle')
    ))

    if 'calculated_safety_stock' in inventory_df.columns:
        fig_inv.add_trace(go.Scatter(
            x=inventory_df['date'],
            y=inventory_df['calculated_safety_stock'],
            mode='lines',
            name='Calculated Safety Stock',
            line=dict(color='purple', width=1, dash='dot'),
            hovertemplate="<b>Date:</b> %{x}<br><b>Safety Stock:</b> %{y:.0f} units<extra></extra>"
        ))

    # Orders Placed Markers
    orders_df = inventory_df[inventory_df['order_placed_qty'] > 0]
    if not orders_df.empty:
        fig_inv.add_trace(go.Scatter(
            x=orders_df['date'],
            y=orders_df['ending_stock'],
            mode='markers',
            name='Orders Placed',
            marker=dict(
                symbol='triangle-up',
                size=10,
                color='green',
                line=dict(width=1, color='DarkSlateGrey')
            ),
            text=[f"Order: {q} units" for q in orders_df['order_placed_qty']],
            hovertemplate="<b>Date:</b> %{x}<br><b>Stock:</b> %{y}<br><b>Order Placed:</b> %{text}<extra></extra>"
        ))

    # Deliveries Received Markers
    deliveries_df = inventory_df[inventory_df['deliveries_today'] > 0]
    if not deliveries_df.empty:
        fig_inv.add_trace(go.Scatter(
            x=deliveries_df['date'],
            y=deliveries_df['ending_stock'],
            mode='markers',
            name='Deliveries Received',
            marker=dict(
                symbol='star',
                size=10,
                color='blue',
                line=dict(width=1, color='DarkSlateGrey')
            ),
            text=[f"Delivery: {q} units" for q in deliveries_df['deliveries_today']],
            hovertemplate="<b>Date:</b> %{x}<br><b>Stock:</b> %{y}<br><b>Delivery Received:</b> %{text}<extra></extra>"
        ))

    fig_inv.update_layout(
        title=f"Daily Inventory for {selected_product_name} at {selected_store_name}",
        xaxis_title="Date",
        yaxis_title="Units in Stock",
        hovermode="x unified",
        height=500
    )
    return fig_inv

def plot_sales_prediction(perf_df):
    """
    Generates a Plotly Express line chart for actual vs. forecasted sales.

    Args:
        perf_df (pd.DataFrame): DataFrame containing performance log data.

    Returns:
        go.Figure: Plotly figure object.
    """
    fig_perf = px.line(
        perf_df, x='forecast_date', y=['actual_sales', 'forecasted_sales'],
        title='Daily Actual Sales vs. Predicted Sales',
        labels={'value': 'Sales Units', 'forecast_date': 'Date'},
        color_discrete_map={'actual_sales': 'blue', 'forecasted_sales': 'red'}
    )
    fig_perf.update_layout(hovermode="x unified")
    return fig_perf

def plot_daily_cost_breakdown(financial_df):
    """
    Generates a Plotly Express area chart for daily cost breakdown.

    Args:
        financial_df (pd.DataFrame): DataFrame containing financial log data.

    Returns:
        go.Figure: Plotly figure object.
    """
    fig_costs = px.area(financial_df, 
                        x='date', 
                        y=['holding_cost', 'ordering_cost', 'stockout_cost'],
                        title='Daily Cost Breakdown Over Time',
                        labels={'value': 'Cost ($)', 'variable': 'Cost Type'},
                        color_discrete_map={
                            'holding_cost': 'lightgreen',
                            'ordering_cost': 'skyblue',
                            'stockout_cost': 'salmon'
                        })
    fig_costs.update_layout(hovermode="x unified")
    return fig_costs