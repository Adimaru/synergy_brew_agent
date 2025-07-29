import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from datetime import datetime, timedelta

# Import functions from your modules
from src.data_generation.creator import generate_predefined_data_file, generate_custom_product_data
from src.agent.core import simulate_agent_for_n_days
from src.agent.state_manager import clear_all_state_data # Ensure this is imported for the clear button

# Import settings
from config.settings import (
    AGENT_STATE_DIR,
    DEFAULT_LEAD_TIME, DEFAULT_SAFETY_STOCK_FACTOR, DEFAULT_MIN_ORDER_QTY,
    DEFAULT_SERVICE_LEVEL, DEFAULT_NUM_DAYS_TO_SIMULATE,
    FINANCIAL_LOG_FILE, INVENTORY_LOG_FILE, PERFORMANCE_LOG_FILE,
    PRODUCT_CATALOG, STORE_LOCATIONS, PRODUCT_CONFIGS,
    SIM_DATA_DIR,
    START_DATE, END_DATE,
    BASE_DIR # <--- ADDED THIS LINE HERE
)

# Ensure necessary directories exist at startup
os.makedirs(AGENT_STATE_DIR, exist_ok=True)
os.makedirs(SIM_DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(PERFORMANCE_LOG_FILE), exist_ok=True) # Ensure LOG_DIR exists

st.set_page_config(layout="wide", page_title="Synergy Brew Smart Inventory Assistant")

st.title("🚀 Synergy Brew Smart Inventory Assistant")
st.markdown("---")
st.markdown(
    """
    Welcome to your **Smart Inventory Assistant** for Synergy Brew! 
    This tool helps you simulate how an automated system can manage coffee product stock 
    at different store locations. It forecasts future sales and recommends orders 
    to keep your shelves stocked and customers happy, while tracking your profits.
    """
)

# --- Sidebar for user inputs ---
st.sidebar.header("⚙️ Control Your Inventory Assistant")

# --- Logo Implementation ---
logo_path = os.path.join(BASE_DIR, 'images', 'synergy_brew_logo.png') # Adjust 'synergy_brew_logo.png' to your actual logo filename

if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_column_width=True)
else:
    st.sidebar.warning("Logo image not found. Please ensure 'synergy_brew_logo.png' is in the 'images' folder.")
st.sidebar.markdown("---") # Add a separator below the logo


# Product and Store Selection
st.sidebar.subheader("📍 Product & Store Selection")
all_product_keys = list(PRODUCT_CATALOG.keys())
all_store_keys = list(STORE_LOCATIONS.keys())

selected_product_key = st.sidebar.selectbox(
    "Choose a Coffee Product",
    options=all_product_keys,
    format_func=lambda x: PRODUCT_CATALOG[x],
    help="Select the specific coffee product you want to simulate inventory for."
)
selected_store_key = st.sidebar.selectbox(
    "Choose a Store Location",
    options=all_store_keys,
    format_func=lambda x: STORE_LOCATIONS[x],
    help="Select the store where you want to simulate the inventory management."
)

# Simulation Parameters
st.sidebar.subheader("🗓️ Simulation Period & Starting Stock")
num_days_to_simulate = st.sidebar.number_input(
    "How Many Days to Simulate?",
    min_value=30,
    max_value=(END_DATE - START_DATE).days + 1,
    value=DEFAULT_NUM_DAYS_TO_SIMULATE, step=30,
    help=f"Set how many days the inventory assistant should run, starting from {START_DATE.strftime('%Y-%m-%d')} up to {END_DATE.strftime('%Y-%m-%d')}."
)

initial_stock = st.sidebar.number_input(
    "Starting Inventory Level",
    min_value=0, value=100, step=50,
    help="The amount of product you have on hand at the very beginning of the simulation."
)

# Agent Policy Parameters
st.sidebar.subheader("📜 Inventory Ordering Rules")
lead_time_days = st.sidebar.slider(
    "Delivery Lead Time (Days)",
    min_value=1, max_value=7, value=DEFAULT_LEAD_TIME, step=1,
    help="The number of days it takes for a new order to arrive at the store after it's placed."
)
safety_stock_factor = st.sidebar.slider(
    "Safety Stock Buffer (Percentage)",
    min_value=0.0, max_value=1.0, value=DEFAULT_SAFETY_STOCK_FACTOR, step=0.05, format="%.0f%%",
    help="An extra buffer of stock (as a percentage of expected demand) to avoid running out during unexpected sales spikes or delays. Higher means more buffer. *(Note: This is used if 'Desired Service Level' is not set for dynamic calculation)*"
)
service_level = st.sidebar.slider(
    "Desired Service Level",
    min_value=0.80, max_value=0.99, value=DEFAULT_SERVICE_LEVEL, step=0.01, format="%.1f%%",
    help="The percentage of customer demand you aim to meet directly from stock. Higher levels reduce stockouts but increase holding costs. (e.g., 0.95 means you aim to fulfill 95% of demand instantly)."
)
min_order_quantity = st.sidebar.number_input(
    "Minimum Order Size",
    min_value=0, value=DEFAULT_MIN_ORDER_QTY, step=5,
    help="The smallest quantity of product the assistant will order at one time."
)

# Demand Simulation Parameters
st.sidebar.subheader("📈 Simulate Sales Fluctuations")
spike_probability = st.sidebar.slider(
    "Chance of a Sudden Sales Spike (Daily)",
    min_value=0.0, max_value=0.2, value=0.01, step=0.005, format="%.1f%%",
    help="The probability (chance) that a random, high-demand event occurs on any given day. E.g., a viral trend or local event."
)
spike_multiplier = st.sidebar.slider(
    "Sales Increase During a Spike",
    min_value=1.0, max_value=3.0, value=1.5, step=0.1,
    help="How much sales increase during a sudden spike. A multiplier of 1.5 means sales are 1.5 times higher."
)

# Financial Parameters
st.sidebar.subheader("💰 Calculate Your Profit")
unit_cost = st.sidebar.number_input(
    "Cost to Buy One Unit ($)",
    min_value=0.01, value=1.50, step=0.10, format="%.2f",
    help="The cost you pay to your supplier for each unit of product."
)
sales_price = st.sidebar.number_input(
    "Selling Price Per Unit ($)",
    min_value=0.01, value=4.50, step=0.10, format="%.2f",
    help="The price at which you sell each unit of product to your customers."
)

# Debug Mode Toggle
debug_mode = st.sidebar.checkbox(
    "Show Detailed Console Messages (for developers)", 
    value=False,
    help="Enabling this will print more technical details to your terminal where Streamlit is running."
)

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear All Saved Data"):
    clear_all_state_data(debug_mode=debug_mode)
    st.rerun()


# --- NEW SECTION: Data Management (Crucial for one-time generation) ---
st.subheader("🗃️ Data Management")
st.markdown("Generate or update sales history files for your products. This only needs to be done once, or when you want to reset the historical data.")

force_overwrite_data = st.checkbox("Force Overwrite All Existing Predefined Sales Data Files", value=False,
                                    help="Check this to regenerate all predefined sales data files, even if they already exist. Useful for resetting or updating parameters.")

if st.button("Generate All Predefined Sales Data", type="secondary"):
    with st.spinner("Generating sales data for all predefined products and stores... This may take a moment."):
        generated_count = 0
        for product_key in PRODUCT_CATALOG.keys():
            for store_key in STORE_LOCATIONS.keys():
                # Construct file path
                data_file_name = f"{store_key}_{product_key}_enriched_sales_history.csv"
                data_file_path = os.path.join(SIM_DATA_DIR, data_file_name)

                # Only generate if forced overwrite or file doesn't exist
                if force_overwrite_data or not os.path.exists(data_file_path):
                    try:
                        generate_predefined_data_file(
                            product_key=product_key,
                            store_key=store_key,
                            spike_probability=spike_probability, # Use UI values for generation
                            spike_multiplier=spike_multiplier,
                            force_recreate=True # Always force recreate when this button is used
                        )
                        generated_count += 1
                    except Exception as e:
                        st.error(f"Error generating data for {PRODUCT_CATALOG[product_key]} at {STORE_LOCATIONS[store_key]}: {e}")
        if generated_count > 0:
            st.success(f"Successfully generated/overwritten sales data for {generated_count} files!")
            st.info("You can now run simulations without regenerating data until you decide to overwrite again.")
        else:
            st.info("No data files needed regeneration. If you want to force it, check 'Force Overwrite Existing Data'.")

st.markdown("---")

# --- Custom Product Creation Section ---
st.subheader("🛠️ Create Your Own Coffee Product Data")
with st.expander("Click here to define a custom product and generate its sales history."):
    custom_product_name = st.text_input(
        "Give your Custom Coffee Product a Name (e.g., 'Vanilla Dream Coffee')",
        help="This name will appear in the product selection dropdown after you generate its data."
    )
    st.write("Set its typical sales patterns:")
    col1, col2, col3 = st.columns(3)
    with col1:
        custom_base_sales = st.number_input("Average Daily Sales (Units)", min_value=1, value=100, help="The typical number of units sold on a regular day.")
    with col2:
        custom_weekly_peak = st.slider("Weekend Sales Boost", min_value=0.5, max_value=2.0, value=1.2, step=0.1, help="How much sales increase on weekends compared to weekdays (1.0 means no change, 1.5 means 50% more).")
    with col3:
        custom_summer_factor = st.slider("Summer Sales Change", min_value=0.5, max_value=2.0, value=1.0, step=0.1, help="How sales change during summer months (June-August). For example, 0.8 means 20% less sales.")
    
    col4, col5 = st.columns(2)
    with col4:
        custom_winter_factor = st.slider("Winter Sales Change", min_value=0.5, max_value=2.0, value=1.1, step=0.1, help="How sales change during winter months (Dec-Feb). For example, 1.1 means 10% more sales.")
    with col5:
        custom_store_key = st.selectbox(
            "Generate Data for Which Store?",
            options=all_store_keys,
            format_func=lambda x: STORE_LOCATIONS[x],
            key="custom_product_store_select",
            help="Choose the store location for which this custom product's sales data will be generated."
        )
    
    if st.button("Generate Custom Product Sales Data"):
        if custom_product_name:
            with st.spinner(f"Creating sales history for '{custom_product_name}'..."):
                custom_file_path = generate_custom_product_data(
                    custom_product_name, custom_base_sales, custom_weekly_peak,
                    custom_summer_factor, custom_winter_factor, custom_store_key,
                    spike_probability=spike_probability, 
                    spike_multiplier=spike_multiplier
                )
            # After generating custom data, update the PRODUCT_CATALOG and PRODUCT_CONFIGS
            custom_product_id = f"custom_product_{custom_product_name.replace(' ', '_').lower()}"
            PRODUCT_CATALOG[custom_product_id] = custom_product_name
            PRODUCT_CONFIGS[custom_product_id] = {
                "name": custom_product_name,
                "base_sales": custom_base_sales,
                "weekly_peak_factor": custom_weekly_peak,
                "summer_factor": custom_summer_factor,
                "winter_factor": custom_winter_factor,
                # Add other custom product specific configs here if needed
            }
            st.success(f"Success! Data for '{custom_product_name}' at {STORE_LOCATIONS[custom_store_key]} is ready. You can now select it in 'Choose a Coffee Product' above!")
            st.rerun() # Rerun to update the selectbox options
        else:
            st.warning("Please enter a Custom Coffee Product Name before generating data.")


# --- Main Simulation Button ---
st.subheader("▶️ Run the Inventory Simulation")
if st.button("Start Inventory Simulation", type="primary"):
    data_file_name = f"{selected_store_key}_{selected_product_key}_enriched_sales_history.csv"
    data_file_path = os.path.join(SIM_DATA_DIR, data_file_name)

    # Check if the data file exists BEFORE running simulation
    if not os.path.exists(data_file_path):
        st.error(f"Sales history for {PRODUCT_CATALOG.get(selected_product_key, selected_product_key)} at {STORE_LOCATIONS[selected_store_key]} not found.")
        st.warning("Please generate the data first using the 'Data Management' or 'Custom Product Data' sections above.")
        st.stop() # Stop execution if data is missing
            
    with st.spinner("The Smart Inventory Assistant is simulating days... Please wait."):
        simulate_agent_for_n_days(
            num_days=num_days_to_simulate,
            product_key=selected_product_key,
            store_key=selected_store_key,
            data_file_path=data_file_path,
            initial_stock=initial_stock,
            debug_mode_for_run=debug_mode,
            lead_time_days=lead_time_days,
            safety_stock_factor=safety_stock_factor,
            min_order_quantity=min_order_quantity,
            service_level=service_level
        )
    st.success("Simulation Complete! Scroll down to see the results.")

# --- Load Simulation Logs ---
performance_logs = []
if os.path.exists(PERFORMANCE_LOG_FILE):
    try:
        with open(PERFORMANCE_LOG_FILE, 'r') as f:
            performance_logs = json.load(f)
    except json.JSONDecodeError:
        st.error("Error loading performance log. It might be corrupted.")
        performance_logs = []

inventory_logs = []
if os.path.exists(INVENTORY_LOG_FILE):
    try:
        with open(INVENTORY_LOG_FILE, 'r') as f:
            inventory_logs = json.load(f)
    except json.JSONDecodeError:
        st.error("Error loading inventory log. It might be corrupted.")
        inventory_logs = []

financial_logs = []
if os.path.exists(FINANCIAL_LOG_FILE):
    try:
        with open(FINANCIAL_LOG_FILE, 'r') as f:
            financial_logs = json.load(f)
    except json.JSONDecodeError:
        st.error("Error loading financial log. It might be corrupted.")
        financial_logs = []


# --- Display Simulation Results ---
st.header("📊 What Happened in the Simulation?")
st.markdown("Here's a breakdown of how the inventory assistant performed during the simulation period.")

# --- Financial Performance Metrics ---
st.subheader("💰 Your Business Performance")
total_revenue = 0
total_cost_of_products_sold = 0

if inventory_logs:
    for entry in inventory_logs:
        total_revenue += entry.get('actual_sales_today', 0) * sales_price
        total_cost_of_products_sold += entry.get('actual_sales_today', 0) * unit_cost
        
gross_profit = total_revenue - total_cost_of_products_sold

col_rev, col_cost, col_profit = st.columns(3)
with col_rev:
    st.metric("Total Sales Revenue", f"${total_revenue:,.2f}")
with col_cost:
    st.metric("Total Cost of Products Sold", f"${total_cost_of_products_sold:,.2f}")
with col_profit:
    st.metric("Gross Profit", f"${gross_profit:,.2f}")

st.info(f"*(Calculated based on a **${sales_price:.2f} Selling Price** and **${unit_cost:.2f} Unit Cost**)*")


# --- Inventory Level Over Time ---
st.subheader("📦 Inventory Levels & Orders Over Time")
if inventory_logs:
    inventory_df = pd.DataFrame(inventory_logs)
    inventory_df['date'] = pd.to_datetime(inventory_df['date'])
    inventory_df = inventory_df.sort_values('date')

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
        title=f"Daily Inventory for {PRODUCT_CATALOG.get(selected_product_key, selected_product_key)} at {STORE_LOCATIONS.get(selected_store_key, selected_store_key)}",
        xaxis_title="Date",
        yaxis_title="Units in Stock",
        hovermode="x unified",
        height=500
    )
    st.plotly_chart(fig_inv, use_container_width=True)
else:
    st.info("No inventory data available. Please run a simulation first to see this chart.")

# --- Sales Prediction Accuracy ---
st.subheader("🎯 Sales Prediction Accuracy")
if performance_logs:
    perf_df = pd.DataFrame(performance_logs)
    perf_df['forecast_date'] = pd.to_datetime(perf_df['forecast_date'])
    
    avg_mape = perf_df['mape'].mean()
    avg_mae = perf_df['mae'].mean()

    col_mape, col_mae = st.columns(2)
    with col_mape:
        st.metric(
            "Average Prediction Error (Percentage)", 
            f"{avg_mape:,.2f}%", 
            help="MAPE (Mean Absolute Percentage Error): On average, how much our sales prediction was off by, as a percentage of actual sales."
        )
    with col_mae:
        st.metric(
            "Average Prediction Error (Units)", 
            f"{avg_mae:,.2f} units", 
            help="MAE (Mean Absolute Error): On average, how many units our sales prediction was off by."
        )
    
    st.write("Comparing Daily Sales: Predicted vs. Actual")
    fig_perf = px.line(
        perf_df, x='forecast_date', y=['actual_sales', 'forecasted_sales'],
        title='Daily Actual Sales vs. Predicted Sales',
        labels={'value': 'Sales Units', 'forecast_date': 'Date'},
        color_discrete_map={'actual_sales': 'blue', 'forecasted_sales': 'red'}
    )
    fig_perf.update_layout(hovermode="x unified")
    st.plotly_chart(fig_perf, use_container_width=True)

    st.write("Recent Daily Prediction Details:")
    st.dataframe(perf_df.tail(10).set_index('forecast_date'))
else:
    st.info("No sales prediction data available. Please run a simulation first to see this analysis.")


# --- Financial Performance ---
st.subheader("💰 Financial Performance")
if financial_logs:
    financial_df = pd.DataFrame(financial_logs)
    financial_df['date'] = pd.to_datetime(financial_df['date'])
    financial_df = financial_df.sort_values('date')

    total_holding_cost = financial_df['holding_cost'].sum()
    total_ordering_cost = financial_df['ordering_cost'].sum()
    total_stockout_cost = financial_df['stockout_cost'].sum()
    total_overall_cost = financial_df['total_daily_cost'].sum()

    st.markdown(f"**Total Holding Cost:** ${total_holding_cost:,.2f}")
    st.markdown(f"**Total Ordering Cost:** ${total_ordering_cost:,.2f}")
    st.markdown(f"**Total Stockout Cost:** ${total_stockout_cost:,.2f}")
    st.markdown(f"**Total Overall Cost:** ${total_overall_cost:,.2f}")

    st.markdown("---") # Separator

    # Daily Cost Breakdown Chart
    st.write("Daily Cost Breakdown")
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
    st.plotly_chart(fig_costs, use_container_width=True)

else:
    st.info("No financial data available. Please run a simulation first to see this section.")