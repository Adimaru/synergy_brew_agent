# app.py
import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import plotly.express as px

# Import settings (ensure BASE_DIR is available here)
from config.settings import (
    AGENT_STATE_DIR,
    DEFAULT_LEAD_TIME, DEFAULT_SAFETY_STOCK_FACTOR, DEFAULT_MIN_ORDER_QTY,
    DEFAULT_SERVICE_LEVEL, DEFAULT_NUM_DAYS_TO_SIMULATE,
    FINANCIAL_LOG_FILE, INVENTORY_LOG_FILE, PERFORMANCE_LOG_FILE,
    PRODUCT_CATALOG, STORE_LOCATIONS, PRODUCT_CONFIGS, # These are base/predefined
    SIM_DATA_DIR,
    START_DATE, END_DATE,
    BASE_DIR,
    CUSTOM_PRODUCTS_FILE, CUSTOM_STORES_FILE,
    DEFAULT_FORECAST_HORIZON_DAYS # ADD THIS LINE
)
# Streamlit Page Configuration must be the first Streamlit command
st.set_page_config(layout="wide", page_title="Synergy Brew Smart Inventory Assistant")

# --- CSS Styling Injection ---
def load_css(file_path):
    try:
        with open(file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found at {file_path}. Custom styling may not apply.")
    except Exception as e:
        st.error(f"Error loading CSS file: {e}")

# Load the CSS file AFTER st.set_page_config
css_file_path = os.path.join(BASE_DIR, '.streamlit', 'style.css')
load_css(css_file_path)

# Import other functions from your modules
from src.data_generation.creator import generate_predefined_data_file, generate_custom_product_data
from src.agent.core import simulate_agent_for_n_days
from src.agent.state_manager import clear_all_state_data 
from src.agent.forecasting import calculate_cv_metrics # NEW: Import for Prophet CV

# Import new utility modules
from src.utils.plot_generator import plot_inventory_levels, plot_sales_prediction, plot_daily_cost_breakdown
from src.utils.metrics_calculator import calculate_financial_metrics, calculate_prediction_accuracy_metrics


# Ensure necessary directories exist at startup
os.makedirs(AGENT_STATE_DIR, exist_ok=True)
os.makedirs(SIM_DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(PERFORMANCE_LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(CUSTOM_PRODUCTS_FILE), exist_ok=True) # Ensure config dir exists for custom files

# --- Load Custom Configurations at Startup and Merge with Predefined ---
# Moved load_custom_products and load_custom_stores to be functions inside config_manager.py
# Assuming these functions are available:
from src.data_manager.config_manager import load_custom_products, save_custom_products, load_custom_stores, save_custom_stores

CUSTOM_PRODUCT_CATALOG = load_custom_products()
CUSTOM_STORE_LOCATIONS = load_custom_stores()

PRODUCT_CATALOG_ALL = PRODUCT_CATALOG.copy()
PRODUCT_CATALOG_ALL.update(CUSTOM_PRODUCT_CATALOG)

STORE_LOCATIONS_ALL = STORE_LOCATIONS.copy()
STORE_LOCATIONS_ALL.update(CUSTOM_STORE_LOCATIONS)

logo_path = os.path.join(BASE_DIR, 'images', 'synergy_brew_logo.png')

if os.path.exists(logo_path):
    # Display logo in sidebar
    st.sidebar.image(logo_path, use_container_width=True) # Logo size controlled by CSS
else:
    st.sidebar.warning("Logo image not found. Please ensure 'synergy_brew_logo.png' is in the 'images' folder.")

# Product and Store Selection - KEEP THESE IN SIDEBAR as they are global selectors
st.sidebar.subheader("PRODUCT & STORE") # Shorter title
all_product_keys = list(PRODUCT_CATALOG_ALL.keys())
all_store_keys = list(STORE_LOCATIONS_ALL.keys())

selected_product_key = st.sidebar.selectbox(
    "Choose a Coffee Product",
    options=all_product_keys,
    format_func=lambda x: PRODUCT_CATALOG_ALL[x],
    help="Select the specific coffee product you want to simulate inventory for."
)
selected_store_key = st.sidebar.selectbox(
    "Choose a Store Location",
    options=all_store_keys,
    format_func=lambda x: STORE_LOCATIONS_ALL[x],
    help="Select the store where you want to simulate the inventory management."
)

# Simulation Parameters - KEEP THESE IN SIDEBAR
st.sidebar.subheader("SIMULATION & STOCK") # Shorter title
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

# --- NEW: Demand Forecasting Parameters ---
st.sidebar.subheader("DEMAND FORECASTING")
forecasting_model = st.sidebar.selectbox(
    "Select Forecasting Model",
    options=["Prophet", "Moving Average", "Actual Sales Data (Baseline)"], # UPDATED: Added Prophet
    help="Choose the model used to predict future sales."
)

moving_average_window = 0 # Initialize outside of conditional scope
if forecasting_model == "Moving Average":
    moving_average_window = st.sidebar.number_input(
        "Moving Average Window (Days)",
        min_value=1, max_value=60, value=7, step=1,
        help="Number of past days' sales to average for forecasting. Recommended: 7 or 30."
    )
st.sidebar.markdown("---") # Separator

# Agent Policy Parameters - KEEP THESE IN SIDEBAR
st.sidebar.subheader("ORDERING RULES") # Shorter title
lead_time_days = st.sidebar.slider(
    "Delivery Lead Time (Days)",
    min_value=1, max_value=7, value=DEFAULT_LEAD_TIME, step=1,
    help="The number of days it takes for a new order to arrive at the store after it's placed."
)
# We will derive safety stock from service level and demand variability
# The safety_stock_factor can be removed or used for a fixed safety stock if service_level isn't preferred.
# For now, let's keep it but indicate service_level is primary.
safety_stock_factor = st.sidebar.slider(
    "Safety Stock Buffer (Percentage)",
    min_value=0.0, max_value=1.0, value=DEFAULT_SAFETY_STOCK_FACTOR, step=0.05, format="%.0f%%",
    help="An extra buffer of stock (as a percentage of expected demand) to avoid running out during unexpected sales spikes or delays. *(This will be used if 'Desired Service Level' is not active for dynamic safety stock calculation, e.g., if error std dev is zero or insufficient history.)*"
)
service_level = st.sidebar.slider(
    "Desired Service Level",
    min_value=0.80, max_value=0.99, value=DEFAULT_SERVICE_LEVEL, step=0.01, format="%.1f%%",
    help="The percentage of customer demand you aim to meet directly from stock. Higher levels reduce stockouts but increase holding costs. (e.g., 0.95 means you aim to fulfill 95% of demand instantly). This drives dynamic safety stock."
)
min_order_quantity = st.sidebar.number_input(
    "Minimum Order Size",
    min_value=0, value=DEFAULT_MIN_ORDER_QTY, step=5,
    help="The smallest quantity of product the assistant will order at one time. The system will order at least this much."
)
# We might add Economic Order Quantity (EOQ) parameters here later if we go for more complex ROQ.

# Demand Simulation Parameters - KEEP THESE IN SIDEBAR
st.sidebar.subheader("SALES FLUCTUATIONS") # Shorter title
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

# Financial Parameters - KEEP THESE IN SIDEBAR
st.sidebar.subheader("PROFIT CALCULATION") # Shorter title
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

# Debug Mode Toggle - KEEP THIS IN SIDEBAR
debug_mode = st.sidebar.checkbox(
    "Show Detailed Console Messages (for developers)", 
    value=False,
    help="Enabling this will print more technical details to your terminal where Streamlit is running."
)

st.sidebar.markdown("---")
if st.sidebar.button("Clear All Saved Data"):
    clear_all_state_data(debug_mode=debug_mode)
    # Clear custom configs as well
    if os.path.exists(CUSTOM_PRODUCTS_FILE):
        os.remove(CUSTOM_PRODUCTS_FILE)
    if os.path.exists(CUSTOM_STORES_FILE):
        os.remove(CUSTOM_STORES_FILE)
    st.rerun()

# --- Helper function to load all sales data for CV or plotting ---
@st.cache_data
def load_all_enriched_sales_data_for_cv(product_key, store_key):
    """Loads all enriched sales data for a specific product/store for CV calculation."""
    data_file_name = f"{store_key}_{product_key}_enriched_sales_history.csv"
    data_file_path = os.path.join(SIM_DATA_DIR, data_file_name)
    
    if not os.path.exists(data_file_path):
        st.error(f"Required data file for Prophet CV not found: {data_file_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(data_file_path)
    df['ds'] = pd.to_datetime(df['ds'])
    if df['ds'].dt.tz is not None:
        df['ds'] = df['ds'].dt.tz_localize(None) # Ensure timezone-naive for Prophet
    return df.sort_values('ds')

# --- Helper function for Prophet CV display ---
def display_prophet_cv_metrics(sales_data_for_cv_analysis, debug_mode):
    st.subheader("Prophet Cross-Validation Metrics")

    if sales_data_for_cv_analysis.empty:
        st.info("No sales data loaded for CV.")
        return
    
    # Debugging lines you might have added (can keep or remove)
    st.write(f"Sales data for CV analysis has {len(sales_data_for_cv_analysis)} rows.")

    if len(sales_data_for_cv_analysis) < 60:
        st.info("Insufficient historical sales data (less than 60 days) to perform meaningful Prophet cross-validation.")
        return
    
    # Dynamic calculation of CV parameters
    # Ensuring minimums for robustness with small datasets
    horizon_days_str = f"{DEFAULT_FORECAST_HORIZON_DAYS} days" 
    initial_train_days_val = f"{max(30, int(len(sales_data_for_cv_analysis) * 0.7))} days" # Renamed to avoid confusion with parameter name
    period_between_cv_val = f"{max(7, int(len(sales_data_for_cv_analysis) * 0.1))} days"   # Renamed to avoid confusion with parameter name

    st.info(f"Running Prophet cross-validation with: Initial training = {initial_train_days_val}, Retrain period = {period_between_cv_val}, Horizon = {horizon_days_str}")
    
    # Debugging lines you might have added (can keep or remove)
    st.write(f"CV Parameters: initial={initial_train_days_val}, period={period_between_cv_val}, horizon={horizon_days_str}")

    try:
        cv_metrics_df = calculate_cv_metrics(
            df=sales_data_for_cv_analysis,
            initial_train_days=initial_train_days_val,  # <--- CORRECTED PARAMETER NAME
            period_between_cv=period_between_cv_val,    # <--- CORRECTED PARAMETER NAME
            horizon_forecast_days=horizon_days_str,     # <--- CORRECTED PARAMETER NAME
            debug_mode=debug_mode
        )

        if not cv_metrics_df.empty:
            st.write("Cross-Validation Performance Metrics:")
            st.dataframe(cv_metrics_df.head())

            # Plotting MAE over the horizon
            fig_mae = px.line(cv_metrics_df, x='horizon', y='mae', title='Mean Absolute Error (MAE) over Forecast Horizon')
            fig_mae.update_layout(xaxis_title="Forecast Horizon (Days)", yaxis_title="MAE")
            st.plotly_chart(fig_mae)

            # Plotting MAPE over the horizon
            fig_mape = px.line(cv_metrics_df, x='horizon', y='mape', title='Mean Absolute Percentage Error (MAPE) over Forecast Horizon')
            fig_mape.update_layout(xaxis_title="Forecast Horizon (Days)", yaxis_title="MAPE")
            st.plotly_chart(fig_mape)
        else:
            st.warning("No cross-validation metrics could be calculated. Please check the data and logs.")

    except Exception as e:
        st.error(f"An error occurred while displaying Prophet CV metrics: {e}")
        if debug_mode:
            st.exception(e)


# --- Main Content Area with Tabs (now on top) ---
tab1, tab2, tab3 = st.tabs(["Welcome & Setup", "Data Management & Customization", "Simulation Results"])

with tab1:
    st.header("Welcome to Synergy Brew Inventory") # CHANGED: Shorter title
    st.markdown(
        """
        This powerful tool helps you simulate and analyze inventory management strategies for your coffee business.
        Use the sidebar to configure your simulation parameters, and navigate through the tabs to manage your data,
        run simulations, and view detailed performance reports.
        """
    )
    st.markdown("---")
    st.subheader("Quick Start Guide:")
    st.markdown(
        """
        1.  **Select Product & Store** in the sidebar.
        2.  Go to the "**Data Management & Customization**" tab to generate or update sales data and manage custom products/stores.
        3.  Adjust **Simulation Period & Stock**, and **Inventory Ordering Rules** in the sidebar.
        4.  Come back to this tab and click "**Start Inventory Simulation**" below.
        5.  View detailed **Simulation Results** in the dedicated tab!
        """
    )
    st.markdown("---")
    # This section remains here as the main simulation trigger for simplicity
    st.subheader("Run the Inventory Simulation")
    if st.button("Start Inventory Simulation", type="primary"):
        data_file_name = f"{selected_store_key}_{selected_product_key}_enriched_sales_history.csv"
        data_file_path = os.path.join(SIM_DATA_DIR, data_file_name)

        # Check if the data file exists BEFORE running simulation
        if not os.path.exists(data_file_path):
            st.error(f"Sales history for {PRODUCT_CATALOG_ALL.get(selected_product_key, selected_product_key)} at {STORE_LOCATIONS_ALL[selected_store_key]} not found.")
            st.warning("Please generate the data first using the 'Data Management & Customization' tab.")
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
                safety_stock_factor=safety_stock_factor, # Keep for now, might be removed later
                min_order_quantity=min_order_quantity,
                service_level=service_level,
                forecasting_model=forecasting_model,        # NEW: Pass forecasting model
                moving_average_window=moving_average_window # NEW: Pass MA window
            )
        st.success("Simulation Complete! Navigate to the 'Simulation Results' tab to see the detailed breakdown.")

        # After successful simulation, ensure logs are reloaded and stored in session_state
        # These are reloaded directly from files to ensure the latest data is displayed
        st.session_state.performance_logs = []
        if os.path.exists(PERFORMANCE_LOG_FILE):
            with open(PERFORMANCE_LOG_FILE, 'r') as f:
                st.session_state.performance_logs = json.load(f)

        st.session_state.inventory_logs = []
        if os.path.exists(INVENTORY_LOG_FILE):
            with open(INVENTORY_LOG_FILE, 'r') as f:
                st.session_state.inventory_logs = json.load(f)

        st.session_state.financial_logs = []
        if os.path.exists(FINANCIAL_LOG_FILE):
            with open(FINANCIAL_LOG_FILE, 'r') as f:
                st.session_state.financial_logs = json.load(f)


with tab2: # Data Management & Customization Tab
    st.header("Data Management & Customization")
    st.markdown("Manage and generate sales history files. You can generate predefined product data or create your own custom products.")

    st.subheader("Predefined Product Data Generation")
    st.markdown("Generate or update sales history files for your standard products. This only needs to be done once, or when you want to reset the historical data based on current simulation parameters.")

    force_overwrite_data = st.checkbox("Force Overwrite All Existing Predefined Sales Data Files", value=False,
                                        help="Check this to regenerate all predefined sales data files, even if they already exist. Useful for resetting or updating parameters.")

    if st.button("Generate All Predefined Sales Data", type="secondary"):
        with st.spinner("Generating sales data for all predefined products and stores... This may take a moment."):
            generated_count = 0
            for product_key in PRODUCT_CATALOG.keys(): # Iterate over original predefined catalog
                for store_key in STORE_LOCATIONS_ALL.keys(): # Iterate over all stores (predefined + custom)
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
                            st.error(f"Error generating data for {PRODUCT_CATALOG.get(product_key, product_key)} at {STORE_LOCATIONS_ALL.get(store_key, store_key)}: {e}")
            if generated_count > 0:
                st.success(f"Successfully generated/overwritten sales data for {generated_count} files!")
                st.info("You can now run simulations without regenerating data until you decide to overwrite again.")
            else:
                st.info("No data files needed regeneration. If you want to force it, check 'Force Overwrite Existing Data'.")

    st.markdown("---")

    st.subheader("Create Your Own Coffee Product Data")
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
            custom_store_key_for_gen = st.selectbox( # Renamed to avoid key clash
                "Generate Data for Which Store?",
                options=all_store_keys, # Use all_store_keys to include custom stores
                format_func=lambda x: STORE_LOCATIONS_ALL[x],
                key="custom_product_store_select",
                help="Choose the store location for which this custom product's sales data will be generated."
            )
        
        if st.button("Generate Custom Product Sales Data"):
            if custom_product_name:
                custom_product_id = f"custom_product_{custom_product_name.replace(' ', '_').lower()}"
                if custom_product_id in CUSTOM_PRODUCT_CATALOG:
                    st.warning(f"Product '{custom_product_name}' already exists. Overwriting its definition and sales data.")

                with st.spinner(f"Creating sales history for '{custom_product_name}'..."):
                    custom_file_path = generate_custom_product_data(
                        custom_product_name, custom_base_sales, custom_weekly_peak,
                        custom_summer_factor, custom_winter_factor, custom_store_key_for_gen, # Use renamed var
                        spike_probability=spike_probability, 
                        spike_multiplier=spike_multiplier
                    )
                
                # Update custom product catalog and save it
                CUSTOM_PRODUCT_CATALOG[custom_product_id] = custom_product_name
                save_custom_products(CUSTOM_PRODUCT_CATALOG) # SAVE TO FILE

                st.success(f"Success! Data for '{custom_product_name}' at {STORE_LOCATIONS_ALL[custom_store_key_for_gen]} is ready. Product saved for future use!")
                st.rerun() 
            else:
                st.warning("Please enter a Custom Coffee Product Name before generating data.")
    
    st.markdown("---")
    st.subheader("Manage Your Store Locations")
    st.markdown("Add new store locations to expand your inventory management to new geographies.")

    with st.expander("Click here to add a new custom store location."):
        new_store_name = st.text_input(
            "Enter New Store Location Name (e.g., 'Downtown Branch')",
            key="new_store_name_input",
            help="This name will appear in the store selection dropdown after you add it."
        )
        if st.button("Add New Store Location"):
            if new_store_name:
                new_store_key = f"custom_store_{new_store_name.replace(' ', '_').lower()}"
                if new_store_key in CUSTOM_STORE_LOCATIONS:
                    st.warning(f"Store '{new_store_name}' already exists.")
                else:
                    CUSTOM_STORE_LOCATIONS[new_store_key] = new_store_name
                    save_custom_stores(CUSTOM_STORE_LOCATIONS)
                    st.success(f"Store '{new_store_name}' added successfully! It is now available in the sidebar.")
                    st.rerun()
            else:
                st.warning("Please enter a name for the new store location.")
    
    with st.expander("View and Delete Custom Stores"):
        # Filter out predefined stores from the list for deletion
        custom_only_stores = {k: v for k, v in STORE_LOCATIONS_ALL.items() if k.startswith("custom_store_")}

        if custom_only_stores:
            current_custom_stores_list = list(custom_only_stores.values())
            store_to_delete = st.selectbox(
                "Select a custom store to delete",
                options=current_custom_stores_list,
                key="delete_store_selectbox"
            )
            if st.button(f"Delete '{store_to_delete}'"):
                key_to_delete = None
                for k, v in custom_only_stores.items(): # Search only in custom_only_stores
                    if v == store_to_delete:
                        key_to_delete = k
                        break
                if key_to_delete:
                    del CUSTOM_STORE_LOCATIONS[key_to_delete] # Modify the dictionary that holds custom stores
                    save_custom_stores(CUSTOM_STORE_LOCATIONS)
                    st.success(f"Store '{store_to_delete}' deleted.")
                    st.rerun()
                else:
                    st.error("Could not find store to delete.")
        else:
            st.info("No custom stores defined yet.")

with tab3: # Simulation Results Tab
    st.header("Simulation Results")
    st.markdown("Here's a breakdown of how the inventory assistant performed during the simulation period.")

    # Load Simulation Logs from session state (or directly from files if not in session state)
    performance_logs = st.session_state.get('performance_logs', [])
    inventory_logs = st.session_state.get('inventory_logs', [])
    financial_logs = st.session_state.get('financial_logs', [])


    # --- Financial Performance Metrics ---
    st.subheader("Your Business Performance")
    if inventory_logs and financial_logs: 
        total_revenue, total_cost_of_products_sold, gross_profit, \
        total_holding_cost, total_ordering_cost, total_stockout_cost, total_overall_cost = \
            calculate_financial_metrics(inventory_logs, financial_logs, sales_price, unit_cost)

        col_rev, col_cost, col_profit = st.columns(3)
        with col_rev:
            st.metric("Total Sales Revenue", f"${total_revenue:,.2f}")
        with col_cost:
            st.metric("Total Cost of Products Sold", f"${total_cost_of_products_sold:,.2f}")
        with col_profit:
            st.metric("Gross Profit", f"${gross_profit:,.2f}")

        st.info(f"*(Calculated based on a **${sales_price:.2f} Selling Price** and **${unit_cost:.2f} Unit Cost**)*")

        st.markdown("---") # Separator for costs below
        st.markdown(f"**Total Holding Cost:** ${total_holding_cost:,.2f}")
        st.markdown(f"**Total Ordering Cost:** ${total_ordering_cost:,.2f}")
        st.markdown(f"**Total Stockout Cost:** ${total_stockout_cost:,.2f}")
        st.markdown(f"**Total Overall Cost:** ${total_overall_cost:,.2f}")

        # Daily Cost Breakdown Chart
        st.write("Daily Cost Breakdown")
        financial_df = pd.DataFrame(financial_logs)
        financial_df['date'] = pd.to_datetime(financial_df['date'])
        financial_df = financial_df.sort_values('date')
        fig_costs = plot_daily_cost_breakdown(financial_df)
        st.plotly_chart(fig_costs, use_container_width=True)

    else:
        st.info("No financial data available. Please run a simulation first to see this section.")


    # --- Inventory Level Over Time ---
    st.subheader("Inventory Levels & Orders Over Time")
    if inventory_logs:
        inventory_df = pd.DataFrame(inventory_logs)
        inventory_df['date'] = pd.to_datetime(inventory_df['date'])
        inventory_df = inventory_df.sort_values('date')

        fig_inv = plot_inventory_levels(inventory_df, 
                                        PRODUCT_CATALOG_ALL.get(selected_product_key, selected_product_key), 
                                        STORE_LOCATIONS_ALL.get(selected_store_key, selected_store_key))
        st.plotly_chart(fig_inv, use_container_width=True)
    else:
        st.info("No inventory data available. Please run a simulation first to see this chart.")

    # --- Sales Prediction Accuracy ---
    st.subheader("Sales Prediction Accuracy")
    if performance_logs:
        perf_df = pd.DataFrame(performance_logs)
        perf_df['forecast_date'] = pd.to_datetime(perf_df['forecast_date'])
        
        avg_mape, avg_mae = calculate_prediction_accuracy_metrics(performance_logs)

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
        fig_perf = plot_sales_prediction(perf_df)
        st.plotly_chart(fig_perf, use_container_width=True)

        st.write("Recent Daily Prediction Details:")
        st.dataframe(perf_df.tail(10).set_index('forecast_date'))
    else:
        st.info("No sales prediction data available. Please run a simulation first to see this analysis.")

    # --- Prophet Cross-Validation Metrics (Conditional Display) ---
    if forecasting_model == "Prophet":
        # Load the full historical sales data to run CV on
        sales_data_for_cv_analysis = load_all_enriched_sales_data_for_cv(selected_product_key, selected_store_key)
        display_prophet_cv_metrics(sales_data_for_cv_analysis, debug_mode)