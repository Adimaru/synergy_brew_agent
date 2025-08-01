# app.py

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import sys
import logging
import numpy as np

# Ensure src directory is in the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    DEFAULT_FORECAST_HORIZON_DAYS,
    HOLDING_COST_PER_UNIT_PER_DAY, ORDERING_COST_PER_ORDER, STOCKOUT_COST_PER_UNIT_LOST_SALE
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
# Correct the import to get the new class
from src.agent.core import InventorySimulationAgent
from src.agent.state_manager import clear_all_state_data
from src.agent.forecasting import calculate_cv_metrics, tune_prophet_hyperparameters
from src.data_manager.config_manager import load_custom_products, save_custom_products, load_custom_stores, save_custom_stores

# Import new utility modules
# Corrected import to include the new function
from src.utils.plot_generator import (
    plot_inventory_levels, plot_sales_prediction, plot_daily_cost_breakdown,
    plot_cumulative_costs, plot_cumulative_lost_sales, plot_inventory_comparison
)
from src.utils.metrics_calculator import calculate_financial_metrics, calculate_prediction_accuracy_metrics


# Ensure necessary directories exist at startup
os.makedirs(AGENT_STATE_DIR, exist_ok=True)
os.makedirs(SIM_DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(PERFORMANCE_LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(CUSTOM_PRODUCTS_FILE), exist_ok=True)

# --- Load Custom Configurations at Startup and Merge with Predefined ---
CUSTOM_PRODUCT_CATALOG = load_custom_products()
CUSTOM_STORE_LOCATIONS = load_custom_stores()

PRODUCT_CATALOG_ALL = PRODUCT_CATALOG.copy()
PRODUCT_CATALOG_ALL.update(CUSTOM_PRODUCT_CATALOG)

STORE_LOCATIONS_ALL = STORE_LOCATIONS.copy()
STORE_LOCATIONS_ALL.update(CUSTOM_STORE_LOCATIONS)

logo_path = os.path.join(BASE_DIR, 'images', 'synergy_brew_logo.png')

if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=True)
else:
    st.sidebar.warning("Logo image not found. Please ensure 'synergy_brew_logo.png' is in the 'images' folder.")

# --- NEW: Prophet Parameter Grid for Tuning ---
PROPHET_PARAM_GRID = {
    'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1],
    'changepoint_range': [0.8, 0.9],
    'seasonality_prior_scale': [1.0, 5.0, 10.0]
}

# --- NEW: Cached function for hyperparameter tuning ---
@st.cache_data
def get_tuned_prophet_params(df, initial, period, horizon, param_grid, debug_mode):
    """
    Wrapper function to cache the expensive hyperparameter tuning process.
    """
    st.info("Running Prophet hyperparameter tuning... This may take a moment.")
    with st.spinner("Tuning Prophet parameters..."):
        best_params, best_score = tune_prophet_hyperparameters(
            df=df,
            initial=initial,
            period=period,
            horizon=horizon,
            param_grid=param_grid,
            debug_mode=debug_mode
        )
    return best_params, best_score

# --- Session State Management for Comparison Results ---
if 'results_A' not in st.session_state:
    st.session_state.results_A = None
if 'results_B' not in st.session_state:
    st.session_state.results_B = None

# Product and Store Selection - KEEP THESE IN SIDEBAR as they are global selectors
st.sidebar.subheader("PRODUCT & STORE")
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
st.sidebar.subheader("GLOBAL SIMULATION SETTINGS")
num_days_to_simulate = st.sidebar.number_input(
    "How Many Days to Simulate?",
    min_value=30,
    max_value=(END_DATE - START_DATE).days + 1,
    value=DEFAULT_NUM_DAYS_TO_SIMULATE, step=30,
    help=f"Set how many days the inventory assistant should run, starting from {START_DATE.strftime('%Y-%m-%d')} up to {END_DATE.strftime('%Y-%m-%d')}."
)

# Demand Forecasting Parameters (and tuning)
st.sidebar.subheader("DEMAND FORECASTING")

# Debug Mode Toggle - KEEP THIS IN SIDEBAR
debug_mode = st.sidebar.checkbox(
    "Show Detailed Console Messages (for developers)",
    value=False,
    help="Enabling this will print more technical details to your terminal where Streamlit is running."
)
st.sidebar.markdown("---")

with st.sidebar.expander("Ordering Rules", expanded=True):
    unit_cost = st.number_input(
        "Cost to Buy One Unit ($)",
        min_value=0.01, value=1.50, step=0.10, format="%.2f",
        help="The cost you pay to your supplier for each unit of product."
    )
    sales_price = st.number_input(
        "Selling Price Per Unit ($)",
        min_value=0.01, value=4.50, step=0.10, format="%.2f",
        help="The price at which you sell each unit of product to your customers."
    )
    st.markdown("---")

if st.sidebar.button("Clear All Saved Data"):
    clear_all_state_data(debug_mode=debug_mode)
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

# --- Main Content Area with Tabs (now on top) ---
tab1, tab2, tab3 = st.tabs(["Welcome & Setup", "Data Management & Customization", "Simulation Results"])

with tab1:
    st.header("Welcome to Synergy Brew Inventory")
    st.markdown("Compare two different inventory strategies to find the one that provides the most value.")
    st.markdown("---")

    # --- UI for Scenario Configuration (Side-by-side) ---
    scenario_A_col, scenario_B_col = st.columns(2)

    def create_scenario_ui(col, scenario_name, scenario_key):
        with col:
            st.subheader(f"Strategy {scenario_name} Parameters")

            # Use st.session_state to persist values and get default values
            initial_stock = st.number_input(
                "Initial Stock",
                min_value=0,
                value=st.session_state.get(f'initial_stock_{scenario_key}', 100),
                key=f'initial_stock_input_{scenario_key}'
            )

            forecasting_models = ["Prophet", "Moving Average", "Actual Sales Data (Baseline)"]
            forecasting_model = st.selectbox(
                "Forecasting Model",
                forecasting_models,
                index=forecasting_models.index(st.session_state.get(f'model_{scenario_key}', "Prophet")),
                key=f'model_select_{scenario_key}'
            )

            moving_average_window = 0
            if forecasting_model == "Moving Average":
                moving_average_window = st.slider(
                    "Moving Average Window",
                    min_value=1,
                    max_value=30,
                    value=st.session_state.get(f'ma_window_{scenario_key}', 7),
                    key=f'ma_window_slider_{scenario_key}'
                )

            enable_prophet_tuning = st.checkbox(
                "Enable Hyperparameter Tuning",
                value=False,
                key=f'prophet_tuning_{scenario_key}',
                disabled=(forecasting_model != "Prophet")
            )

            lead_time_days = st.slider(
                "Delivery Lead Time (Days)",
                min_value=1, max_value=7,
                value=st.session_state.get(f'lead_time_{scenario_key}', DEFAULT_LEAD_TIME),
                step=1,
                key=f'lead_time_slider_{scenario_key}'
            )

            service_level = st.slider(
                "Desired Service Level",
                min_value=0.80, max_value=0.99,
                value=st.session_state.get(f'service_level_{scenario_key}', DEFAULT_SERVICE_LEVEL),
                step=0.01, format="%.1f%%",
                key=f'service_level_slider_{scenario_key}'
            )

            min_order_quantity = st.number_input(
                "Minimum Order Size",
                min_value=0,
                value=st.session_state.get(f'min_order_qty_{scenario_key}', DEFAULT_MIN_ORDER_QTY),
                step=5,
                key=f'min_order_qty_input_{scenario_key}'
            )

            # Save all scenario parameters to session state for later use
            st.session_state[f'initial_stock_{scenario_key}'] = initial_stock
            st.session_state[f'model_{scenario_key}'] = forecasting_model
            st.session_state[f'ma_window_{scenario_key}'] = moving_average_window
            st.session_state[f'enable_prophet_tuning_{scenario_key}'] = enable_prophet_tuning
            st.session_state[f'lead_time_{scenario_key}'] = lead_time_days
            st.session_state[f'service_level_{scenario_key}'] = service_level
            st.session_state[f'min_order_qty_{scenario_key}'] = min_order_quantity

    create_scenario_ui(scenario_A_col, "A", "A")
    create_scenario_ui(scenario_B_col, "B", "B")

    st.markdown("---")

    if st.button("Run Simulation", type="primary"):
        data_file_name = f"{selected_store_key}_{selected_product_key}_enriched_sales_history.csv"
        data_file_path = os.path.join(SIM_DATA_DIR, data_file_name)

        if not os.path.exists(data_file_path):
            st.error("Sales data not found. Please generate it in the 'Data Management & Customization' tab.")
            st.stop()

        # --- NEW: Run Prophet Tuning if enabled for either scenario A or B ---
        tuned_params_A = None
        if st.session_state.enable_prophet_tuning_A and st.session_state.model_A == "Prophet":
            tuning_df = load_all_enriched_sales_data_for_cv(selected_product_key, selected_store_key)
            if not tuning_df.empty and len(tuning_df) > 60:
                initial_train_days_val = f"{max(30, int(len(tuning_df) * 0.7))} days"
                period_between_cv_val = f"{max(7, int(len(tuning_df) * 0.1))} days"
                horizon_days_str = f"{DEFAULT_FORECAST_HORIZON_DAYS} days"
                tuned_params_A, _ = get_tuned_prophet_params(tuning_df, initial_train_days_val, period_between_cv_val, horizon_days_str, PROPHET_PARAM_GRID, debug_mode)

        tuned_params_B = None
        if st.session_state.enable_prophet_tuning_B and st.session_state.model_B == "Prophet":
            tuning_df = load_all_enriched_sales_data_for_cv(selected_product_key, selected_store_key)
            if not tuning_df.empty and len(tuning_df) > 60:
                initial_train_days_val = f"{max(30, int(len(tuning_df) * 0.7))} days"
                period_between_cv_val = f"{max(7, int(len(tuning_df) * 0.1))} days"
                horizon_days_str = f"{DEFAULT_FORECAST_HORIZON_DAYS} days"
                tuned_params_B, _ = get_tuned_prophet_params(tuning_df, initial_train_days_val, period_between_cv_val, horizon_days_str, PROPHET_PARAM_GRID, debug_mode)

        with st.spinner("Running simulations for both scenarios..."):
            # --- SCENARIO A: Instantiate the Agent and Run Simulation ---
            agent_A = InventorySimulationAgent(
                product_key=selected_product_key,
                store_key=selected_store_key,
                initial_stock=st.session_state.initial_stock_A,
                lead_time_days=st.session_state.lead_time_A,
                min_order_quantity=st.session_state.min_order_qty_A,
                service_level=st.session_state.service_level_A,
                forecasting_model=st.session_state.model_A,
                moving_average_window=st.session_state.ma_window_A,
                prophet_params=tuned_params_A,
                debug_mode=debug_mode
            )
            st.session_state.results_A = agent_A.run_simulation(
                num_days=num_days_to_simulate,
                data_file_path=data_file_path,
                unit_cost=unit_cost, # Pass the new parameters
                sales_price=sales_price # Pass the new parameters
            )

            # --- SCENARIO B: Instantiate the Agent and Run Simulation ---
            agent_B = InventorySimulationAgent(
                product_key=selected_product_key,
                store_key=selected_store_key,
                initial_stock=st.session_state.initial_stock_B,
                lead_time_days=st.session_state.lead_time_B,
                min_order_quantity=st.session_state.min_order_qty_B,
                service_level=st.session_state.service_level_B,
                forecasting_model=st.session_state.model_B,
                moving_average_window=st.session_state.ma_window_B,
                prophet_params=tuned_params_B,
                debug_mode=debug_mode
            )
            st.session_state.results_B = agent_B.run_simulation(
                num_days=num_days_to_simulate,
                data_file_path=data_file_path,
                unit_cost=unit_cost, # Pass the new parameters
                sales_price=sales_price # Pass the new parameters
            )

        st.success("Simulations complete! Head to the 'Simulation Results' tab to see the comparison.")


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
            for product_key in PRODUCT_CATALOG.keys():
                for store_key in STORE_LOCATIONS_ALL.keys():
                    data_file_name = f"{store_key}_{product_key}_enriched_sales_history.csv"
                    data_file_path = os.path.join(SIM_DATA_DIR, data_file_name)

                    if force_overwrite_data or not os.path.exists(data_file_path):
                        try:
                            generate_predefined_data_file(
                                product_key=product_key,
                                store_key=store_key,
                                spike_probability=0.01, # Use a fixed value for data generation
                                spike_multiplier=1.5,
                                force_recreate=True
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
            custom_store_key_for_gen = st.selectbox(
                "Generate Data for Which Store?",
                options=all_store_keys,
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
                        custom_summer_factor, custom_winter_factor, custom_store_key_for_gen,
                        spike_probability=0.01,
                        spike_multiplier=1.5
                    )

                CUSTOM_PRODUCT_CATALOG[custom_product_id] = custom_product_name
                save_custom_products(CUSTOM_PRODUCT_CATALOG)

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
                for k, v in custom_only_stores.items():
                    if v == store_to_delete:
                        key_to_delete = k
                        break
                if key_to_delete:
                    del CUSTOM_STORE_LOCATIONS[key_to_delete]
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

    # --- Check for both scenario results, with robust error handling ---
    if st.session_state.results_A and 'financial_log' in st.session_state.results_A and \
       st.session_state.results_B and 'financial_log' in st.session_state.results_B:
        st.subheader("Summary of Performance Metrics")

        # Convert logs to DataFrames
        df_A = pd.DataFrame(st.session_state.results_A['financial_log'])
        df_B = pd.DataFrame(st.session_state.results_B['financial_log'])

        df_inv_A = pd.DataFrame(st.session_state.results_A['inventory_log'])
        df_inv_B = pd.DataFrame(st.session_state.results_B['inventory_log'])

        # Ensure date columns are datetime objects for plotting
        df_A['date'] = pd.to_datetime(df_A['date'])
        df_B['date'] = pd.to_datetime(df_B['date'])
        df_inv_A['date'] = pd.to_datetime(df_inv_A['date'])
        df_inv_B['date'] = pd.to_datetime(df_inv_B['date'])

        # Calculate a more detailed summary
        def calculate_summary_metrics(financial_df, inventory_df):
            total_cost = financial_df['total_daily_cost'].sum()
            holding_cost = financial_df['holding_cost'].sum()
            ordering_cost = financial_df['ordering_cost'].sum()
            stockout_cost = financial_df['stockout_cost'].sum()
            lost_sales = inventory_df['lost_sales_today'].sum()
            total_sales = inventory_df['actual_sales_today'].sum()
            service_level = (total_sales - lost_sales) / total_sales if total_sales > 0 else 1.0

            return {
                "Total Cost": total_cost,
                "Holding Cost": holding_cost,
                "Ordering Cost": ordering_cost,
                "Stockout Cost": stockout_cost,
                "Total Lost Sales (Units)": lost_sales,
                "Service Level": service_level
            }

        metrics_A = calculate_summary_metrics(df_A, df_inv_A)
        metrics_B = calculate_summary_metrics(df_B, df_inv_B)

        summary_df = pd.DataFrame({
            "Metric": list(metrics_A.keys()),
            "Scenario A": list(metrics_A.values()),
            "Scenario B": list(metrics_B.values())
        }).set_index("Metric")

        # Apply formatting to the Service Level row
        def format_row(row):
            if row.name == "Service Level":
                return [f"{v:.2%}" for v in row]
            else:
                return [f"${v:,.2f}" if v > 10 else f"{v:,.2f}" for v in row]

        summary_df_formatted = summary_df.apply(format_row, axis=1)

        st.dataframe(summary_df_formatted, use_container_width=True)

        st.subheader("Visual Comparison")

        # Cumulative Cost Plot
        fig_cost = plot_cumulative_costs(df_A, df_B)
        st.plotly_chart(fig_cost, use_container_width=True)

        # Inventory Level Comparison Plot (Corrected function call)
        fig_stock_comparison = plot_inventory_comparison(
            df_inv_A, df_inv_B,
            PRODUCT_CATALOG_ALL[selected_product_key],
            STORE_LOCATIONS_ALL[selected_store_key]
        )
        st.plotly_chart(fig_stock_comparison, use_container_width=True)

        # Lost Sales Plot
        fig_lost_sales = plot_cumulative_lost_sales(df_inv_A, df_inv_B)
        st.plotly_chart(fig_lost_sales, use_container_width=True)

    else:
        st.info("Please configure and run the simulation for both scenarios in the 'Welcome & Setup' tab to see the results here.")
