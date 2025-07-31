# src/agent/forecasting.py

import pandas as pd
import logging
import os
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LinearRegression # NEW: Import LinearRegression

from config.settings import HOLIDAYS, DEFAULT_FORECAST_HORIZON_DAYS, START_DATE, END_DATE

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_enriched_sales_data(file_path: str) -> pd.DataFrame:
    """
    Loads enriched sales data from a CSV, ensuring 'ds' is datetime and 'y' is numeric.
    Also ensures continuity of dates within the loaded range and fills missing sales with 0.
    Handles potential NaNs in 'y' and 'is_holiday' columns.
    """
    if not os.path.exists(file_path):
        logging.error(f"Data file not found: {file_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Ensure 'y' column exists and is numeric, handle potential NaNs
        if 'y' not in df.columns:
            logging.error(f"'y' column not found in {file_path}")
            return pd.DataFrame()
        
        # Explicitly convert 'y' to numeric, coercing errors to NaN
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        
        # Fill NaNs in 'y' with 0 (missing sales means 0 sales)
        initial_nan_count_y = df['y'].isnull().sum()
        if initial_nan_count_y > 0:
            logging.warning(f"Found {initial_nan_count_y} NaN values in 'y' column of {file_path}. Filling with 0.")
            df['y'] = df['y'].fillna(0)

        # Ensure no duplicate dates
        df = df.drop_duplicates(subset=['ds']).sort_values('ds').reset_index(drop=True)

        # Determine the full date range based on the loaded data
        # If the dataframe is empty after initial load, use global START_DATE/END_DATE for range
        min_date = df['ds'].min() if not df.empty else pd.to_datetime(START_DATE)
        max_date = df['ds'].max() if not df.empty else pd.to_datetime(END_DATE)
        
        full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        full_df = pd.DataFrame({'ds': full_date_range})
        
        # Merge with existing data, filling 'y' with 0 for missing dates
        df_merged = pd.merge(full_df, df, on='ds', how='left')
        df_merged['y'] = df_merged['y'].fillna(0) # Fill new NaNs from merge with 0
        
        # Re-calculate 'is_holiday' for the merged DataFrame to ensure no NaNs in regressors
        # This is crucial because `merge` can introduce NaNs in columns from the right DataFrame
        # if a date exists in `full_df` but not in the original `df`.
        if 'is_holiday' in df_merged.columns: # Check if 'is_holiday' was originally in the loaded data
            holiday_dates = [pd.to_datetime(d) for d in HOLIDAYS]
            df_merged['is_holiday'] = df_merged['ds'].isin(holiday_dates).astype(int)
        
        # Ensure 'y' is integer for sales counts
        df_merged['y'] = df_merged['y'].astype(int)
        
        logging.info(f"Loaded {len(df_merged)} rows of sales data from {file_path} after ensuring continuity and filling NaNs.")
        return df_merged

    except Exception as e:
        logging.error(f"Error loading or processing data from {file_path}: {e}")
        return pd.DataFrame()


def train_and_forecast_model(
    sales_history_df: pd.DataFrame,
    periods_to_forecast: int = DEFAULT_FORECAST_HORIZON_DAYS,
    forecasting_model_type: str = "Prophet",
    moving_average_window: int = 7,
    prophet_params: dict = None,
    debug_mode: bool = False
) -> tuple[object, pd.DataFrame]:
    """
    Trains a forecasting model (Prophet, Moving Average, or Linear Regression)
    and generates a forecast.
    """
    if sales_history_df.empty:
        logging.warning("Sales history is empty. Cannot train model or forecast.")
        return None, pd.DataFrame()

    # Ensure 'ds' is datetime and 'y' is numeric
    sales_history_df['ds'] = pd.to_datetime(sales_history_df['ds'])
    sales_history_df['y'] = pd.to_numeric(sales_history_df['y'])

    train_df = sales_history_df[['ds', 'y']].copy()
    
    last_history_date = train_df['ds'].max()
    future_dates = pd.date_range(start=last_history_date + timedelta(days=1),
                                 periods=periods_to_forecast,
                                 freq='D')
    future_df = pd.DataFrame({'ds': future_dates})

    model = None
    forecast = pd.DataFrame()

    if forecasting_model_type == "Moving Average":
        if len(train_df) < moving_average_window:
            logging.warning(f"Not enough data ({len(train_df)} days) for Moving Average window of {moving_average_window}. Using mean of available history.")
            ma_value = train_df['y'].mean() if not train_df.empty else 0.0
        else:
            ma_value = train_df['y'].tail(moving_average_window).mean()
        
        future_df['yhat'] = ma_value
        
        if debug_mode:
            logging.info(f"Generated Moving Average ({moving_average_window}d) forecast: {ma_value:.2f} for {periods_to_forecast} periods.")
            logging.debug(future_df.head())
        
        forecast = future_df[['ds', 'yhat']]
        
    elif forecasting_model_type == "Prophet":
        holiday_dates_dt = [pd.to_datetime(d) for d in HOLIDAYS]
        train_df['is_holiday'] = train_df['ds'].isin(holiday_dates_dt).astype(int)

        prophet_init_params = {
            'seasonality_mode': 'multiplicative',
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'holidays': pd.DataFrame({'ds': holiday_dates_dt, 'holiday': 'holiday'}) if holiday_dates_dt else None
        }
        
        if prophet_params:
            prophet_init_params.update(prophet_params)
            logging.info(f"Training Prophet with custom parameters: {prophet_params}")
        else:
            logging.info("Training Prophet with default parameters.")

        model = Prophet(**prophet_init_params)

        try:
            model.fit(train_df)
        except Exception as e:
            logging.error(f"Error fitting Prophet model: {e}. Check training data for issues like NaNs in 'y'.")
            return None, pd.DataFrame()

        future_df['is_holiday'] = future_df['ds'].isin(holiday_dates_dt).astype(int)

        forecast = model.predict(future_df)
        if debug_mode:
            logging.info(f"Generated Prophet forecast for {periods_to_forecast} periods.")
            logging.debug(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

        forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    # NEW: Add an elif block for Linear Regression
    elif forecasting_model_type == "Linear Regression":
        model, forecast = linear_regression_forecast(sales_history_df, periods_to_forecast, debug_mode)
        if debug_mode:
            logging.info(f"Generated Linear Regression forecast for {periods_to_forecast} periods.")
            logging.debug(forecast.head())
    # END NEW

    else:
        logging.error(f"Unknown forecasting model type: {forecasting_model_type}")
        return None, pd.DataFrame()

    return model, forecast


def calculate_cv_metrics(
    df: pd.DataFrame,
    initial_train_days: str = '730 days', # 2 years
    period_between_cv: str = '90 days', # Retrain every 3 months
    horizon_forecast_days: str = '30 days', # Forecast 30 days ahead
    debug_mode: bool = False
) -> pd.DataFrame:
    """
    Performs cross-validation for Prophet model and calculates performance metrics.
    Note: Cross-validation is only applicable to Prophet, not Moving Average.
    """
    if df.empty or len(df) < pd.to_timedelta(initial_train_days).days + pd.to_timedelta(horizon_forecast_days).days:
        logging.warning("Not enough data for Prophet cross-validation.")
        return pd.DataFrame()
    
    # Ensure 'ds' is datetime
    df['ds'] = pd.to_datetime(df['ds'])

    # Re-add 'is_holiday' to the full DataFrame for cross-validation
    holiday_dates_dt = [pd.to_datetime(d) for d in HOLIDAYS]
    df['is_holiday'] = df['ds'].isin(holiday_dates_dt).astype(int)

    # Initialize Prophet model for CV
    model_cv = Prophet(
        seasonality_mode='multiplicative',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=pd.DataFrame({'ds': holiday_dates_dt, 'holiday': 'holiday'}) if holiday_dates_dt else None
    )

    try:
        df_cv = cross_validation(
            model_cv,
            df,
            initial=initial_train_days,
            period=period_between_cv,
            horizon=horizon_forecast_days
        )
        if debug_mode:
            logging.info("Prophet cross-validation completed.")
            logging.debug(df_cv.head())
        
        df_p = performance_metrics(df_cv)
        if debug_mode:
            logging.info("Prophet performance metrics calculated.")
            logging.debug(df_p.head())
        
        return df_p
    except Exception as e:
        logging.error(f"Error during Prophet cross-validation or performance metrics calculation: {e}")
        return pd.DataFrame()

# NEW: Function to implement Linear Regression
def linear_regression_forecast(sales_history_df, forecast_horizon, debug_mode):
    """
    Trains a simple linear regression model on historical data
    and forecasts future demand.
    """
    if sales_history_df.empty:
        logging.warning("Sales history is empty for Linear Regression. Cannot forecast.")
        return None, pd.DataFrame()

    # Create a feature for the day number
    sales_history_df_copy = sales_history_df.copy()
    sales_history_df_copy['day_num'] = np.arange(len(sales_history_df_copy)) + 1
    
    # Reshape data for scikit-learn
    X = sales_history_df_copy[['day_num']]
    y = sales_history_df_copy['y']
    
    # Train the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Prepare future data for prediction
    last_day_num = sales_history_df_copy['day_num'].iloc[-1]
    future_day_nums = np.arange(last_day_num + 1, last_day_num + forecast_horizon + 1).reshape(-1, 1)
    
    # Make the forecast
    forecasted_sales = model.predict(future_day_nums)
    
    # Create a DataFrame for the output
    last_date = sales_history_df_copy['ds'].iloc[-1]
    future_dates = pd.to_datetime([last_date + timedelta(days=i) for i in range(1, forecast_horizon + 1)])
    
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': forecasted_sales
    })
    
    # Ensure forecasts are not negative
    forecast_df['yhat'] = forecast_df['yhat'].apply(lambda x: max(0, x))
    
    return model, forecast_df
# END NEW

def tune_prophet_hyperparameters(
    df: pd.DataFrame,
    initial: str,
    period: str,
    horizon: str,
    param_grid: dict,
    metric: str = 'mape', # Default metric to minimize
    debug_mode: bool = False
) -> tuple[dict, float]:
    """
    Tunes Prophet hyperparameters using cross-validation to find the best performing model.
    """
    logging.info(f"Starting Prophet hyperparameter tuning for {len(df)} data points...")

    best_params = None
    best_metric_score = float('inf') # Initialize with a very high value for minimization
    
    # Ensure 'ds' is datetime
    df['ds'] = pd.to_datetime(df['ds'])

    # Re-add 'is_holiday' to the full DataFrame for model fitting
    holiday_dates_dt = [pd.to_datetime(d) for d in HOLIDAYS]
    df['is_holiday'] = df['ds'].isin(holiday_dates_dt).astype(int)

    param_combinations = list(ParameterGrid(param_grid))
    
    if debug_mode:
        logging.debug(f"Testing {len(param_combinations)} parameter combinations.")
    
    for i, params in enumerate(param_combinations):
        if debug_mode:
            logging.debug(f"  Testing combination {i+1}/{len(param_combinations)}: {params}")
        
        try:
            m = Prophet(
                seasonality_mode='multiplicative',
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                holidays=pd.DataFrame({'ds': holiday_dates_dt, 'holiday': 'holiday'}) if holiday_dates_dt else None,
                **params
            )
            
            m.fit(df)

            df_cv = cross_validation(
                m,
                df,
                initial=initial,
                period=period,
                horizon=horizon
            )

            if df_cv.empty:
                logging.warning(f"CV returned empty for params {params}. Skipping.")
                continue

            df_p = performance_metrics(df_cv)
            
            if pd.to_timedelta(horizon) in df_p['horizon'].unique():
                current_metric_score = df_p[df_p['horizon'] == pd.to_timedelta(horizon)][metric].iloc[0]
            else:
                current_metric_score = df_p[metric].mean()
            
            if current_metric_score < best_metric_score:
                best_metric_score = current_metric_score
                best_params = params
                if debug_mode:
                    logging.debug(f"    New best {metric}: {best_metric_score:.4f} with params: {best_params}")

        except Exception as e:
            logging.error(f"Error during CV for params {params}: {e}. Skipping combination.")
            if debug_mode:
                logging.exception(e)

    logging.info(f"Prophet hyperparameter tuning completed. Best {metric}: {best_metric_score:.4f} with params: {best_params}")
    return best_params, best_metric_score