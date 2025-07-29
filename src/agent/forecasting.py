import pandas as pd
import logging
import os
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import numpy as np # <--- NEW: Import numpy for np.sqrt, etc if not already present

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
        # If 'is_holiday' wasn't in original data but Prophet uses it as an extra regressor later,
        # it might need to be explicitly added and calculated based on `HOLIDAYS` in `train_and_forecast_model`
        # For now, this handles if it *was* in the original CSV.

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
    debug_mode: bool = False
) -> tuple[Prophet, pd.DataFrame]:
    """
    Trains a Prophet model on the provided sales history and generates a forecast.
    Includes holidays as a regressor.
    """
    if sales_history_df.empty:
        logging.warning("Sales history is empty. Cannot train model or forecast.")
        return None, pd.DataFrame()

    # Ensure 'ds' is datetime and 'y' is numeric
    sales_history_df['ds'] = pd.to_datetime(sales_history_df['ds'])
    sales_history_df['y'] = pd.to_numeric(sales_history_df['y'])

    # Filter out future dates that might have crept into history for training
    # Only train on historical data where 'y' is known
    train_df = sales_history_df[['ds', 'y']].copy()
    
    # Add holiday regressors to the training data if not already present
    # This is important if `load_enriched_sales_data` didn't add it for all dates or if
    # the sales_history_df is a slice that excludes some holiday calculations.
    holiday_dates_dt = [pd.to_datetime(d) for d in HOLIDAYS]
    train_df['is_holiday'] = train_df['ds'].isin(holiday_dates_dt).astype(int)

    # Initialize Prophet model
    model = Prophet(
        seasonality_mode='multiplicative',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False, # Assuming daily sales might not have strong fixed patterns by hour
        holidays=pd.DataFrame({'ds': holiday_dates_dt, 'holiday': 'holiday'}) if holiday_dates_dt else None
    )

    # Add 'is_holiday' as an extra regressor IF it exists and model isn't configured for holidays directly
    # Prophet's `holidays` argument is preferred. If we're using the holidays argument,
    # adding `is_holiday` as an extra regressor might be redundant or problematic if the holiday effect is double-counted.
    # Let's rely on the `holidays` argument for now, it's more standard.
    # If custom regressors are needed, they would be added via `model.add_regressor`.
    # For now, remove the `add_regressor` line if you're using `holidays` param in Prophet init.
    # If the `holidays` argument wasn't sufficient, and `is_holiday` represents other events, keep it.
    # For this project, the `holidays` argument in Prophet constructor is usually enough.
    # Let's remove the explicit `add_regressor` for `is_holiday` here to avoid confusion and rely on the `holidays` parameter in the model init.
    # If your original design intent was for 'is_holiday' to be a generic regressor *beyond* predefined Prophet holidays,
    # then you'd keep this. But typically, if you have a `holidays` list, you use Prophet's built-in holiday functionality.

    # We are already passing `holidays=pd.DataFrame(...)` to Prophet constructor.
    # This should be sufficient. No need to `add_regressor('is_holiday')` unless 'is_holiday' means something else.

    try:
        model.fit(train_df)
    except Exception as e:
        logging.error(f"Error fitting Prophet model: {e}. Check training data for issues like NaNs in 'y'.")
        return None, pd.DataFrame()

    # Create future dataframe for forecasting
    future = model.make_future_dataframe(periods=periods_to_forecast, include_history=False)
    
    # Add 'is_holiday' regressor to future dataframe
    future['is_holiday'] = future['ds'].isin(holiday_dates_dt).astype(int)

    # Ensure all regressor columns present in training data are also in future data
    # (Prophet automatically handles regressors if passed in fit and predict)

    # Predict
    forecast = model.predict(future)
    if debug_mode:
        logging.info(f"Generated forecast for {periods_to_forecast} periods.")
        logging.debug(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

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
    """
    if df.empty or len(df) < pd.to_timedelta(initial_train_days).days + pd.to_timedelta(horizon_forecast_days).days:
        logging.warning("Not enough data for cross-validation.")
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
    # model_cv.add_regressor('is_holiday') # Only add if it's not covered by holidays param

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