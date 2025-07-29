# config/catalogs.py

# --- Starbucks Theme Configuration ---
# Expanded Product Catalog
PRODUCT_CATALOG = {
    "Latte": "Latte", 
    "PikePlaceRoast": "Pike Place Roast",
    "CaramelMacchiato": "Caramel Macchiato",
    "ColdBrew": "Cold Brew",
    "Espresso": "Espresso Shot", 
    "GreenTeaLatte": "Green Tea Latte", 
    "Frappuccino": "Frappuccino" 
}

# Expanded Store Locations
STORE_LOCATIONS = {
    "Store_A_Downtown": "Downtown Seattle", 
    "Store_B_Campus": "University Campus",
    "Store_C_Suburban": "Suburban Hub",
    "Store_D_Airport": "Airport Terminal", 
    "Store_E_Mall": "Shopping Mall" 
}

# Product and Store Configurations for data generation
# These will be moved here from generate_sim_data.py
PRODUCT_CONFIGS = {
    "Latte": {"base_sales": 150, "weekly_peak_factor": 1.2, "summer_factor": 1.0, "winter_factor": 1.1},
    "PikePlaceRoast": {"base_sales": 100, "weekly_peak_factor": 1.1, "summer_factor": 0.9, "winter_factor": 1.2},
    "CaramelMacchiato": {"base_sales": 120, "weekly_peak_factor": 1.3, "summer_factor": 1.1, "winter_factor": 1.0},
    "ColdBrew": {"base_sales": 80, "weekly_peak_factor": 1.0, "summer_factor": 1.5, "winter_factor": 0.5},
    "Espresso": {"base_sales": 60, "weekly_peak_factor": 1.1, "summer_factor": 0.95, "winter_factor": 1.05}, 
    "GreenTeaLatte": {"base_sales": 90, "weekly_peak_factor": 1.15, "summer_factor": 1.2, "winter_factor": 0.9}, 
    "Frappuccino": {"base_sales": 180, "weekly_peak_factor": 1.3, "summer_factor": 1.8, "winter_factor": 0.7} 
}

STORE_CONFIGS = {
    "Store_A_Downtown": {"sales_multiplier": 1.0, "holiday_impact": 0.8},
    "Store_B_Campus": {"sales_multiplier": 0.9, "holiday_impact": 0.5},
    "Store_C_Suburban": {"sales_multiplier": 0.8, "holiday_impact": 1.1},
    "Store_D_Airport": {"sales_multiplier": 1.2, "holiday_impact": 0.9, "daily_variation": 0.15},
    "Store_E_Mall": {"sales_multiplier": 1.1, "holiday_impact": 1.5, "daily_variation": 0.10}
}