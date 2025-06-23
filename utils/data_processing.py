import polars as pl
import pandas as pd
import io


def load_csv_data(uploaded_file):
    """Load CSV data using Polars"""
    try:
        # Read the uploaded file as bytes and convert to string
        bytes_data = uploaded_file.getvalue()
        string_data = bytes_data.decode('utf-8')
        
        # Use Polars to read CSV from string
        df = pl.read_csv(io.StringIO(string_data))
        return df, None
    except Exception as e:
        return None, str(e)


def get_data_info(df: pl.DataFrame):
    """Get basic information about the dataset"""
    try:
        # Convert Date column to datetime for analysis
        df_temp = df.with_columns(pl.col('Date').str.to_datetime().dt.date())
        
        info = {
            'shape': df.shape,
            'date_range': (df_temp['Date'].min(), df_temp['Date'].max()),
            'unique_dates': df_temp['Date'].n_unique(),
            'unique_users': df['ID_Cust'].n_unique()
        }
        return info, None
    except Exception as e:
        return None, str(e)


def data_cleaning_pipeline(df: pl.DataFrame):
    """Data cleaning pipeline from the notebook"""
    try:
        # Convert 'Date' column to datetime
        df = df.with_columns(pl.col('Date').str.to_datetime().dt.date())

        # Remove duplicates based on 'ID_Cust' and 'Date'
        df = df.unique(subset=['ID_Cust', 'Date'])

        # Check if dates are ordered
        is_ordered = df['Date'].is_sorted()
        if not is_ordered:
            df = df.sort(['Date', 'ID_Cust'])

        return df, None
    except Exception as e:
        return None, str(e)


def get_cleaning_stats(original_df: pl.DataFrame, cleaned_df: pl.DataFrame):
    """Get statistics after cleaning"""
    try:
        reduction_percentage = round((original_df.shape[0] - cleaned_df.shape[0]) / original_df.shape[0] * 100, 1)
        
        stats = {
            'new_shape': cleaned_df.shape,
            'reduction_percentage': reduction_percentage,
            'date_range': (cleaned_df['Date'].min(), cleaned_df['Date'].max()),
            'unique_dates': cleaned_df['Date'].n_unique(),
            'unique_users': cleaned_df['ID_Cust'].n_unique()
        }
        return stats, None
    except Exception as e:
        return None, str(e)


def transform_to_pivot(df: pl.DataFrame):
    """Transform data to pivot table"""
    try:
        # Add Active column
        data = df.with_columns(pl.lit(1).alias("Active"))

        # Create pivot table
        pivot_table = data.pivot(
            index="ID_Cust",
            columns="Date",
            values="Active",
            aggregate_function="first"
        ).fill_null(0).cast(pl.Int16)

        # Sort by ID_Cust and drop the ID_Cust column
        pivot_table = pivot_table.sort("ID_Cust").drop("ID_Cust")
        
        return pivot_table, None
    except Exception as e:
        return None, str(e)


def longest_inactivity_streak(row):
    """Compute longest inactivity streak per user"""
    inactive_str = "".join(map(str, row))  # Convert row to string of 1s and 0s
    return max(map(len, inactive_str.split("1"))) if "0" in inactive_str else 0


def analyze_dropout_clients(pivot_df: pl.DataFrame):
    """Analyze dropout and 100% active clients"""
    try:
        # Convert to pandas for the streak calculation (as in original notebook)
        pivot_pandas = pivot_df.to_pandas()
        
        # Calculate longest inactivity streak
        pivot_pandas["Longest_Inactivity_Streak"] = pivot_pandas.apply(longest_inactivity_streak, axis=1)
        
        # Identify different client types
        dropout_clients = len(pivot_pandas[(pivot_pandas["Longest_Inactivity_Streak"] >= 120)])
        
        active_100_clients = len(pivot_pandas[pivot_pandas["Longest_Inactivity_Streak"] == 0])
        
        total_clients = len(pivot_pandas)
        
        # Get indices to drop (dropout clients)
        to_drop_indices = pivot_pandas[
            (pivot_pandas["Longest_Inactivity_Streak"] >= 120) | 
            (pivot_pandas["Longest_Inactivity_Streak"] == 0)
        ].index.tolist()
        
        stats = {
            'total_clients': total_clients,
            'dropout_clients': dropout_clients,
            'active_100_clients': active_100_clients,
            'to_drop_indices': to_drop_indices,
            'streaks_df': pivot_pandas[["Longest_Inactivity_Streak"]]
        }
        
        return stats, None
    except Exception as e:
        return None, str(e)


def filter_clients(pivot_df: pl.DataFrame, to_drop_indices: list):
    """Filter out dropout clients"""
    try:
        # Convert to pandas for filtering (to match notebook logic)
        pivot_pandas = pivot_df.to_pandas()
        
        # Filter out the clients
        filtered_df = pivot_pandas[~pivot_pandas.index.isin(to_drop_indices)]
        
        # Convert back to polars
        filtered_polars = pl.DataFrame(filtered_df)
        
        return filtered_polars, None
    except Exception as e:
        return None, str(e)


def save_to_csv(df, filename="cleaned_filtered_data.csv"):
    """Convert polars DataFrame to CSV string for download"""
    try:
        if isinstance(df, pl.DataFrame):
            # Convert to pandas for CSV export
            pandas_df = df.to_pandas()
            csv_string = pandas_df.to_csv(index=False)
        else:
            # Already pandas
            csv_string = df.to_csv(index=False)
        return csv_string, None
    except Exception as e:
        return None, str(e)