import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import polars as pl
import streamlit as st


def plot_daily_active_users(pivot_df):
    """Plot daily active users over time"""
    try:
        # Convert to pandas for plotting
        if isinstance(pivot_df, pl.DataFrame):
            pivot_pandas = pivot_df.to_pandas()
        else:
            pivot_pandas = pivot_df
            
        daily_active_users = pivot_pandas.sum(axis=0)
        
        daily_active_df = pd.DataFrame({
            'Date': daily_active_users.index,
            'Active_Users': daily_active_users.values
        })
        
        daily_active_df['Date'] = pd.to_datetime(daily_active_df['Date'])
        
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.lineplot(data=daily_active_df, x='Date', y='Active_Users', ax=ax)
        ax.set_title('Daily Active Users Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Active Users')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error plotting daily active users: {str(e)}")
        return None


def plot_weekly_active_users(pivot_df):
    """Plot weekly active users (7-day rolling average)"""
    try:
        # Convert to pandas for plotting
        if isinstance(pivot_df, pl.DataFrame):
            pivot_pandas = pivot_df.to_pandas()
        else:
            pivot_pandas = pivot_df
            
        daily_active_users = pivot_pandas.sum(axis=0)
        
        daily_active_df = pd.DataFrame({
            'Date': daily_active_users.index,
            'Active_Users': daily_active_users.values
        })
        
        daily_active_df['Date'] = pd.to_datetime(daily_active_df['Date'])
        daily_active_df['Weekly_Active_Users'] = daily_active_df['Active_Users'].rolling(window=7).mean()
        
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.lineplot(data=daily_active_df, x='Date', y='Weekly_Active_Users', ax=ax)
        ax.set_title('Weekly Active Users (7-Day Rolling Average) Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Active Users (7-Day Rolling Average)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error plotting weekly active users: {str(e)}")
        return None


def plot_active_days_distribution(pivot_df):
    """Plot distribution of active days per user"""
    try:
        # Convert to pandas for plotting
        if isinstance(pivot_df, pl.DataFrame):
            pivot_pandas = pivot_df.to_pandas()
        else:
            pivot_pandas = pivot_df
            
        active_days_per_user = pivot_pandas.sum(axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(active_days_per_user, bins=25, kde=True, ax=ax)
        ax.set_title('Distribution of Active Days Per User')
        ax.set_xlabel('Number of Active Days')
        ax.set_ylabel('Number of Users')
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error plotting active days distribution: {str(e)}")
        return None


def plot_inactivity_streaks_distribution(streaks_df):
    """Plot distribution of longest inactivity streaks"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(streaks_df["Longest_Inactivity_Streak"], bins=10, kde=True, ax=ax)
        ax.set_title('Distribution of Longest Inactivity Streaks Per User')
        ax.set_xlabel('Longest Inactivity Streak (Days)')
        ax.set_ylabel('Number of Users')
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error plotting inactivity streaks: {str(e)}")
        return None


def plot_activity_pattern(pivot_df, user_id):
    """Plot activity pattern for a specific user"""
    try:
        # Convert to pandas for plotting
        if isinstance(pivot_df, pl.DataFrame):
            pivot_pandas = pivot_df.to_pandas()
        else:
            pivot_pandas = pivot_df
            
        if user_id >= len(pivot_pandas):
            st.error(f"User ID {user_id} not found. Available range: 0 to {len(pivot_pandas)-1}")
            return None
            
        user_activity = pivot_pandas.iloc[user_id]
        
        fig, ax = plt.subplots(figsize=(12, 3))
        sns.heatmap([user_activity], cmap="Blues", cbar=False, 
                   xticklabels=False, yticklabels=[f"User {user_id}"], ax=ax)
        ax.set_title(f"Activity Pattern for User {user_id}")
        ax.set_xlabel("Time (Days)")
        ax.set_ylabel("Activity")
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error plotting activity pattern: {str(e)}")
        return None