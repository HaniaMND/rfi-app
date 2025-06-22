import streamlit as st
import polars as pl
import pandas as pd
from utils.data_processing import (
    load_csv_data, get_data_info, data_cleaning_pipeline, 
    get_cleaning_stats, transform_to_pivot, analyze_dropout_clients, 
    filter_clients, save_to_csv
)
from utils.visualization import (
    plot_daily_active_users, plot_weekly_active_users, 
    plot_active_days_distribution, plot_inactivity_streaks_distribution,
    plot_activity_pattern
)


def main():
    st.set_page_config(
        page_title="RFI Data Processing App",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 RFI Data Processing Application")
    st.markdown("---")
    
    # Initialize session state
    if 'data_uploaded' not in st.session_state:
        st.session_state.data_uploaded = False
    if 'data_cleaned' not in st.session_state:
        st.session_state.data_cleaned = False
    if 'data_transformed' not in st.session_state:
        st.session_state.data_transformed = False
    if 'data_filtered' not in st.session_state:
        st.session_state.data_filtered = False
    
    # Step 1: Data Upload
    st.header("1. 📁 Data Upload")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        if st.button("Upload Data") or st.session_state.data_uploaded:
            with st.spinner("Loading data..."):
                df, error = load_csv_data(uploaded_file)
                
                if error:
                    st.error(f"Error loading data: {error}")
                else:
                    st.session_state.original_df = df
                    st.session_state.data_uploaded = True
                    
                    # Display basic info
                    info, error = get_data_info(df)
                    if error:
                        st.error(f"Error getting data info: {error}")
                    else:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Shape", f"{info['shape'][0]} x {info['shape'][1]}")
                        with col2:
                            st.metric("Date Range", f"{info['date_range'][0]} to {info['date_range'][1]}")
                        with col3:
                            st.metric("Unique Dates", info['unique_dates'])
                        with col4:
                            st.metric("Unique Users", info['unique_users'])
                        
                        # Display head
                        st.subheader("Data Preview")
                        st.dataframe(df.head().to_pandas())
    
    # Step 2: Data Cleaning
    if st.session_state.data_uploaded:
        st.markdown("---")
        st.header("2. 🧹 Data Cleaning")
        
        if st.button("Clean Data") or st.session_state.data_cleaned:
            with st.spinner("Cleaning data..."):
                cleaned_df, error = data_cleaning_pipeline(st.session_state.original_df)
                
                if error:
                    st.error(f"Error cleaning data: {error}")
                else:
                    st.session_state.cleaned_df = cleaned_df
                    st.session_state.data_cleaned = True
                    
                    # Display cleaning stats
                    stats, error = get_cleaning_stats(st.session_state.original_df, cleaned_df)
                    if error:
                        st.error(f"Error getting cleaning stats: {error}")
                    else:
                        st.success("Data cleaned successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("New Shape", f"{stats['new_shape'][0]} x {stats['new_shape'][1]}")
                        with col2:
                            st.metric("Reduction", f"{stats['reduction_percentage']}%")
                        with col3:
                            st.metric("Unique Users", stats['unique_users'])
                        
                        # Display head after cleaning
                        st.subheader("Cleaned Data Preview")
                        st.dataframe(cleaned_df.head().to_pandas())
    
    # Step 3: Data Transformation
    if st.session_state.data_cleaned:
        st.markdown("---")
        st.header("3. 🔄 Data Transformation")
        
        if st.button("Transform to Pivot Table") or st.session_state.data_transformed:
            with st.spinner("Transforming data..."):
                pivot_df, error = transform_to_pivot(st.session_state.cleaned_df)
                
                if error:
                    st.error(f"Error transforming data: {error}")
                else:
                    st.session_state.pivot_df = pivot_df
                    st.session_state.data_transformed = True
                    
                    st.success("Data transformed successfully!")
                    st.metric("Pivot Table Shape", f"{pivot_df.shape[0]} x {pivot_df.shape[1]}")
                    
                    # Display head of pivot table
                    st.subheader("Pivot Table Preview")
                    st.dataframe(pivot_df.head().to_pandas())
    
    # Step 4: Dropout Analysis and Filtering
    if st.session_state.data_transformed:
        st.markdown("---")
        st.header("4. 🎯 Dropout Analysis & Filtering")
        
        if st.button("Analyze Clients") or 'client_stats' in st.session_state:
            with st.spinner("Analyzing clients..."):
                stats, error = analyze_dropout_clients(st.session_state.pivot_df)
                
                if error:
                    st.error(f"Error analyzing clients: {error}")
                else:
                    st.session_state.client_stats = stats
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Clients", stats['total_clients'])
                    with col2:
                        st.metric("Dropout Clients", stats['dropout_clients'])
                    with col3:
                        st.metric("100% Active Clients", stats['active_100_clients'])
                    
                    # Filter button
                    if st.button("Filter Dropout Clients") or st.session_state.data_filtered:
                        with st.spinner("Filtering clients..."):
                            filtered_df, error = filter_clients(
                                st.session_state.pivot_df, 
                                stats['to_drop_indices']
                            )
                            
                            if error:
                                st.error(f"Error filtering clients: {error}")
                            else:
                                st.session_state.filtered_df = filtered_df
                                st.session_state.data_filtered = True
                                
                                st.success("Clients filtered successfully!")
                                st.metric("Remaining Users", filtered_df.shape[0])
    
    # Step 5: Download
    if st.session_state.data_filtered:
        st.markdown("---")
        st.header("5. 💾 Download Processed Data")
        
        csv_string, error = save_to_csv(st.session_state.filtered_df)
        if error:
            st.error(f"Error preparing download: {error}")
        else:
            st.download_button(
                label="Download Cleaned & Filtered Data (CSV)",
                data=csv_string,
                file_name="cleaned_filtered_pivot_table.csv",
                mime="text/csv"
            )
    
    # Step 6: Visualizations
    if st.session_state.data_filtered:
        st.markdown("---")
        st.header("6. 📈 Data Visualizations")
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Daily Active Users", 
            "Weekly Active Users", 
            "Active Days Distribution", 
            "Inactivity Streaks", 
            "User Activity Pattern"
        ])
        
        with tab1:
            fig = plot_daily_active_users(st.session_state.filtered_df)
            if fig:
                st.pyplot(fig)
        
        with tab2:
            fig = plot_weekly_active_users(st.session_state.filtered_df)
            if fig:
                st.pyplot(fig)
        
        with tab3:
            fig = plot_active_days_distribution(st.session_state.filtered_df)
            if fig:
                st.pyplot(fig)
        
        with tab4:
            if 'client_stats' in st.session_state:
                fig = plot_inactivity_streaks_distribution(st.session_state.client_stats['streaks_df'])
                if fig:
                    st.pyplot(fig)
        
        with tab5:
            st.subheader("Individual User Activity Pattern")
            user_id = st.number_input(
                "Enter User ID (0-based index):", 
                min_value=0, 
                max_value=st.session_state.filtered_df.shape[0]-1 if 'filtered_df' in st.session_state else 0,
                value=0
            )
            
            if st.button("Show Activity Pattern"):
                fig = plot_activity_pattern(st.session_state.filtered_df, user_id)
                if fig:
                    st.pyplot(fig)


if __name__ == "__main__":
    main()