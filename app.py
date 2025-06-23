import streamlit as st
import polars as pl
import pandas as pd
import time
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

from utils.rfi_analysis import (
    get_rfi_matrix,
    calculate_dormancy,
    process_all_users
)


def render_individual_user_analysis(filtered_df):
    """Render the individual user RFI analysis section."""
    st.subheader("Process RFI Matrix for Individual Users")
    
    user_id = st.number_input(
        "Enter User ID for RFI Analysis:",
        min_value=0,
        max_value=filtered_df.shape[0]-1,
        value=0
    )
    
    if st.button("Calculate RFI Matrix"):
        rfi_matrix, features = get_rfi_matrix(filtered_df, user_id)

        # Calculate 6 months dormancy (observation period only)
        first_period_duration = 180
        observation_df = filtered_df.select(filtered_df.columns[:first_period_duration])
        dormancy_value = calculate_dormancy(observation_df, user_id)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("RFI Matrix")
            st.dataframe(rfi_matrix)

            st.subheader("6 Months Dormancy")
            st.metric(
                label="Dormancy Value (Observation Period : First 180 Days only)", 
                value=int(dormancy_value)
            )
            
        with col2:
            st.subheader("User Features")
            feature_names = [
                "Activity Ratio", "Num Episodes", "Avg Recency", "Min Recency",
                "Avg Relevance", "Max Relevance", "Activity Periodicity Score",
                "Inactivity Linearity", "Activity Variability", 
                "Inactivity Growth Rate", "Recent Activity Density"
            ]
            
            features_df = pl.DataFrame({
                "Feature": feature_names,
                "Value": features
            })
            st.dataframe(features_df)


def render_process_all_users(filtered_df):
    """Render the process all users section."""
    st.subheader("Processing All Users")

    if st.button("Process All Users"):
        start_time = time.time()
        total_users = filtered_df.shape[0]
    
        # Initialize results list
        results = []
    
        # Create progress bar and status containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        timer_text = st.empty()
        
        first_period_duration = 180
        observation_df = filtered_df.select(filtered_df.columns[:first_period_duration])
        
        for i, user_id in enumerate(range(total_users)):
            # Update timer
            elapsed_time = time.time() - start_time
            timer_text.text(f"⏱️ Time elapsed: {elapsed_time:.2f} seconds")
            
            # Update progress
            progress = (i + 1) / total_users
            progress_bar.progress(progress)
            status_text.text(f"Processing user {i + 1}/{total_users}")
        
            try:
                # Calculate RFI matrix and features
                _, features = get_rfi_matrix(observation_df, user_id)
                
                # Calculate dormancy
                dormancy_value = calculate_dormancy(observation_df, user_id)
                
                # Store results
                results.append({
                    "User ID": user_id,
                    "Activity Ratio": features[0],  # activity_ratio
                    "Inactivity Linearity": features[7],  # inactivity_linearity
                    "6 Months Dormancy": int(dormancy_value)
                })
            
            except Exception as e:
                # Handle errors gracefully
                results.append({
                    "User ID": user_id,
                    "Activity Ratio": 0,
                    "Inactivity Linearity": 0,
                    "6 Months Dormancy": 0
                })
    
        # Final processing complete
        total_time = time.time() - start_time
        final_df = pl.DataFrame(results)
        
        # Clear progress indicators and show completion
        progress_bar.empty()
        status_text.empty()
        timer_text.empty()
    
        st.success(f"✅ Processing completed! Total time: {total_time:.2f} seconds | Users processed: {total_users}")
    
        # Display results
        st.subheader("Results Summary")
        final_df = final_df.with_columns([pl.col(col).round(2) for col in final_df.columns if final_df[col].dtype in [pl.Float32, pl.Float64]])
        st.dataframe(final_df.head(10))
    
        # Download button
        csv_data = final_df.to_pandas().to_csv(index=False)
        st.download_button(
            label="📥 Download Full Results as CSV",
            data=csv_data,
            file_name="user_analysis_results.csv",
            mime="text/csv",
            key="download_results"
        )
        st.session_state.processed_all_users = True


def render_rfi_analysis_section(filtered_df):
    """Render the complete RFI analysis section with both individual and batch processing."""
    st.markdown("---")
    render_individual_user_analysis(filtered_df)
    
    st.markdown("---")
    render_process_all_users(filtered_df)


def get_filtered_dataframe():
    """Get the filtered DataFrame from session state or file upload."""
    # Check if we already have filtered data from Phase 1
    if st.session_state.data_filtered and 'filtered_df' in st.session_state:
        return st.session_state.filtered_df
    
    # If not, prompt for file upload
    st.info("Please upload a filtered dataset (pivot table structure: users as rows, dates as columns)")
    uploaded_filtered_file = st.file_uploader("Choose a filtered CSV file", type="csv", key="rfi_upload")
    
    if uploaded_filtered_file is not None:
        if st.button("Upload Filtered Data") or 'uploaded_filtered_df' in st.session_state:
            with st.spinner("Loading filtered data..."):
                filtered_df, error = load_csv_data(uploaded_filtered_file)
                
                if error:
                    st.error(f"Error loading data: {error}")
                    return None
                else:
                    st.session_state.uploaded_filtered_df = filtered_df
                    st.success("Filtered data uploaded successfully!")
                    return filtered_df
    
    # Return uploaded data if it exists in session state
    if 'uploaded_filtered_df' in st.session_state:
        return st.session_state.uploaded_filtered_df
    
    return None


def main():
    st.set_page_config(
        page_title="RFI App",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 RFI Processing for Dormancy Calculation")
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
    
    # Phase 1: Data Preprocessing
    # Step 1: Data Upload
    st.header("Data Preprocessing")
    st.subheader("1. 📁 Data Upload")
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
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Shape", f"{info['shape'][0]} x {info['shape'][1]}")
                        with col2:
                            st.metric("Date Range", f"{info['date_range'][0]} to {info['date_range'][1]}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Unique Dates", info['unique_dates'])
                        with col2:
                            st.metric("Unique Users", info['unique_users'])
                        
                        # Display head
                        st.subheader("Data Preview")
                        st.dataframe(df.head().to_pandas())
    
    # Step 2: Data Cleaning
    if st.session_state.data_uploaded:
        st.markdown("---")
        st.subheader("2. 🔧 Data Cleaning")
        
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
        st.subheader("3. 🔄 Data Transformation")
        
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
        st.subheader("4. 🎯 Dropout Analysis & Filtering")
        
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
        st.subheader("5. 💾 Download Processed Data")
        
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
        st.subheader("6. 📈 Data Visualizations")
        
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

    # Phase 2: RFI Matrix Analysis
    st.markdown("---")
    st.header("RFI Matrix Analysis")

    # Get filtered DataFrame (either from Phase 1 or file upload)
    filtered_df = get_filtered_dataframe()
    
    # If we have a filtered DataFrame, show the RFI analysis options
    if filtered_df is not None:
        render_rfi_analysis_section(filtered_df)


if __name__ == "__main__":
    main()