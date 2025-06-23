# RFI Processing App for Dormancy Calculation

A comprehensive Streamlit application for processing user activity data and calculating RFI (Recency, Frequency, Inactivity) matrices to determine user dormancy patterns.

## Overview

This application provides a complete pipeline for analyzing user activity data, from raw data preprocessing to advanced RFI matrix calculations. It's designed to help telecom companies identify dormant users by calculating personalized inactivity thresholds based on past usage data.

## Architecture

The application is built with a modular architecture:

```
app.py                 # Main Streamlit application
├── utils/
│   ├── data_processing.py    # Data cleaning and transformation
│   ├── visualization.py      # Charts generation
│   └── rfi_analysis.py      # RFI matrix calculations
```

## Data Flow

1. **Raw Data Input** → CSV upload with user activity records
2. **Data Cleaning** → Remove duplicate entries
3. **Pivot Transformation** → Convert to user-date matrix format
4. **Client Filtering** → Remove dropout and fully active clients
5. **RFI Analysis** → Calculate activity patterns and dormancy scores
6. **Results Export** → Download processed data and analysis results

## Installation

1. Clone the repository:
```bash
git clone git@github.com:HaniaMND/rfi-app.git
cd rfi-app
```

2. Install required dependencies:
```bash
pip install streamlit polars pandas matplotlib plotly seaborn
```

3. Run the application:
```bash
streamlit run app.py
```

## Input Data Format

### Raw Data (Phase 1 Input)
Your CSV file should contain user activity data with columns for:
- User ID
- Date/Timestamp

### Preprocessed Data (Phase 2 Direct Input)
Alternatively, upload a pivot table format:
- Rows: Users
- Columns: Dates
- Values: Activity indicators (0/1)

## Usage Guide

### Phase 1: Data Preprocessing

1. **Upload Data**: Click "Choose a CSV file" and select raw data
2. **Data Cleaning**: Review data info and click "Clean Data"
3. **Transform Data**: Convert to pivot table format
4. **Analyze Clients**: Identify dropout patterns
5. **Filter Clients**: Remove dropout users and fully active clients
6. **Download Results**: Export cleaned data
7. **Visualize**: Explore data through interactive charts

### Phase 2: RFI Analysis

1. **Load Data**: Either use Phase 1 output or upload preprocessed data
2. **Individual Analysis**: Enter user ID to calculate RFI matrix
3. **Batch Processing**: Process all users with "Process All Users" button
4. **Download Results**: Export analysis results as CSV

## Generated Features

The RFI analysis generates 11 key features per user:

1. **Activity Ratio**:  The proportion of days the user was active during the observation period.
2. **Number of Episodes**: The total count of distinct inactivity episodes experienced by the user.
3. **Average Recency**: The mean of R values from the RFI matrix, indicating the average recency of all their typical inactivity durations.
4. **Minimum Recency**: The smallest R value, highlighting the most recent occurrence of any patterned inactivity.
5. **Average Relevance**: The mean of the calculated Relevance scores from the RFI matrix.
6. **Maximum Relevance**: The highest Relevance score observed for any inactivity duration for that user.
7. **Activity Periodicity Score**: Regularity of activity patterns. Higher scores indicate more periodic activity patterns.
8. **Inactivity Linearity**: Assesses how well the durations of historical inactivity episodes (I) relate linearly to their recencies (R). Scores closer to 1 reflect a more predictable, structured inactivity pattern.
9. **Activity Variability**:  Measures consistency in the timing between active days. Lower variability implies more regular activity timing.
10. **Inactivity Growth Rate**: Captures the mean ratio of successive inactivity-episode durations. Values > 1 suggest episodes are generally getting longer.
11. **Recent Activity Density**: The proportion of active days within the last 30 days of the observation period, reflecting the user’s current engagement level.


## Output Files

### Processed Data
- `cleaned_filtered_pivot_table.csv`: Preprocessed data ready for analysis

### Analysis Results
- `user_analysis_results.csv`: Contains User ID, Activity Ratio, Inactivity Linearity, and 6 Months Dormancy for all users

## Visualizations

Available chart types:
- **Daily Active Users**: Time series of daily activity
- **Weekly Active Users**: Aggregated weekly patterns
- **Active Days Distribution**: Histogram of user activity levels
- **Inactivity Streaks**: Analysis of dormancy periods
- **Individual Activity Patterns**: User-specific activity heatmaps

## Technical Requirements

- **Python 3.7+**
- **Streamlit**: Web interface framework
- **Polars**: High-performance data processing
- **Pandas**: Data manipulation compatibility
- **Matplotlib/Plotly**: Visualization libraries
