import polars as pl
import numpy as np
from scipy import stats

def get_rfi_matrix(df, user_id):

    activity = df.row(user_id)
    reference_day = activity[-1]
    observation_days = activity[:-1]

    # Extend observation if user is inactive today
    if reference_day == 0:
        observation_days = np.append(observation_days, 0)

    episodes = []
    i = 0
    while i < len(observation_days):
        if observation_days[i] == 0:
            start = i
            while i < len(observation_days) and observation_days[i] == 0:
                i += 1
            duration = i - start
            episodes.append((start, duration))
        else:
            i += 1


    # Separate ongoing inactivity episode if exists
    ongoing_episode = None
    if episodes:
        last_start, last_duration = episodes[-1]
        last_end = last_start + last_duration - 1
        if reference_day == 0 and last_end == len(observation_days) - 1:
            ongoing_episode = (last_start, last_duration)
            episodes = episodes[:-1]  # remove ongoing from previous episodes

    rfi_data = {}

    # Process previous episodes
    for start, duration in episodes:
        last_day_of_episode = start + duration - 1
        recency = len(observation_days) - last_day_of_episode - (1 if reference_day == 0 else 0)

        if duration not in rfi_data:
            rfi_data[duration] = {'F': 1, 'R': recency}
        else:
            rfi_data[duration]['F'] += 1
            rfi_data[duration]['R'] = min(rfi_data[duration]['R'], recency)

    # Prepare rows including ongoing episode separately
    rfi_rows = [
        {'I': duration, 'F': data['F'], 'R': data['R']}
        for duration, data in rfi_data.items()
    ]

    if ongoing_episode:
        dur = ongoing_episode[1]
        rfi_rows.append({'I': dur, 'F': 1, 'R': 0})

    rfi_df = pl.DataFrame(rfi_rows)



    # Handle case where no inactivity episodes are found
    if rfi_df.is_empty():
        total_days = len(observation_days)
        active_days = np.sum(observation_days)
        activity_ratio = active_days / total_days if total_days else 0

        features = (
            activity_ratio,
            0,  # num_episodes
            0,  # avg_recency
            0,  # min_recency
            0,  # avg_relevance
            0,  # max_relevance
            0,  # activity_periodicity_score
            0,  # inactivity_linearity
            0,  # activity_variability
            0,  # inactivity_growth_rate
            0   # recent_activity_density
        )
        return pl.DataFrame(schema={'R': pl.Int64, 'F': pl.Int64, 'I': pl.Int64, 'Relevance': pl.Float64}), features

    # Features
    total_days = len(observation_days)
    active_days = np.sum(observation_days)

    activity_ratio = active_days / total_days if total_days else 0
    num_episodes = rfi_df['F'].sum()

    avg_recency = rfi_df['R'].mean() if num_episodes else 0
    min_recency = rfi_df['R'].min() if num_episodes else 0

    # Calculate Relevance
    #k = 0.1
    k = 1 / 30

    rfi_df = rfi_df.with_columns(
        (pl.col("I") * pl.col("F") *
        np.exp(-k * np.sqrt(pl.col("R") + 1)) *
        (pl.col("R") < 90).cast(pl.Int32) *
        (pl.col("R") != 0).cast(pl.Int32)).round(2).alias("Relevance")
    )

    avg_relevance = rfi_df['Relevance'].mean() if num_episodes else 0
    max_relevance = rfi_df['Relevance'].max() if num_episodes else 0

    # New Feature 1: Activity Periodicity Score
    # Calculate autocorrelation to detect periodicity in activity patterns
    if len(observation_days) > 7:  # Need sufficient data for meaningful autocorrelation
        autocorr = np.correlate(observation_days, observation_days, mode='full')
        # Take the center part (positive lags)
        autocorr = autocorr[len(autocorr)//2:]
        # Normalize
        autocorr = autocorr / np.max(autocorr)
        # Calculate periodicity as the sum of non-zero lag correlations
        activity_periodicity_score = np.sum(autocorr[1:min(30, len(autocorr))]) / min(29, len(autocorr)-1)
    else:
        activity_periodicity_score = 0

    # New Feature 2: Inactivity Linearity (R² Score)
    # Calculate how well inactivity durations correlate with recency values for individual episodes
    if len(episodes) > 1:
        individual_I_values = []
        individual_R_values = []

        for start, duration in episodes:
            last_day_of_episode = start + duration - 1
            recency = len(observation_days) - last_day_of_episode - (1 if reference_day == 0 else 0)
            individual_I_values.append(duration)
            individual_R_values.append(recency)

        # Use linear regression to get R² score between individual I and R values
        _, _, r_value, _, _ = stats.linregress(individual_R_values, individual_I_values)
        inactivity_linearity = r_value ** 2  # R² score
    else:
        inactivity_linearity = 0

    # New Feature 3: Activity Variability
    # Measure variation in gap between active days
    active_indices = np.where(observation_days == 1)[0]
    if len(active_indices) > 1:
        gaps = np.diff(active_indices)
        activity_variability = np.std(gaps)
    else:
        activity_variability = 0

    # New Feature 4: Inactivity Growth Rate
    # Rate at which inactivity duration increases from one episode to the next
    if len(episodes) > 1:
        episode_durations = [duration for _, duration in episodes]
        growth_rates = [episode_durations[i]/episode_durations[i-1] if episode_durations[i-1] > 0 else 1
                         for i in range(1, len(episode_durations))]
        inactivity_growth_rate = np.mean(growth_rates)
    else:
        inactivity_growth_rate = 1

    # New Feature 5: Recent Activity Density
    # Proportion of active days in the most recent 30-day period
    recent_window = min(30, len(observation_days))
    recent_days = observation_days[-recent_window:]
    recent_activity_density = np.sum(recent_days) / recent_window

    features = (
        activity_ratio,
        num_episodes,
        avg_recency,
        min_recency,
        avg_relevance,
        max_relevance,
        activity_periodicity_score,
        inactivity_linearity,
        activity_variability,
        inactivity_growth_rate,
        recent_activity_density
    )

    return rfi_df.sort('R').select(['R', 'F', 'I', 'Relevance']), features


def calculate_dormancy(observation_df, user_id):
    """Calculate 6 months dormancy for an individual user"""
    rfi_matrix, _ = get_rfi_matrix(observation_df, user_id)
    
    if not rfi_matrix.is_empty() and rfi_matrix["Relevance"].sum() > 0:
        weighted_avg_dormancy = np.average(rfi_matrix["I"], weights=rfi_matrix["Relevance"])
    else:
        weighted_avg_dormancy = 0
    
    return round(weighted_avg_dormancy, 0)

def process_all_users(df):
    """Process all users and calculate features and dormancy"""
    first_period_duration = 180
    second_period_duration = 187
    
    observation_df = df.select(df.columns[:first_period_duration])
    test_df = df.select(df.columns[-second_period_duration:])

    
    print("First Period: ", observation_df.shape)
    print("Second Period: ", test_df.shape)
    
    feature_names = [
        "activity_ratio",
        "num_episodes", 
        "avg_recency",
        "min_recency",
        "avg_relevance",
        "max_relevance",
        "activity_periodicity_score",
        "inactivity_linearity",
        "activity_variability",
        "inactivity_growth_rate",
        "recent_activity_density"
    ]
    
    # Initialize empty dataframes
    features_data = []
    dormancy_data = []
    
    user_ids = df.get_column(df.columns[0]).to_list()  # assuming first column contains user_ids
    
    for user_id in user_ids:
        # Calculate dormancy
        dormancy_value = calculate_dormancy(observation_df, user_id, get_rfi_matrix)
        dormancy_data.append({"user_id": user_id, "6_months_dormancy": int(dormancy_value)})
        
        # Calculate features
        _, features = get_rfi_matrix(df, user_id)
        feature_row = {"user_id": user_id}
        for i, feature_name in enumerate(feature_names):
            feature_row[feature_name] = features[i] if i < len(features) else 0
        features_data.append(feature_row)
    
    # Create polars dataframes
    features_df = pl.DataFrame(features_data)
    dormancy_df = pl.DataFrame(dormancy_data)
    
    # Rename feature columns
    features_df = features_df.rename({
        'activity_ratio': 'Activity Ratio',
        'num_episodes': 'Number of Inactivity Episodes', 
        'avg_recency': 'Average Recency',
        'min_recency': 'Minimum Recency',
        'avg_relevance': 'Average Relevance',
        'max_relevance': 'Maximum Relevance',
        'activity_periodicity_score': 'Activity Periodicity Score',
        'inactivity_linearity': 'Inactivity Linearity',
        'activity_variability': 'Activity Variability',
        'inactivity_growth_rate': 'Inactivity Growth Rate',
        'recent_activity_density': 'Recent Activity Density'
    })
    
    # Apply data types and rounding
    int_cols = [
        'Number of Inactivity Episodes',
        'Minimum Recency'
    ]
    float_cols = [
        'Activity Ratio',
        'Average Recency', 
        'Average Relevance',
        'Maximum Relevance',
        'Activity Periodicity Score',
        'Inactivity Linearity',
        'Activity Variability',
        'Inactivity Growth Rate',
        'Recent Activity Density'
    ]
    
    features_df = features_df.with_columns([
        pl.col(col).cast(pl.Int32) for col in int_cols
    ] + [
        pl.col(col).cast(pl.Float64).round(2) for col in float_cols
    ])
    
    return features_df, dormancy_df