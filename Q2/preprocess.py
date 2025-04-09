import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import gc

# Config 
input_csv_path = 'flights_data.csv'
sampling_fraction = 0.10  # Use % of data to be used
chunk_size = 500_000     # chunked loading based on RAM requirements
min_trajectory_length = 11 # Sequence length (10) + 1 target point
SEQUENCE_LENGTH = 10       # Input sequence length for RNN
random_seed = 1073           # For reproducibility

np.random.seed(random_seed)

# Unique Aircraft IDs (Memory constraint)
print(f"Reading unique 'icao24' IDs from {input_csv_path}")
all_icao = None
try:
    # loading just the identifier column
    print("  Attemp to load 'icao24' column directly")
    df_ids = pd.read_csv(input_csv_path, usecols=['icao24'])
    all_icao = df_ids['icao24'].unique()
    del df_ids # Free memory
    gc.collect() # request garbage collection
    print(f"loaded unique IDs directly.")
except MemoryError:
    print("loading failed (MemoryError). Switching to chunked reading for IDs")
    unique_icao_set = set()
    for chunk in pd.read_csv(input_csv_path, usecols=['icao24'], chunksize=chunk_size):
        unique_icao_set.update(chunk['icao24'].dropna().unique())
    all_icao = np.array(list(unique_icao_set))
    del unique_icao_set # Free memory
    gc.collect()
    print(f"  Successfully extracted unique IDs using chunking.")

if all_icao is None or len(all_icao) == 0:
    print("Error: No 'icao24' IDs were loaded. Check the CSV file and column name.")
    exit()

print(f"Found {len(all_icao)} unique aircraft IDs.")

# Sample Aircraft IDs
print("\nSampling Aircraft IDs")
num_to_select = int(len(all_icao) * sampling_fraction)
if num_to_select == 0 and len(all_icao) > 0:
    num_to_select = 1 # Ensure at least one is selected if possible
selected_icao = np.random.choice(all_icao, size=num_to_select, replace=False)

print(f"Selected {len(selected_icao)} aircraft IDs for processing")
del all_icao # Free memory
gc.collect()

# Load Data ONLY for Selected Aircraft
print(f"\nLoading data for {len(selected_icao)} selected aircraft using chunksize={chunk_size}...")
df_list = []
selected_icao_set = set(selected_icao)
del selected_icao
gc.collect()

try:
    chunk_count = 0
    for chunk in pd.read_csv(input_csv_path, chunksize=chunk_size):
        chunk_count += 1
        # Filter each chunk
        filtered_chunk = chunk[chunk['icao24'].isin(selected_icao_set)]
        if not filtered_chunk.empty:
            df_list.append(filtered_chunk)
        if chunk_count % 5 == 0: # Print progress every 5 chunks
             print(f"  Processed {chunk_count * chunk_size / 1e6:.1f}M rows")

    if not df_list:
        print("No data found for the selected aircraft IDs.")
        exit()

    df = pd.concat(df_list, ignore_index=True)
    del df_list # Free memory
    gc.collect()
    print(f"loaded data for selected aircraft")
    print(f"Initial sampled data shape: {df.shape}")
    print("Sampled data columns:", df.columns.tolist())

except FileNotFoundError:
    print(f"Input file not found at {input_csv_path} during chunked loading.")
    exit()
except Exception as e:
    print(f"Error during chunked data loading: {e}")
    exit()

# Optimize Data Types to Sampled Data
print("\nOptimizing data types for memory efficiency")
original_memory = df.memory_usage(deep=True).sum() / (1024**2) # Memory in MB

numeric_columns = ['lat', 'lon', 'velocity', 'heading', 'baroaltitude', 'geoaltitude']
# Convert to numeric first, coercing errors, then downcast floats
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].astype(np.float32) # Downcast to float32

if 'time' in df.columns:
    df['time'] = pd.to_numeric(df['time'], errors='coerce', downcast='integer')

# Convert identifiers to category
id_cols = ['icao24', 'callsign']
for col in id_cols:
     if col in df.columns:
         df[col] = df[col].astype(str).astype('category')

optimized_memory = df.memory_usage(deep=True).sum() / (1024**2)
print(f"Memory Usage Before Optimization: {original_memory:.2f} MB")
print(f"Memory Usage After Optimization:  {optimized_memory:.2f} MB")
print("Data types after optimization:\n", df.dtypes)
gc.collect()

# Initial Inspection & Basic Cleansing on Sampled Data
print("\nBasic Cleaning and Timestamp Conversion")
# Convert time to datetime objects
if 'time' in df.columns:
    df['timestamp'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True) # Drop rows where timestamp conversion failed
    print("Converted 'time' to 'timestamp' (datetime).")
else:
    print("Warning: 'time' column not found for timestamp conversion.")
    # If time is missing, sorting and sequence logic will fail later, so exit
    exit()

# Check for exact duplicate rows
initial_rows = len(df)
df.drop_duplicates(inplace=True)
print(f"Dropped {initial_rows - len(df)} exact duplicate rows.")

# Check for invalid lat/lon/velocity (float32 columns now)
valid_rows = len(df)
df = df[(df['lat'] <= 90.0) & (df['lat'] >= -90.0)]
df = df[(df['lon'] <= 180.0) & (df['lon'] >= -180.0)]
if 'velocity' in df.columns:
    df = df[df['velocity'] >= 0.0] # Velocity should never be negative
print(f"Dropped {valid_rows - len(df)} rows with invalid lat/lon/velocity values.")
print(f"Data shape after basic cleaning: {df.shape}")
gc.collect()

# Feature Selection/Engineering
print("\n Feature Engineering (Calculating Time Delta)")
# Define features for the model and sorting/grouping
# need 'icao24' and 'timestamp' for processing steps
core_features = ['lat', 'lon', 'velocity', 'heading', 'geoaltitude']
target_features = ['lat', 'lon']
features_for_rnn_input = core_features.copy()

all_needed_cols = ['icao24', 'timestamp'] + core_features
df_processed = df[all_needed_cols].copy()
del df # Free memory of original sampled df
gc.collect()

# Calculate time difference
df_processed.sort_values(by=['icao24', 'timestamp'], inplace=True)
df_processed['time_delta'] = df_processed.groupby('icao24')['timestamp'].diff().dt.total_seconds()
df_processed['time_delta'] = df_processed['time_delta'].astype(np.float32)
# Add time_delta to the features list for the RNN
features_for_rnn_input.append('time_delta')
print("Calculated 'time_delta' between consecutive points for each aircraft.")

# Handling Missing Values
print("\nandling Missing Values")
print("Missing values before handling:\n", df_processed.isnull().sum())

# Fill time_delta NaN (first point for each aircraft) with 0
df_processed['time_delta'].fillna(0.0, inplace=True)

# Impute other features using linear interpolation within each trajectory
numeric_cols_to_impute = core_features # Impute the numeric features
print(f"Interpolating NaNs within groups for: {numeric_cols_to_impute}")
for col in numeric_cols_to_impute:
   if col in df_processed.columns:
        df_processed[col] = df_processed.groupby('icao24')[col].transform(lambda group: group.interpolate(method='linear', limit_direction='both'))

print("\nMissing values after interpolation:")
print(df_processed.isnull().sum())
initial_rows = len(df_processed)
# Drop rows where essential features might still be NaN (start/end of traj if whole segment missing)
df_processed.dropna(subset=numeric_cols_to_impute, inplace=True)
print(f"Dropped {initial_rows - len(df_processed)} rows with remaining NaNs after interpolation.")
gc.collect()

# Filter Short Trajectories
print("\nFiltering out trajectories shorter than minimum length")
print(f"Minimum trajectory length required: {min_trajectory_length} (Sequence: {SEQUENCE_LENGTH} + Target: 1)")
trajectory_lengths = df_processed.groupby('icao24').size()
valid_icao = trajectory_lengths[trajectory_lengths >= min_trajectory_length].index
initial_rows = len(df_processed)
df_processed = df_processed[df_processed['icao24'].isin(valid_icao)].copy()
print(f"Filtered out {len(trajectory_lengths) - len(valid_icao)} aircraft with trajectories shorter than {min_trajectory_length} points.")
print(f"Removed {initial_rows - len(df_processed)} data points belonging to short trajectories.")
print(f"Data shape after filtering short trajectories: {df_processed.shape}")
del trajectory_lengths, valid_icao # Free memory
gc.collect()

if df_processed.empty:
    print("No trajectories remaining after filtering for minimum length)")
    exit()

# Data Splitting (Train/Validation/Test by Aircraft ID)
print("\nSplitting data into Train/Validation/Test sets by Aircraft ID")
# Get unique aircraft IDs remaining in the processed data
remaining_icao = df_processed['icao24'].unique()

# Split IDs into Train (70%), Validation (15%), Test (15%)
splitter_train_temp = GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=random_seed)
train_idx, temp_idx = next(splitter_train_temp.split(df_processed, groups=df_processed['icao24']))

train_data = df_processed.iloc[train_idx].copy()
temp_data = df_processed.iloc[temp_idx].copy()

splitter_val_test = GroupShuffleSplit(test_size=0.5, n_splits=1, random_state=random_seed)
val_idx, test_idx = next(splitter_val_test.split(temp_data, groups=temp_data['icao24']))

val_data = temp_data.iloc[val_idx].copy()
test_data = temp_data.iloc[test_idx].copy()

del df_processed, temp_data, train_idx, temp_idx, val_idx, test_idx # Free memory
gc.collect()

train_icao = train_data['icao24'].unique()
val_icao = val_data['icao24'].unique()
test_icao = test_data['icao24'].unique()

print(f"Data Split Results:")
print(f"  Train set: {len(train_data)} points, {len(train_icao)} aircraft")
print(f"  Validation set: {len(val_data)} points, {len(val_icao)} aircraft")
print(f"  Test set: {len(test_data)} points, {len(test_icao)} aircraft")
assert len(set(train_icao) & set(val_icao)) == 0
assert len(set(train_icao) & set(test_icao)) == 0
assert len(set(val_icao) & set(test_icao)) == 0

# Data Normalization/Scaling 
print(f"\nScaling features using StandardScaler (fitted on Train data)")
print(f"Features to scale: {features_for_rnn_input}")

scaler = StandardScaler()

# Fit scaler only on training data
scaler.fit(train_data[features_for_rnn_input])

# Apply fitted scaler to all sets
train_data.loc[:, features_for_rnn_input] = scaler.transform(train_data[features_for_rnn_input])
val_data.loc[:, features_for_rnn_input] = scaler.transform(val_data[features_for_rnn_input])
test_data.loc[:, features_for_rnn_input] = scaler.transform(test_data[features_for_rnn_input])

print("Scaling complete.")
# Save the scaler
scaler_filename = 'trajectory_scaler.pkl'
joblib.dump(scaler, scaler_filename)
print(f"Scaler saved to {scaler_filename}")

# Data Reshaping (Sequence Generation)
print("\nGenerating sequences for RNN input")

def create_sequences(data, sequence_length, feature_cols, target_cols):
    X, y = [], []
    # Ensure target columns exist in the scaled data for extraction
    local_target_cols = [col for col in target_cols if col in data.columns]
    print(f"  Generating sequences with length {sequence_length}...")
    print(f"  Input features ({len(feature_cols)}): {feature_cols}")
    print(f"  Target features ({len(local_target_cols)}): {local_target_cols}")

    # Ensure data is sorted correctly before creating sequences
    data = data.sort_values(by=['icao24', 'timestamp'])
    grouped_data = data.groupby('icao24')
    total_sequences = 0

    for name, group in grouped_data:
        features = group[feature_cols].values
        targets = group[local_target_cols].values # Extract targets

        if len(group) >= sequence_length + 1:
            for i in range(len(group) - sequence_length):
                seq_end = i + sequence_length
                target_idx = seq_end
                X.append(features[i:seq_end])
                y.append(targets[target_idx])
                total_sequences += 1

    print(f"  Generated {total_sequences} sequences.")
    # Ensure output is float32 to save memory
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# Generate sequences for each dataset
# predict the *scaled* lat/lon
X_train, y_train = create_sequences(train_data, SEQUENCE_LENGTH, features_for_rnn_input, target_features)
X_val, y_val = create_sequences(val_data, SEQUENCE_LENGTH, features_for_rnn_input, target_features)
X_test, y_test = create_sequences(test_data, SEQUENCE_LENGTH, features_for_rnn_input, target_features)

del train_data, val_data, test_data # Free memory
gc.collect()

print("\nSequence Generation Complete:")
print(f"  X_train shape: {X_train.shape if 'X_train' in locals() else 'Not generated'}")
print(f"  y_train shape: {y_train.shape if 'y_train' in locals() else 'Not generated'}")
print(f"  X_val shape: {X_val.shape if 'X_val' in locals() else 'Not generated'}")
print(f"  y_val shape: {y_val.shape if 'y_val' in locals() else 'Not generated'}")
print(f"  X_test shape: {X_test.shape if 'X_test' in locals() else 'Not generated'}")
print(f"  y_test shape: {y_test.shape if 'y_test' in locals() else 'Not generated'}")

# Save processed sequences
sequences_filename = 'processed_sequences_sampled.npz'
np.savez_compressed(sequences_filename,
                     X_train=X_train, y_train=y_train,
                     X_val=X_val, y_val=y_val,
                     X_test=X_test, y_test=y_test,
                     features=features_for_rnn_input, # Save feature names
                     targets=target_features)         # Save target names
print(f"Processed sequences saved to {sequences_filename}")

print("\nPreprocessing complete.")
print(f"Final Train sequences: X={X_train.shape}, y={y_train.shape}")