import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import datetime

# Configuration
sequences_file = 'processed_sequences_sampled.npz'
scaler_file = 'trajectory_scaler.pkl'
best_model_path = 'best_lstm_trajectory_model.keras' # Keras 3 preferred format
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Training Hyperparameters
EPOCHS = 30
BATCH_SIZE = 64
PATIENCE = 10      # Patience for EarlyStopping

# Model Architecture Hyperparameters matching the design
LSTM_UNITS_L1 = 64
LSTM_UNITS_L2 = 64
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001

# Load Preprocessed Data and Scaler
print("\nLoading Data and Scaler")
try:
    # Load the .npz file which contains multiple arrays
    data = np.load(sequences_file)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    # Load feature/target names saved during preprocessing
    features_for_rnn_input = data['features'].tolist() if 'features' in data else None
    target_features = data['targets'].tolist() if 'targets' in data else None

    print(f"Loaded data shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    # Determine the number of input features and output dimensions 
    if features_for_rnn_input is None:
        num_features = X_train.shape[2]
        print(f"Inferred num_features: {num_features}")
    else:
        num_features = len(features_for_rnn_input)
        print(f"Input features: {features_for_rnn_input}")

    if target_features is None:
        output_dim = y_train.shape[1]
        print(f"Inferred output_dim (targets): {output_dim}")
    else:
        output_dim = len(target_features)
        print(f"Target features: {target_features}")

    sequence_length = X_train.shape[1]
    print(f"Sequence length: {sequence_length}")

except FileNotFoundError:
    print(f"Error: Could not find the sequences file '{sequences_file}'.")
    print("Please ensure you have run the preprocessing script first.")
    exit()
except Exception as e:
    print(f"Error loading sequences file: {e}")
    exit()
# Load the StandardScaler object saved during preprocessing
try:
    scaler = joblib.load(scaler_file)
    print(f"Scaler loaded from '{scaler_file}'")
    if features_for_rnn_input and hasattr(scaler, 'n_features_in_'):
        if scaler.n_features_in_ != len(features_for_rnn_input):
             print(f"Warning: Scaler expected {scaler.n_features_in_} features, but loaded data description has {len(features_for_rnn_input)}.")
        else:
            print("Scaler feature count matches data description.")
    # Identify target feature indices within the scaler's features
    if target_features and features_for_rnn_input:
         target_indices = [features_for_rnn_input.index(tf) for tf in target_features if tf in features_for_rnn_input]
         if len(target_indices) != len(target_features):
             print(f"Warning: Could not find all target features {target_features} within the scaler's features {features_for_rnn_input}")
             target_indices = list(range(output_dim)) # Fallback assumption
             print(f"Assuming targets correspond to the first {output_dim} scaled features for inverse transform.")
         else:
             print(f"Target features '{target_features}' correspond to indices {target_indices} in the scaler.")
    else:
         print(f"Warning: Feature/target names not available. Assuming targets are the first {output_dim} columns for inverse transform.")
         target_indices = list(range(output_dim))

except FileNotFoundError:
    print(f"Error: Could not find the scaler file '{scaler_file}'.")
    exit()
except Exception as e:
    print(f"Error loading scaler file: {e}")
    exit()

# Building LSTM Model
print("\nBuilding LSTM Model")
model = Sequential(name="Stacked_LSTM_Trajectory_Predictor")
# Define the input shape explicitly using an Input layer 
model.add(Input(shape=(sequence_length, num_features), name="Input_Sequence"))
# Add the first LSTM layer, This layer will output the hidden state for each time step in the sequence
model.add(LSTM(units=LSTM_UNITS_L1, return_sequences=True, name="LSTM_Layer_1"))
# Add the second LSTM layer This layer only outputs the hidden state from the final time step. 
model.add(LSTM(units=LSTM_UNITS_L2, return_sequences=False, name="LSTM_Layer_2"))
# Add a Dropout layer for regularization
model.add(Dropout(rate=DROPOUT_RATE, name="Dropout_Layer"))
model.add(Dense(units=output_dim, activation='linear', name="Output_Coordinates"))

# Compile the Model
print("\nCompiling Model")
# start the Adam optimizer 
optimizer = Adam(learning_rate=LEARNING_RATE)
# Compile the model, specifying the optimizer, loss function, and metrics.
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae']) # MSE for loss, MAE for easier interpretation
model.summary()

# set Callbacks, used for saving the best model, stopping training early
print("\nSetting Up Callbacks")
# best modelsaved based on validation loss
checkpoint_callback = ModelCheckpoint(filepath=best_model_path,
                                      save_best_only=True,
                                      monitor='val_loss',
                                      mode='min',        # Save when val_loss is minimized
                                      verbose=1)

# Stop training early if validation loss doesn't improve
early_stopping_callback = EarlyStopping(monitor='val_loss',
                                        patience=PATIENCE, # Number of epochs with no improvement
                                        mode='min',
                                        verbose=1,
                                        restore_best_weights=True) # Restore weights from the best epoch

callbacks_list = [checkpoint_callback, early_stopping_callback]

# Training the Model
print("\nTraining Model")
print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, Early Stopping Patience: {PATIENCE}")
# Call model.fit() to train the model.
history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_val, y_val), #  validation set used here
                    callbacks=callbacks_list,
                    verbose=1)
print("Training finished")

# After training, load the model weights that achieved the best performance on the validation set 
print(f"\nLoading Best Model from {best_model_path}")
try:
    # Load the entire model (architecture + weights + optimizer state) from the .keras file
    best_model = load_model(best_model_path)
    print("Best model loaded")
except Exception as e:
    print(f"Could not load the best model from {best_model_path}. Using the model from the last epoch. Error: {e}")
    best_model = model # Fallback to the model state at the end of training


# Evaluate the Model on Test Set (Scaled Data)
# This data was completely held out and not seen during training or validation.
print("\nEvaluating Model on Test Set (Scaled Data)")
# returns loss and metrics
test_loss, test_mae = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss (MSE - Scaled): {test_loss:.6f}")
print(f"Test Mean Absolute Error (MAE - Scaled): {test_mae:.6f}")


# Make Predictions and Inverse Transform
# this uses the trained model to make predictions on the test set and then convert the predictions back to the original scale.
print("\nMaking Predictions and Inverse Transforming")
y_pred_scaled = best_model.predict(X_test)

# inverse transform (matching scaler's input shape)
num_scaler_features = scaler.n_features_in_
y_pred_full = np.zeros((y_pred_scaled.shape[0], num_scaler_features), dtype=np.float32)
y_test_full = np.zeros((y_test.shape[0], num_scaler_features), dtype=np.float32)

# Place the predicted/actual target values into the correct columns
# based on target_indices identified earlier
np.put_along_axis(y_pred_full, np.array(target_indices)[np.newaxis, :], y_pred_scaled, axis=1)
np.put_along_axis(y_test_full, np.array(target_indices)[np.newaxis, :], y_test, axis=1)


# Perform the inverse transformation using the scaler object
y_pred_original = scaler.inverse_transform(y_pred_full)[:, target_indices]
y_test_original = scaler.inverse_transform(y_test_full)[:, target_indices]

print("Inverse transformation complete.")
print(f"Sample Scaled Prediction: {y_pred_scaled[0]}")
print(f"Sample Original Prediction: {y_pred_original[0]}")
print(f"Sample Scaled Actual: {y_test[0]}")
print(f"Sample Original Actual: {y_test_original[0]}")


# Evaluate on Original Scale
print("\n--- Evaluating Model on Original Scale (Lat/Lon) ---")
# Calculate metrics on the unscaled (original) coordinate data
rmse_original = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
mae_original = mean_absolute_error(y_test_original, y_pred_original)

print(f"Test Root Mean Squared Error (RMSE - Original Scale): {rmse_original:.6f} (degrees/units)")
print(f"Test Mean Absolute Error (MAE - Original Scale):  {mae_original:.6f} (degrees/units)")

if output_dim == 2 and target_features:
    lat_idx_orig = 0
    lon_idx_orig = 1 
    mae_lat = mean_absolute_error(y_test_original[:, lat_idx_orig], y_pred_original[:, lat_idx_orig])
    mae_lon = mean_absolute_error(y_test_original[:, lon_idx_orig], y_pred_original[:, lon_idx_orig])
    print(f"  MAE Latitude : {mae_lat:.6f} degrees")
    print(f"  MAE Longitude: {mae_lon:.6f} degrees")


# Plot training & validation loss curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE During Training')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png')
print("Training curves saved to 'training_curves.png'")