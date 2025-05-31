import os
import numpy as np
import librosa
import librosa.display
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import joblib # For saving the LabelEncoder

# --- Configuration ---
# Path to your dataset CSV file.
# IMPORTANT: Replace 'data/donateacry_features.csv' with the actual path to your CSV.
# This CSV is expected to contain pre-extracted features and a 'Cry_Reason' column for labels.
DATASET_PATH = './baby_cry_dataset/donateacry-corpus_features_final.csv'

# Mapping for numerical labels to human-readable cry reasons as per donateacry-corpus
CRY_LABELS_MAP = {
    0: 'belly pain',
    1: 'burping',
    2: 'discomfort',
    3: 'hungry',
    4: 'tired'
}

# These parameters are for the `extract_features_for_live_audio` function, which will be used
# in your Flask app for new audio. They are NOT directly used for loading
# features from your pre-processed CSV for training.
N_MFCC = 40
MAX_PAD_LEN = 174 # Adjust based on typical cry duration if you were extracting raw MFCCs

# --- 1. Feature Extraction Function (for new audio in Flask app) ---
# This function is kept here for reference and will be used in your app.py.
# It's not directly used for loading the training data from the CSV.
# IMPORTANT: For accurate live classification with a model trained on your CSV's diverse features,
# this function MUST be expanded to extract ALL 26 features (Amplitude_Envelope_Mean, RMS_Mean, ZCR_Mean,
# STFT_Mean, SC_Mean, SBAN_Mean, SCON_Mean, MFCCs13Mean, delMFCCs13, del2MFCCs13, MelSpec, MFCCs20,
# and individual MFCCs 1-13) from the raw audio, in the same order and format as in your CSV.
# Currently, it only extracts MFCCs.
def extract_features_for_live_audio(audio_path, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN):
    """
    Extracts MFCC features from an audio file for live/uploaded audio.
    Pads or truncates the MFCC sequence to a fixed length.
    This function's output shape must match the input shape expected by your trained model.
    If your model is trained on flattened features (as assumed for this CSV),
    you'll need to flatten the MFCCs here too.
    """
    try:
        audio, sample_rate = librosa.load(audio_path, sr=22050, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

        if mfccs.shape[1] > max_pad_len:
            mfccs = mfccs[:, :max_pad_len]
        else:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

        # If your model expects a flattened 1D array of features, flatten the MFCCs here.
        # This assumes the model will be trained on a flat feature vector from the CSV.
        # You might need to adjust this based on the exact features in your CSV.
        flattened_mfccs = mfccs.flatten()
        return flattened_mfccs
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# --- 2. Load Dataset and Extract Features from CSV ---
def load_dataset_from_csv(csv_path, label_col='Cry_Reason'): # Updated label_col
    """
    Loads features and labels directly from a CSV file.

    Args:
        csv_path (str): Path to the CSV file.
        label_col (str): Name of the column containing numerical labels (0-4).

    Returns:
        tuple: (np.array of features, np.array of corresponding string labels)
    """
    print(f"Loading dataset from CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return np.array([]), np.array([])
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return np.array([]), np.array([])

    if label_col not in df.columns:
        print(f"Error: CSV must contain '{label_col}' column.")
        return np.array([]), np.array([])

    # Separate features (X) and labels (y)
    # Exclude 'Cry_Audio_File' and the 'label_col' from features
    feature_cols = [col for col in df.columns if col not in [label_col, 'Cry_Audio_File']]
    X = df[feature_cols].values # Convert features to a NumPy array
    y_numerical = df[label_col].values # Get numerical labels

    # Map numerical labels to string labels
    y_string = np.array([CRY_LABELS_MAP.get(num_label, 'unknown') for num_label in y_numerical])

    print(f"Finished loading dataset. Found {len(X)} samples with {X.shape[1]} features each.")
    return X, y_string

# Load the dataset using the new function
X, y = load_dataset_from_csv(DATASET_PATH, label_col='Cry_Reason') # Explicitly set label_col

# Check if any data was loaded
if len(X) == 0:
    print("No data found or processed. Please check DATASET_PATH, CSV columns, and file content.")
    exit()

# Define the labels (categories) for your baby cries dynamically from the loaded data.
# This ensures CRY_LABELS matches the actual labels in your dataset.
CRY_LABELS = sorted(list(np.unique(y)))
print(f"Detected cry labels: {CRY_LABELS}")

# --- 3. Preprocessing Labels and Features ---
# Encode string labels to numerical values (0, 1, 2, ...)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
# Convert numerical labels to one-hot encoded vectors (e.g., 0 -> [1,0,0,0,0])
y_categorical = to_categorical(y_encoded, num_classes=len(CRY_LABELS))

# Standardize features (important for neural networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
# Using stratify ensures that the proportion of each class is maintained in both splits
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# --- 4. Define the Dense Network Model ---
# This model is suitable for flat feature vectors.
def create_dense_model(input_shape, num_classes):
    model = Sequential([
        # Input layer expects a 1D feature vector
        Dense(256, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3), # Dropout for regularization

        Dense(128, activation='relu'),
        Dropout(0.3),

        Dense(64, activation='relu'),
        Dropout(0.3),

        # Output layer with softmax for multi-class classification
        Dense(num_classes, activation='softmax')
    ])
    return model

# Get input shape from our data (number of features)
input_shape = X_train.shape[1]
num_classes = len(CRY_LABELS)

# Create the model
model = create_dense_model(input_shape, num_classes)

# Compile the model
# Adam optimizer is a good default choice
# Categorical crossentropy for multi-class classification
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary() # Print a summary of the model architecture

# --- 5. Train the Model ---
# Define callbacks for better training control
# EarlyStopping: Stop training if validation accuracy doesn't improve for 'patience' epochs
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
# ModelCheckpoint: Save the best model based on validation accuracy
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'model/best_baby_cry_classifier_model.h5', # Path to save the model
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Train the model
history = model.fit(X_train, y_train,
                    epochs=200, # Increased epochs, but EarlyStopping will prevent overfitting
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping, model_checkpoint],
                    verbose=1)

# --- 6. Evaluate the Model ---
print("\n--- Model Evaluation ---")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 7. Save Label Encoder and StandardScaler for later use in app.py ---
# These are important to correctly preprocess new data and convert predictions back to labels.
joblib.dump(le, 'model/label_encoder.pkl')
joblib.dump(scaler, 'model/feature_scaler.pkl') # Save the scaler too!
print("Label encoder saved to model/label_encoder.pkl")
print("Feature scaler saved to model/feature_scaler.pkl")
print("Model saved to model/best_baby_cry_classifier_model.h5")
