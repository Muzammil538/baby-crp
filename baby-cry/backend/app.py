import os
import numpy as np
import librosa
import librosa.display
from flask import Flask, request, jsonify, send_from_directory # Import send_from_directory
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib # For loading LabelEncoder and StandardScaler
import warnings

# Suppress specific librosa warnings that might appear during feature extraction
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

app = Flask(__name__)
# Enable CORS to allow requests from your frontend (running on a different port/origin)
CORS(app)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads' # Directory to temporarily save uploaded audio files
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the path to your frontend directory
FRONTEND_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend')
# Ensure the frontend folder exists
if not os.path.exists(FRONTEND_FOLDER):
    print(f"Error: Frontend folder not found at {FRONTEND_FOLDER}. Please ensure your project structure is correct.")

# Path to your trained model and preprocessing tools
MODEL_PATH = 'model/best_baby_cry_classifier_model.h5'
LABEL_ENCODER_PATH = 'model/label_encoder.pkl'
SCALER_PATH = 'model/feature_scaler.pkl'

# --- Load trained ML assets ---
try:
    MODEL = load_model(MODEL_PATH)
    print(f"ML model loaded successfully from {MODEL_PATH}!")
except Exception as e:
    print(f"Error loading ML model from {MODEL_PATH}: {e}")
    MODEL = None # Handle case where model fails to load

try:
    LABEL_ENCODER = joblib.load(LABEL_ENCODER_PATH)
    print(f"Label encoder loaded successfully from {LABEL_ENCODER_PATH}!")
except Exception as e:
    print(f"Error loading Label Encoder from {LABEL_ENCODER_PATH}: {e}")
    LABEL_ENCODER = None

try:
    SCALER = joblib.load(SCALER_PATH)
    print(f"Feature scaler loaded successfully from {SCALER_PATH}!")
except Exception as e:
    print(f"Error loading Feature Scaler from {SCALER_PATH}: {e}")
    SCALER = None

# Define the order of features expected by the trained model.
# This MUST match the order of feature columns in your training CSV,
# excluding 'Cry_Audio_File' and 'Cry_Reason'.
FEATURE_COLUMNS_ORDER = [
    'Amplitude_Envelope_Mean', 'RMS_Mean', 'ZCR_Mean', 'STFT_Mean', 'SC_Mean',
    'SBAN_Mean', 'SCON_Mean', 'MFCCs13Mean', 'delMFCCs13', 'del2MFCCs13',
    'MelSpec', 'MFCCs20', 'MFCCs1', 'MFCCs2', 'MFCCs3', 'MFCCs4', 'MFCCs5',
    'MFCCs6', 'MFCCs7', 'MFCCs8', 'MFCCs9', 'MFCCs10', 'MFCCs11', 'MFCCs12', 'MFCCs13'
]

# --- Feature Extraction Function for Live/Uploaded Audio ---
def extract_features_for_live_audio(audio_path, sr=22050):
    """
    Extracts the specified features from a raw audio file.
    This function attempts to replicate the features present in your training CSV.
    The order and type of features extracted here must match the training data.
    """
    try:
        # librosa.load requires soundfile or audioread, and often resampy for resampling.
        # If you encounter "No module named 'resampy'" or similar errors,
        # ensure these libraries are installed:
        # pip install soundfile audioread resampy
        audio, sample_rate = librosa.load(audio_path, sr=sr, res_type='kaiser_fast')

        # Ensure audio is not empty
        if len(audio) == 0:
            return None

        features = {}

        # 1. Temporal Features
        # Amplitude Envelope Mean
        amplitude_envelope = np.abs(audio)
        features['Amplitude_Envelope_Mean'] = np.mean(amplitude_envelope)

        # RMS Mean
        features['RMS_Mean'] = np.mean(librosa.feature.rms(y=audio))

        # ZCR Mean
        features['ZCR_Mean'] = np.mean(librosa.feature.zero_crossing_rate(y=audio))

        # 2. Spectral Features
        # STFT Mean (Mean of magnitude spectrogram)
        stft_result = np.abs(librosa.stft(y=audio))
        features['STFT_Mean'] = np.mean(stft_result)

        # Spectral Centroid Mean
        features['SC_Mean'] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))

        # Spectral Bandwidth Mean
        features['SBAN_Mean'] = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate))

        # Spectral Contrast Mean
        features['SCON_Mean'] = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate))

        # Mel Spectrogram Mean (Mean of Mel-scaled spectrogram)
        melspec = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        features['MelSpec'] = np.mean(melspec)

        # 3. MFCCs and their derivatives
        # MFCCs (up to 20 for MFCCs20, and 13 for individual MFCCs)
        mfccs_all = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)

        # MFCCs13Mean
        features['MFCCs13Mean'] = np.mean(mfccs_all[:13]) # Mean of first 13 MFCCs

        # delMFCCs13 (Mean of delta MFCCs)
        mfccs_delta = librosa.feature.delta(mfccs_all[:13])
        features['delMFCCs13'] = np.mean(mfccs_delta)

        # del2MFCCs13 (Mean of delta-delta MFCCs)
        mfccs_delta2 = librosa.feature.delta(mfccs_all[:13], order=2)
        features['del2MFCCs13'] = np.mean(mfccs_delta2)

        # MFCCs20 (Mean of all 20 MFCCs)
        features['MFCCs20'] = np.mean(mfccs_all)

        # Individual MFCCs 1-13 (Mean of each individual MFCC coefficient)
        for i in range(1, 14): # MFCCs are 0-indexed, so mfccs_all[0] is MFCC1
            features[f'MFCCs{i}'] = np.mean(mfccs_all[i-1])


        # Create a single feature vector in the correct order
        feature_vector = np.array([features[col] for col in FEATURE_COLUMNS_ORDER])

        return feature_vector.reshape(1, -1) # Reshape for single sample prediction
    except Exception as e:
        print(f"Error extracting features from audio: {e}")
        return None

# --- Root URL Route to serve index.html ---
@app.route('/', methods=['GET'])
def home():
    """
    Serves the index.html file from the frontend directory.
    """
    try:
        return send_from_directory(FRONTEND_FOLDER, 'index.html')
    except Exception as e:
        print(f"Error serving index.html: {e}")
        return "Error serving frontend. Please check server logs.", 500


# --- API Endpoint for Cry Classification ---
@app.route('/classify_cry', methods=['POST'])
def classify_cry():
    if not MODEL or not LABEL_ENCODER or not SCALER:
        return jsonify({'error': 'ML model or preprocessing tools not loaded. Check server logs.'}), 500

    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filepath = None
    try:
        # Save the uploaded audio file temporarily
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
        audio_file.save(filepath)

        # 1. Extract features from the uploaded audio
        features = extract_features_for_live_audio(filepath)
        if features is None:
            return jsonify({'error': 'Failed to extract audio features. Audio might be too short or corrupted.'}), 500

        # Ensure the number of extracted features matches the model's input
        expected_features_count = MODEL.input_shape[1]
        if features.shape[1] != expected_features_count:
            return jsonify({
                'error': f'Feature count mismatch. Expected {expected_features_count} features, but got {features.shape[1]}. '
                         'Ensure feature extraction matches training data exactly.'
            }), 500

        # 2. Apply the same scaling used during training
        features_scaled = SCALER.transform(features)

        # 3. Make a prediction using the loaded model
        prediction_probs = MODEL.predict(features_scaled)[0]
        predicted_index = np.argmax(prediction_probs)
        predicted_label = LABEL_ENCODER.inverse_transform([predicted_index])[0]

        return jsonify({'prediction': predicted_label})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'An error occurred during classification: {str(e)}'}), 500
    finally:
        # Clean up the uploaded file
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    # Run the Flask app
    # IMPORTANT: In a production environment, use a more robust WSGI server like Gunicorn.
    app.run(debug=True, port=5000) # Run on port 5000, debug=True for development
