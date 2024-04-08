# Import necessary libraries
import pandas as pd
import numpy as np
import shelve
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from keras_tuner import RandomSearch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Ensure a clear session - Best practice to avoid potential issues in model training
tf.keras.backend.clear_session()

# Configuration constants
DEBUG, INFO, ERROR = 1, 2, 4
MASK = DEBUG | INFO | ERROR

# MASK toggles for debugging and information
# MASK ^= DEBUG
# MASK ^= INFO
# MASK ^= ERROR

# Load your data (adjust paths as necessary)
assay_list = None
with shelve.open('../DataStore/store') as f:
    assay_list = f['assay_list']

chemical_fingerprint_data = pd.read_csv('../Datasets/fingerprint_filtered_9875.csv').drop(["Unnamed: 0"], axis=1)
chemical_fingerprint_data['fp'] = chemical_fingerprint_data['fp'].apply(lambda x: np.array([int(num) for num in x.strip('[]').split()]))

# Define the model architecture with tunable hyperparameters
def chemnet(hp):
    model = Sequential([
        Input(shape=(157,)),
        *[Sequential([
            Dense(units=hp.Int(f'units_{i}', min_value=16, max_value=128, step=16), activation='relu', kernel_regularizer=l2(hp.Choice(f'l2_{i}', values=[0.001, 0.0001]))),
            Dropout(rate=hp.Float('dropout_{i}', 0.1, 0.5, step=0.1))
          ]) for i in range(hp.Int('num_layers', 1, 2))],
        Dense(1, activation='relu')
    ])
    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])),
                  loss='mean_squared_error',
                  metrics=['mse'])
    return model

# Function to train the model and evaluate using various metrics
def train_chemnet(fingerprint_data, assay):
    if DEBUG & MASK: print("Assay : ", assay, "\n", fingerprint_data['fp'].values)

    X = np.stack(fingerprint_data['fp'].values)
    y = fingerprint_data[assay]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tuner = RandomSearch(chemnet, objective='val_loss', max_trials=10, executions_per_trial=1, directory='keras_tuner_demo', project_name='bioactivity_prediction')
    tuner.search(X_train, y_train, epochs=10, validation_split=0.2)

    best_model = tuner.get_best_models(num_models=1)[0]
    predictions = best_model.predict(X_test).flatten()
    optimal_threshold = find_optimal_threshold(y_test, predictions)

    binary_predictions = (predictions > optimal_threshold).astype(int)
    metrics = {
        'Accuracy': accuracy_score(y_test, binary_predictions),
        'F1 Score': f1_score(y_test, binary_predictions),
        'Precision': precision_score(y_test, binary_predictions),
        'Recall': recall_score(y_test, binary_predictions),
        'AUC Score': roc_auc_score(y_test, predictions)
    }

    if INFO & MASK:
        for metric, value in metrics.items():
            print(f'{metric}: {value}')

def find_optimal_threshold(y_true, y_pred):
    thresholds = np.linspace(0, 1, 101)
    best_threshold = 0
    best_score = 0

    for threshold in thresholds:
        binary_predictions = (y_pred > threshold).astype(int)
        score = f1_score(y_true, binary_predictions)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    if DEBUG & MASK: print(f"Best Threshold: {best_threshold}")
    return best_threshold

# Run the training for one assay as an example
for assay in assay_list:
    filtered_data = chemical_fingerprint_data[pd.notnull(chemical_fingerprint_data[assay])][["fp", assay]]
    train_chemnet(filtered_data, assay)
    break  # Process only one assay for demonstration
