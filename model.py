# Import libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Step 1: Load and preprocess data
# Assume `data` is a DataFrame with features and labels
X = data.drop('label', axis=1)  # Feature columns
y = data['label']  # Labels: 0 for normal, 1 for DDoS

# Normalize features for compatibility with models
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 2: Build the Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 3: Build the Autoencoder for anomaly detection
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(16, activation='relu')(input_layer)
encoded = Dense(8, activation='relu')(encoded)
decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test), verbose=1)

# Use reconstruction error for anomaly detection
reconstructions = autoencoder.predict(X_test)
mse = tf.keras.losses.mean_squared_error(X_test, reconstructions)
threshold = mse.numpy().mean() + 2 * mse.numpy().std()  # Set threshold
anomalies = mse > threshold  # Flag anomalies
