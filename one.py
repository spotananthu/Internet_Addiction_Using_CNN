import numpy as np
import pandas as pd
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from scipy import signal
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers

# Load raw EEG data from CSV files for all 48 subjects
raw_data_files = [f"/home/iiit-kottayam/new/mdp/{i}.csv" for i in range(1, 49)]
raw_data = []

for file in raw_data_files:
    data = pd.read_csv(file)
    raw_data.append(data.values)

# Function to calculate the minimum sample length across subjects' EEG data
def calculate_min_sample_length(data_list):
    return min(data.shape[0] for data in data_list)

# Now 'raw_data' contains all EEG data with varying lengths for each subject

# Define sliding window parameters
window_size = 2000  # Number of sampled data points over 8 seconds
overlap = 500  # Number of sampled data points for 2-second overlap

# Function to create sliding windows with overlap
def create_sliding_windows_2d(data, window_size, overlap):
    windows = []
    start = 0
    end = window_size
    while end <= data.shape[0]:
        windows.append(data[start:end, :])
        start += window_size - overlap
        end = start + window_size
    return windows

# Create sliding windows for each subject's EEG data
sliding_windows_data_2d = []
for data in raw_data:
    windows = create_sliding_windows_2d(data, window_size, overlap)
    sliding_windows_data_2d.append(windows)

# Now 'sliding_windows_data_2d' contains sliding windows for each subject's EEG data


# Perform detrending and standardization
scaler = StandardScaler()
for subject_windows in sliding_windows_data_2d:
    for i, window in enumerate(subject_windows):
        # Detrending
        detrended_window = signal.detrend(window, axis=0)
        # Standardization
        standardized_window = scaler.fit_transform(detrended_window)
        # Store detrended and standardized data back into sliding_windows_data_2d
        subject_windows[i] = standardized_window

        # Print output dimension for each subject
for i, subject_windows in enumerate(sliding_windows_data_2d[:5], 1):
    print(f"Subject {i}:")
    print(f"Output dimension of one sample: {subject_windows[0].shape}")


# Print output dimension for each subject and data inside each window
for i, subject_windows in enumerate(sliding_windows_data_2d[:1], 1):
    print(f"Subject {i}:")
    for j, window in enumerate(subject_windows, 1):
        print(f"Window {j}:")
        print(window)  # Print the data inside the window
        print()  # Print an empty line for readability


# Print the length of each list in psd_data_resampled
for i, subject_windows in enumerate(sliding_windows_data_2d, 1):
    print(f"Subject {i}: Length = {len(subject_windows)}")




# Flatten the sliding windows data
X = np.array([window for subject_windows in sliding_windows_data_2d for window in subject_windows])

# Reshape sliding window data to match input shape
num_channels = X.shape[2]
X_reshaped = X.reshape((-1, window_size, num_channels, 1))

# Assign labels to all samples after reshaping
num_samples = X_reshaped.shape[0]
labels = np.array([0] * (num_samples // 2) + [1] * (num_samples // 2))


# Calculate the number of samples for each label
num_samples_label_0 = X_reshaped.shape[0] // 2
num_samples_label_1 = X_reshaped.shape[0] // 2

print("Number of samples labeled as 0:", num_samples_label_0)
print("Number of samples labeled as 1:", num_samples_label_1)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, labels, test_size=0.2, random_state=42, stratify=labels)


# Check the shape of train and test sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)



def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()

    # Convolutional Layer 1
    model.add(layers.Conv2D(4, (1, 5), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional Layer 2
    model.add(layers.Conv2D(8, (1, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional Layer 3
    model.add(layers.Conv2D(16, (1, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional Layer 4
    model.add(layers.Conv2D(32, (3, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional Layer 5
    model.add(layers.Conv2D(64, (3, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten Layer
    model.add(layers.Flatten())

    # Fully Connected Layer
    model.add(layers.Dense(250, activation='relu'))

    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Define input shape and number of classes
input_shape = (window_size, num_channels, 1)  # Assuming input data shape is (2000, 37, 1)
num_classes = 2  # Number of classes (0 and 1)

# Create CNN model
model = create_cnn_model(input_shape, num_classes)

# Display model summary
model.summary()


# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=8, batch_size=32, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy: {:.2f}%".format(accuracy * 100))

import matplotlib.pyplot as plt
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy: {:.2f}%".format(accuracy * 100))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Get predictions on the test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Convert one-hot encoded labels to class labels
y_true_classes = y_test

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

# Display confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
