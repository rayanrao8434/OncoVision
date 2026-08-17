import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import opendatasets as od
import pandas as pd
import glob
import os

print(f"TensorFlow Version: {tf.__version__}")

# Ensure the output directory exists
os.makedirs("models", exist_ok=True)

# 1. Download Dataset via Kaggle API
# This will ask for your Kaggle Username and Kaggle Key
print("Downloading Kaggle Dataset (Breast Histopathology Images)...")
dataset_url = "https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images"
od.download(dataset_url)
print("Download and extraction complete.")

# 2. Gather file paths using Pandas to prevent RAM crashes
print("Gathering image paths...")
# The dataset has folders named by patient ID, then subfolders '0' (Benign) and '1' (Malignant)
dataset_path = "breast-histopathology-images"
if not os.path.exists(dataset_path):
    # Sometimes it extracts into a nested folder
    dataset_path = "breast-histopathology-images/IDC_regular_ps50_idx5"

all_images = glob.glob(f"{dataset_path}/**/*.png", recursive=True)

# Extract class label from the parent directory name ('0' or '1')
labels = [os.path.basename(os.path.dirname(x)) for x in all_images]

df = pd.DataFrame({'filename': all_images, 'class': labels})
print(f"Total images found: {len(df)}")

# To speed up training in this demo, let's take a random 10% subset
# You can change frac=1.0 to train on all 277,000 images!
df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)
print(f"Training on a subset of {len(df)} images to save time.")

# 3. Preprocessing & Augmentation via DataGenerator
IMG_SIZE = 50 # Kaggle IDC patches are 50x50
BATCH_SIZE = 64

datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2, # 80% training, 20% validation
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20
)

print("Preparing data generators...")
train_gen = datagen.flow_from_dataframe(
    dataframe=df,
    x_col='filename',
    y_col='class',
    subset='training',
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='binary',
    batch_size=BATCH_SIZE
)

val_gen = datagen.flow_from_dataframe(
    dataframe=df,
    x_col='filename',
    y_col='class',
    subset='validation',
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='binary',
    batch_size=BATCH_SIZE
)

# 4. Build the CNN Model
print("Building CNN architecture...")
model = models.Sequential([
    layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid') # Binary classification
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 5. Train the Model
print("Starting training...")
EPOCHS = 5 
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# 6. Evaluate and Save
val_loss, val_acc = model.evaluate(val_gen)
print(f"Validation Accuracy: {val_acc*100:.2f}%")

model_path = os.path.join("models", "cancer_detection_cnn.h5")
model.save(model_path)
print(f"Model saved successfully to {model_path}")
