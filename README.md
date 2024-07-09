# DeepFake_Detection
# DeepFake Detection Project

## Overview
The DeepFake Detection Project is designed to identify manipulated media content, specifically images and videos, using advanced machine learning techniques. This system employs a sophisticated model trained on a large dataset to distinguish between real and fake media. It features a Django-based web interface for ease of use, allowing users to upload media files and receive results.

## Features
- **Image Processing**: Identifies whether an uploaded image is real or a DeepFake.
- **Video Processing**: Analyzes video frames to determine if the video content has been manipulated.
- **Blink Detection**: Uses eye blink analysis as an additional metric for detecting DeepFakes in videos.
- **Django Frontend**: Provides an intuitive web interface for media upload and result display.

## Installation
1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/deepfake-detection.git
    cd deepfake-detection
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up Django:**
    ```bash
    python manage.py migrate
    python manage.py runserver
    ```

## Usage
1. **Start the Django server:**
    ```bash
    python manage.py runserver
    ```

2. **Access the web interface:**
    Open your web browser and navigate to `http://127.0.0.1:8000`.

3. **Upload Media:**
    - Upload an image or video to detect if it is a DeepFake.

## Model Training
The project uses the EfficientNetB7 model, fine-tuned on the FaceForensics dataset. The model is trained to perform binary classification, distinguishing between real and fake media. The training process involves data augmentation to enhance the model's robustness.

### Training Script
```python
import scipy
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Assuming you have prepared your dataset and split it into training and validation sets
train_dir = 'C:\\Users\\arman\\Downloads\\Compressed\\deepfake_dataset\\Train'
val_dir = 'C:\\Users\\arman\\Downloads\\Compressed\\deepfake_dataset\\Validation'

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load and prepare data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(600, 600),
    batch_size=32,
    class_mode='binary',  # Since we have two classes, binary classification
    classes=['real', 'fake']  # Specify the class names
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(600, 600),
    batch_size=32,
    class_mode='binary',
    classes=['real', 'fake']
)

# Load EfficientNetB7 with pre-trained ImageNet weights
base_model = EfficientNetB7(weights='imagenet', include_top=False)

# Modify model architecture
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # Additional dense layer
predictions = Dense(1, activation='sigmoid')(x)  # Binary classification, sigmoid activation

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Checkpoint to save the best model
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, mode='max')

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,  # Adjust as needed
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[checkpoint]
)


https://pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
