# DeepFake Detection Project

## Overview
The DeepFake Detection Project is engineered to detect manipulated media content, specifically targeting images and videos, using state-of-the-art machine learning techniques. The core functionality revolves around an eye-blink detection mechanism, leveraging the dlib library for facial landmark matching and calculating Eye Aspect Ratio (EAR) values. The system utilizes a sophisticated model trained on the FaceForensics++ dataset (1.4GB) to differentiate between real and fake media, applicable to both images and videos shorter than 10 seconds (due to the inconsistent blink patterns in shorter videos). The project includes a user-friendly Django-based web interface, enabling users to upload media files and obtain results seamlessly.
- The EyeBlink Detection is based upon the [DeepVision: Deepfakes detection using human eye blinking pattern](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9072088) Paper.

Citation: T. Jung, S. Kim and K. Kim, "DeepVision: Deepfakes Detection Using Human Eye Blinking Pattern," in IEEE Access, vol. 8, pp. 83144-83154, 2020, doi: 10.1109/ACCESS.2020.2988660.
keywords: {Gallium nitride;Detectors;Visualization;Target tracking;Machine learning;Generative adversarial networks;Biology;Cyber security;deep-fake;GANs;deep learning},

<img width="877" alt="image" src="https://github.com/sumithanwate3/DeepFake_Detection/assets/96422074/33c2e6db-e90d-4b46-927b-a226b66f13b3">
<img width="400" alt="image" src="https://github.com/sumithanwate3/DeepFake_Detection/assets/96422074/f6a9de17-6880-480f-9633-decaee7f579f"> <img width="435" alt="image" src="https://github.com/sumithanwate3/DeepFake_Detection/assets/96422074/f3758977-4f71-44cf-b751-5b6c69e77bef">


### DataSet 
- FaceForensics++
FaceForensics++ is a forensics dataset consisting of 1000 original video sequences that have been manipulated with four automated face manipulation methods: Deepfakes, Face2Face, FaceSwap and NeuralTextures. The data has been sourced from 977 youtube videos and all videos contain a trackable mostly frontal face without occlusions which enables automated tampering methods to generate realistic forgeries
  - [Github](https://github.com/ondyari/FaceForensics)
  - [Paper](https://arxiv.org/pdf/2005.05535v5)
  - [EfficientNet with FaceForensics](https://arxiv.org/pdf/2004.07676v1)
 
### Model Training
The project uses the [EfficientNetB7 model](https://keras.io/api/applications/efficientnet/), fine-tuned on the FaceForensics dataset. The model is trained to perform binary classification, distinguishing between real and fake media. The training process involves data augmentation to enhance the model's robustness.


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
train_dir = 'C:\\Users\\sumit\\Downloads\\Compressed\\deepfake_dataset\\Train'
val_dir = 'C:\\Users\\sumit\\Downloads\\Compressed\\deepfake_dataset\\Validation'

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
```

https://pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
