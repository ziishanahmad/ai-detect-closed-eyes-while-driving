# AI Detect Closed Eyes While Driving

A deep learning project to detect closed eyes while driving using a Convolutional Neural Network (CNN).

## Description

This project trains a Convolutional Neural Network (CNN) to detect whether a driver's eyes are open or closed. The dataset includes images of open and closed eyes, and the model is trained to classify these images accurately.

## Dataset

The dataset used for this project is included in the repository as a zip file named `eyeopenclose.zip`. It contains images of open and closed eyes organized into training and testing sets.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.x
- TensorFlow
- OpenCV
- Matplotlib

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/ziishanahmad/ai-detect-closed-eyes-while-driving.git
   cd ai-detect-closed-eyes-while-driving
   ```

2. Extract the dataset:
   ```python
   import zipfile
   import os

   # Path to the uploaded dataset zip file
   zip_file_path = 'eyeopenclose.zip'
   extract_path = 'eyeopenclose-dataset'

   # Extract the zip file
   with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
       zip_ref.extractall(extract_path)

   print(f'Extracted to {extract_path}')
   ```

### Training the Model

1. Run the following script to train the model and save it:
   ```python
   import tensorflow as tf
   from tensorflow.keras.preprocessing.image import ImageDataGenerator
   from tensorflow.keras.models import Sequential, load_model
   from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
   import cv2
   import numpy as np
   import matplotlib.pyplot as plt

   # Load and preprocess the dataset
   train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

   train_generator = train_datagen.flow_from_directory(
       'eyeopenclose-dataset/dataset/train', 
       target_size=(24, 24), 
       color_mode='grayscale', 
       batch_size=32, 
       class_mode='binary',
       subset='training')

   validation_generator = train_datagen.flow_from_directory(
       'eyeopenclose-dataset/dataset/train', 
       target_size=(24, 24), 
       color_mode='grayscale', 
       batch_size=32, 
       class_mode='binary',
       subset='validation')

   # Build the CNN model
   model = Sequential([
       Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)),
       MaxPooling2D(2, 2),
       Conv2D(64, (3, 3), activation='relu'),
       MaxPooling2D(2, 2),
       Flatten(),
       Dense(128, activation='relu'),
       Dropout(0.5),
       Dense(1, activation='sigmoid')
   ])

   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

   # Train the model
   history = model.fit(
       train_generator,
       validation_data=validation_generator,
       epochs=10
   )

   # Save the model
   model.save('open_closed_eye_model.h5')
   ```

### Testing the Model

1. Test the trained model using sample images:
   ```python
   import cv2
   import numpy as np
   from tensorflow.keras.models import load_model
   import matplotlib.pyplot as plt

   # Load the model
   model = load_model('open_closed_eye_model.h5')

   # Function to preprocess the test image
   def preprocess_test_image(image_path):
       img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
       img = cv2.resize(img, (24, 24))  # Resize to the input size of the model
       img = img / 255.0  # Normalize
       img = np.expand_dims(img, axis=0)  # Add batch dimension
       img = np.expand_dims(img, axis=-1)  # Add channel dimension
       return img

   # Test the model with sample images
   test_image_paths = [
       'eyes-open.jpg',
       'eyes-closed.jpg'
   ]

   for image_path in test_image_paths:
       img = preprocess_test_image(image_path)
       prediction = model.predict(img)
       state = 'Open' if prediction[0] > 0.5 else 'Closed'
       
       # Load and display the image with prediction
       img_display = cv2.imread(image_path)
       plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
       plt.title(f'Predicted: {state}')
       plt.axis('off')
       plt.show()
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

- **Name**: Zeeshan Ahmad
- **Email**: ziishanahmad@gmail.com
- **LinkedIn**: [Zeeshan Ahmad](https://www.linkedin.com/in/ziishanahmad/)
- **GitHub**: [Zeeshan Ahmad](https://github.com/ziishanahmad)

## License

Distributed under the MIT License. See `LICENSE` for more information.
