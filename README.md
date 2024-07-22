# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Download and Extract Dataset
import zipfile
import os

# Define the path to the zip file in Google Drive
zip_path = '/content/drive/MyDrive/archive(2).zip'
extract_path = '/content/dataset'

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Import Libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Ensure reproducibility
tf.random.set_seed(42)
np.random.set_seed(42)

# Load and Preprocess Dataset
def load_data(data_dir):
    images = []
    labels = []
    class_names = os.listdir(data_dir)
    class_indices = {name: idx for idx, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if os.path.isfile(img_path):  # Check if it's a file
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
                img = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img)
                labels.append(class_indices[class_name])
    
    images = np.array(images)
    labels = np.array(labels)
    
    return train_test_split(images, labels, test_size=0.2, random_state=42), class_indices

(train_images, test_images, train_labels, test_labels), class_indices = load_data('/content/dataset/Sports-celebrity images')

num_classes = len(class_indices)

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

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(train_images, train_labels, batch_size=32)
test_generator = test_datagen.flow(test_images, test_labels, batch_size=32)

# Build and Compile the Model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the Model
history = model.fit(train_generator, epochs=10, validation_data=test_generator)

# Save the Model
model.save('/content/drive/MyDrive/sports_celebrity_classifier.h5')

# Evaluate the Model
test_loss, test_acc = model.evaluate(test_generator)

print(f'Test accuracy: {test_acc}')

# Plot training & validation accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, label='Training accuracy')
plt.plot(epochs, val_acc, label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

