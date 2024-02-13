import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import json
from PIL import Image

# Function to read images from a folder
def read_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpeg"):  # Ensure the file has .jpeg extension
            image_path = os.path.join(folder_path, filename)
            image = np.array(Image.open(image_path))
            images.append(image)
    return np.array(images)

# Function to read descriptions from a JSON file
def read_descriptions(json_path):
    with open(json_path) as json_file:
        data = json.load(json_file)
    descriptions = []
    for item in data:
        descriptions.append(item["description"])
    return descriptions

# Load data
images_folder = "images"
json_file = os.path.join("data", "data.json")  # Point to the data folder
images = read_images(images_folder)
descriptions = read_descriptions(json_file)

# Build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(5, 5)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(4)  # 4 outputs for four different descriptions
])

# Prepare data
images = images.astype('float32')  # Convert values to float
# Prepare the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(images, np.array(range(len(descriptions))), epochs=10)

# Save the trained model
model_dir = "App/models"  # Path to the folder created to save models
model_filename = "trained_model.h5"
model.save(os.path.join(model_dir, model_filename))

# Function to generate images from descriptions
def generate_image(description):
    descriptions_mapping = {"A black square": 0, "A white square": 1, "A white circle": 2, "A black circle": 3}
    label = descriptions_mapping[description]
    noise = np.random.normal(0, 1, (1, 100))  # Generate some noise as random variables
    generated_image = model.predict(noise)  # Generate image using the model
    return generated_image

# Using the model to generate an image
description = "A black square"  # Can be replaced with any description from the data
generated_image = generate_image(description)
