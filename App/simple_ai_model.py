import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import json
from PIL import Image

# قراءة بيانات الصور من مجلد الصور
def read_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpeg"):  # التأكد من أن الملف هو بامتداد .jpeg
            image_path = os.path.join(folder_path, filename)
            image = np.array(Image.open(image_path))
            images.append(image)
    return np.array(images)

# قراءة بيانات الوصف من ملف JSON
def read_descriptions(json_path):
    with open(json_path) as json_file:
        data = json.load(json_file)
    descriptions = []
    for item in data:
        descriptions.append(item["description"])
    return descriptions

# تحميل البيانات
images_folder = "images"
json_file = os.path.join("data", "data.json")  # توجيه إلى مجلد البيانات
images = read_images(images_folder)
descriptions = read_descriptions(json_file)

# بناء النموذج
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(5, 5)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(4)  # 4 خرج لأربعة أوصاف مختلفة
])

# تجهيز البيانات
images = images.astype('float32')  # تحويل القيم إلى أعداد عشرية

# تجهيز النموذج
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# تدريب النموذج
model.fit(images, np.array(range(len(descriptions))), epochs=10)

# حفظ النموذج بعد التدريب
model_dir = "App/models"  # مسار المجلد الذي تم إنشاؤه لحفظ النماذج
model_filename = "trained_model.h5"
model.save(os.path.join(model_dir, model_filename))

# توليد الصور من الوصف
def generate_image(description):
    descriptions_mapping = {"A black square": 0, "A white square": 1, "A white circle": 2, "A black circle": 3}
    label = descriptions_mapping[description]
    noise = np.random.normal(0, 1, (1, 100))  # إنشاء بعض الضوضاء كمتغيرات عشوائية
    generated_image = model.predict(noise)  # توليد الصورة باستخدام النموذج
    return generated_image

# استخدام النموذج لتوليد صورة
description = "A black square"  # يمكن استبدالها بأي وصف من بيانات الوصف
generated_image = generate_image(description)

