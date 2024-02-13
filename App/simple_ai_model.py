import numpy as np
import tensorflow as tf
from tensorflow import keras

# مثال بسيط لتدريب النموذج على صور مع الأوصاف
# سيتم استخدام مصفوفة بسيطة لتمثيل الصور

# بيانات الصور
images = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
])

# بيانات الأوصاف المطابقة لكل صورة
descriptions = ["A black square", "A white square", "A white circle", "A black circle"]

# بناء النموذج
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(5, 5)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(4)  # 4 خرج لأربعة أوصاف مختلفة
])

# تجهيز البيانات
images = images.reshape(-1, 5, 5)  # إعادة تشكيل الصور
images = images.astype('float32')  # تحويل القيم إلى أعداد عشرية

# تجهيز النموذج
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# تدريب النموذج
model.fit(images, np.array([0, 1, 2, 3]), epochs=10)

# توليد الصور من الوصف
def generate_image(description):
    descriptions_mapping = {"A black square": 0, "A white square": 1, "A white circle": 2, "A black circle": 3}
    label = descriptions_mapping[description]
    noise = np.random.normal(0, 1, (1, 100))  # إنشاء بعض الضوضاء كمتغيرات عشوائية
    generated_image = model.predict(noise)  # توليد الصورة باستخدام النموذج
    return generated_image

# استخدام النموذج لتوليد صورة
description = "A black square"
generated_image = generate_image(description)