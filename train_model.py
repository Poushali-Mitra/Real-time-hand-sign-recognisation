
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image


def load_data(image_dir, landmark_dir):
    images = []
    landmarks = []
    labels = []

    for label_idx, label in enumerate(sorted(os.listdir(image_dir))):
        label_path = os.path.join(image_dir, label)
        landmark_path = os.path.join(landmark_dir, label)

        if os.path.isdir(label_path):
            image_files = [f for f in os.listdir(label_path) if f.endswith(".png")]
            for file_name in sorted(image_files):
                
                img_path = os.path.join(label_path, file_name)
                img = Image.open(img_path).convert("RGB").resize((64, 64))
                images.append(np.array(img))

                landmark_file = file_name.replace(".png", "_landmarks.npy")
                landmark_path_full = os.path.join(landmark_path, landmark_file)
                try:
                    landmarks.append(np.load(landmark_path_full))
                except FileNotFoundError:
                    landmarks.append(np.zeros((21, 2)))  

                
                labels.append(label_idx)


    max_landmarks = max(len(landmark) for landmark in landmarks) if landmarks else 0
    landmarks = [
        np.pad(landmark, ((0, max_landmarks - len(landmark)), (0, 0)), mode="constant") if len(landmark) < max_landmarks else landmark
        for landmark in landmarks
    ]

    return np.array(images), np.array(landmarks, dtype=np.float32), np.array(labels)


image_dir = "dataset_3"
landmark_dir = "dataset_3"


X_images, X_landmarks, y = load_data(image_dir, landmark_dir)


X_images = X_images / 255.0


landmark_max = max((np.max(landmark) for landmark in X_landmarks if len(landmark) > 0 and np.max(landmark) > 0), default=1)
X_landmarks = X_landmarks / landmark_max if landmark_max > 0 else X_landmarks


X_img_train, X_img_test, X_land_train, X_land_test, y_train, y_test = train_test_split(
    X_images, X_landmarks, y, test_size=0.2, random_state=42
)


num_classes = len(np.unique(y))
def to_one_hot(labels, num_classes):
    return np.eye(num_classes)[labels]

y_train = to_one_hot(y_train, num_classes)
y_test = to_one_hot(y_test, num_classes)


image_input = tf.keras.Input(shape=(64, 64, 3), name="image_input")
x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(image_input)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu")(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Flatten()(x)


landmark_input = tf.keras.Input(shape=(X_landmarks.shape[1], X_landmarks.shape[2]), name="landmark_input")
y = tf.keras.layers.Flatten()(landmark_input)
y = tf.keras.layers.Dense(128, activation="relu")(y)
y = tf.keras.layers.Dense(64, activation="relu")(y)


combined = tf.keras.layers.concatenate([x, y])
z = tf.keras.layers.Dense(128, activation="relu")(combined)
z = tf.keras.layers.Dropout(0.5)(z)
z = tf.keras.layers.Dense(num_classes, activation="softmax")(z)


model = tf.keras.Model(inputs=[image_input, landmark_input], outputs=z)


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


history = model.fit(
    [X_img_train, X_land_train], y_train,
    epochs=100,
    batch_size=32,
    validation_data=([X_img_test, X_land_test], y_test)
)


y_pred = model.predict([X_img_test, X_land_test])
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)


print(classification_report(y_test_classes, y_pred_classes, target_names=[chr(i + 65) for i in range(num_classes)]))

confusion_mtx = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
plt.imshow(confusion_mtx, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, [chr(i + 65) for i in range(num_classes)], rotation=45)
plt.yticks(tick_marks, [chr(i + 65) for i in range(num_classes)])
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()


model.save("hand_sign_landmark_model.h5")
