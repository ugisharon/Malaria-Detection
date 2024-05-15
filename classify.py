import os
import numpy as np
from PIL import Image
from skimage import io, transform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data_dir = r"C:\Users\user\Desktop\MARTHA CEMA TEST\source\cell_images"
parasitized_dir = r"C:\Users\user\Desktop\MARTHA CEMA TEST\source\cell_images\Parasitized"
uninfected_dir = r"C:\Users\user\Desktop\MARTHA CEMA TEST\source\cell_images\Uninfected"

def preprocess_image(img_path, label, image_size=(64, 64)):
    try:
        img = io.imread(img_path)
        
        img = transform.resize(img, image_size, mode='constant', anti_aliasing=True)

        img = img.astype("float32") / 255.0

        return img, label
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None, None
    
X = []
y = []

# Preprocess parasitized images
for filename in os.listdir(parasitized_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img_path = os.path.join(parasitized_dir, filename)
        img, label = preprocess_image(img_path, 1)  # Label 1 for parasitized
        if img is not None:
            X.append(img)
            y.append(label)

# Preprocess uninfected images
for filename in os.listdir(uninfected_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img_path = os.path.join(uninfected_dir, filename)
        img, label = preprocess_image(img_path, 0)  # Label 0 for uninfected
        if img is not None:
            X.append(img)
            y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Finished preprocessing data. X shape: {X.shape}, y shape: {y.shape}")

X_flat = X.reshape(X.shape[0], -1)

print(f"X_flat shape: {X_flat.shape}")

X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

# Initializing and train the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Model accuracy: {accuracy_score(y_test, y_pred):.2f}")

for i in range(len(y_pred)):
  if y_pred[i] == 1:
    print("Image", i, "predicted:", "Parasitized")
  else:
    print("Image", i, "predicted:", "Uninfected")