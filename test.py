import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Load and preprocess CIFAR-10 data
def load_and_preprocess_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return X_train, X_test, y_train, y_test

# Split data
X_train, X_test, y_train, y_test = load_and_preprocess_data()


# Define a simple CNN model
def create_classification_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Create the model
model = create_classification_model()

#number of epochs
epochs = 15

# Train the model
model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_split=0.2)


from sklearn.metrics import classification_report, confusion_matrix

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)


# Classification Report
class_report = classification_report(y_true_classes, y_pred_classes)
print("Classification Report:")
print(class_report)

# Identify difficult-to-classify images
confidence_scores = np.max(y_pred, axis=1)
difficult_indices = np.argsort(np.abs(confidence_scores - 0.5))[:5]

for lay in model.layers:
    print(lay.name)

def analyze_difficult_images(X_test, y_test, difficult_indices, model):
    for layer in model.layers:
        # Visualize for each convolutional layer
        last_conv_layer = model.get_layer(layer.name)

        for idx in difficult_indices:
            # Visualize the original image
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(X_test[idx])
            plt.title(f"True Class: {y_true_classes[idx]}, Predicted Class: {y_pred_classes[idx]}")

            # Visualize activation map using Grad-CAM
            plt.subplot(1, 2, 2)
            grad_cam_image = generate_grad_cam(X_test[idx], model, last_conv_layer)
            plt.imshow(grad_cam_image, cmap='jet')
            plt.title("Grad-CAM")

            # Overlay Grad-CAM on the original image
            plt.subplot(1, 3, 3)
            overlay_image = overlay_grad_cam(X_test[idx], grad_cam_image)
            plt.imshow(overlay_image)
            plt.title(f"Grad-CAM - Layer: {layer.name}")

            plt.show()

            # Display class probabilities
            class_probabilities = model.predict(np.expand_dims(X_test[idx], axis=0))
            top_classes = np.argsort(class_probabilities[0])[::-1][:3]

            print("Top Predicted Classes:")
            for i, class_idx in enumerate(top_classes):
                class_name = get_class_name(class_idx)
                print(f"{i + 1}. {class_name}: {class_probabilities[0][class_idx]:.4f}")

def overlay_grad_cam(img_array, cam):
    # Convert the image and Grad-CAM to RGB
    img_rgb = cv2.cvtColor(img_array.astype('float32'), cv2.COLOR_BGR2RGB)
    cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Ensure both images have the same data type
    img_rgb = img_rgb.astype(cam.dtype)

    # Blend the images
    alpha = 0.5
    overlay_image = cv2.addWeighted(cam, alpha, img_rgb, 1 - alpha, 0)

    return overlay_image


def get_class_name(class_idx):
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    return class_names[class_idx]


def generate_grad_cam(img_array, model, last_conv_layer):
    img_array = np.expand_dims(img_array, axis=0)

    # activate heat maps
    grad_model = Model(model.inputs, [last_conv_layer.output, model.output])

    # Compute the gradient
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    # Global average pooling for Dense layers
    if 'dense' in last_conv_layer.name:
        cam = tf.reduce_mean(grads, axis=(0, 1))
    # MaxPooling2D visualization
    elif 'max_pooling2d' in last_conv_layer.name:
        cam = output
    else:
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = np.dot(output, weights)

    # Resize the CAM to match the original image size
    cam = cv2.resize(np.array(cam), (32, 32), interpolation=cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)  # ReLU-like activation

    # Normalize
    cam = (cam - cam.min()) / (cam.max() - cam.min()) * 255

    # make cam 2d
    cam = np.mean(cam, axis=-1) if len(cam.shape) == 3 else cam

    return cam.astype(np.uint8)


# Analyze difficult images
analyze_difficult_images(X_test, y_test, difficult_indices, model)
print('done')
