import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import torch

vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
vit_processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

def load_model_and_layer(model_path):
    model = load_model(model_path, compile=False)
    effnet = model.get_layer("efficientnetb0")
    for layer in reversed(effnet.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            print(f"🧪 Found last Conv2D layer: {layer.name}")
            return model, layer.name
    print("⚠️ No Conv2D layer found in EfficientNetB0!")
    return model, None

def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = img_to_array(img) / 255.0
    return img_array

def extract_vit_feature(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray((img * 255).astype(np.uint8))
    inputs = vit_processor(images=img, return_tensors="pt")
    outputs = vit_model(**inputs)
    cls_token = outputs.last_hidden_state[:, 0, :]
    return cls_token.detach().numpy()[0]

def generate_gradcam(model, img_array, layer_name):
    print(f"Generating Grad-CAM with layer: {layer_name}")
    if layer_name is None:
        print("⚠️ No layer provided, returning original image.")
        return np.array(img_array * 255, dtype=np.uint8)

    effnet = model.get_layer("efficientnetb0")
    print(f"Accessing layer: {layer_name} in EfficientNetB0")
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[effnet.get_layer(layer_name).output, model.output]
    )

    img_input = tf.convert_to_tensor(np.expand_dims(img_array, axis=0), dtype=tf.float32)
    vit_input = tf.convert_to_tensor(np.expand_dims(extract_vit_feature(img_array), axis=0), dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_input)
        conv_outputs, predictions = grad_model([img_input, vit_input])
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_output = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed = heatmap * 0.4 + (img_array * 255)
    return superimposed.astype(np.uint8)