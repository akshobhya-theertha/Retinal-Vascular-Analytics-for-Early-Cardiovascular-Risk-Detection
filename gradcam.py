"""
Retinexia - Grad-CAM (Gradient-weighted Class Activation Mapping)
================================================================
Explainable AI: highlights regions of the retinal image that most
influenced the model's prediction (e.g. abnormal vessels, microaneurysms).

How Grad-CAM works:
1. Take the last convolutional layer output (feature maps).
2. Compute gradients of the predicted class score w.r.t. each feature map.
3. Weight each feature map by its mean gradient (importance).
4. Sum weighted maps, ReLU, and resize to input size → heatmap.
5. Overlay heatmap on the original image for visualization.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras


def find_target_layer(model, layer_name=None):
    """
    Find the last convolutional layer in the model for Grad-CAM.
    MobileNetV2's last conv block is typically 'out_relu' or the last Conv2D.
    """
    if layer_name is not None:
        for layer in model.layers:
            if layer.name == layer_name:
                return layer
    # Default: last Conv2D in the base (before global pooling)
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    return None


def make_gradcam_model(model, layer_name=None):
    """
    Build a submodel that outputs (model_outputs, target_layer_output).
    model_outputs is [dr_out, cvd_out]; we need the conv layer for gradients.
    """
    layer = find_target_layer(model, layer_name)
    if layer is None:
        raise ValueError("No Conv2D layer found for Grad-CAM")
    return keras.Model(
        inputs=model.input,
        outputs=[model.output, layer.output],
    ), layer.name


def compute_heatmap(gradcam_model, img_array, output_index=0, pred_index=None):
    """
    Compute Grad-CAM heatmap for one image.
    - output_index: 0 = DR head, 1 = CVD head.
    - pred_index: which class to explain (default: predicted class).
    Returns heatmap (H, W) normalized to [0, 1].
    """
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        # gradcam_model returns (model_outputs, layer_output); model_outputs = [dr_out, cvd_out]
        model_outs, layer_out = gradcam_model(img_tensor)
        logits = model_outs[output_index]
        if pred_index is None:
            pred_index = int(tf.argmax(logits[0]).numpy())
        class_channel = logits[0, pred_index]
    grads = tape.gradient(class_channel, layer_out)
    if grads is None:
        return np.zeros(img_array.shape[1], img_array.shape[2])
    # Global average of gradients (weights)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    # Weighted combination of feature maps
    cam = tf.reduce_sum(weights * layer_out, axis=-1)
    cam = tf.nn.relu(cam)[0].numpy()
    # Resize to input size
    h, w = img_array.shape[1], img_array.shape[2]
    cam = tf.image.resize(
        tf.expand_dims(tf.expand_dims(cam, -1), 0),
        (h, w),
        method="bilinear",
    ).numpy()[0, :, :, 0]
    # Normalize to [0, 1]
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam.astype(np.float32)


def overlay_heatmap_on_image(img_array, heatmap, alpha=0.5, colormap="jet"):
    """
    Overlay heatmap on the original image for visualization.
    img_array: (1, H, W, 3) or (H, W, 3), values 0-1.
    heatmap: (H, W), values 0-1.
    Returns RGB image (H, W, 3) for saving/display.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if img_array.ndim == 4:
        img = img_array[0]
    else:
        img = img_array
    img = np.squeeze(img)
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    h, w = heatmap.shape
    heatmap_resized = np.uint8(255 * heatmap)
    if colormap == "jet":
        try:
            jet = plt.get_cmap("jet")
        except AttributeError:
            jet = cm.get_cmap("jet")
        heatmap_rgb = jet(heatmap_resized)[:, :, :3]
    else:
        heatmap_rgb = np.stack([heatmap_resized] * 3, axis=-1) / 255.0
    if img.shape[:2] != (h, w):
        heatmap_rgb = tf.image.resize(
            np.expand_dims(heatmap_rgb, 0), (img.shape[0], img.shape[1])
        ).numpy()[0]
    else:
        heatmap_rgb = heatmap_rgb
    overlay = (alpha * heatmap_rgb + (1 - alpha) * (img / 255.0))
    overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
    return overlay


def get_gradcam_overlay(model, img_array, output_index=0, alpha=0.5):
    """
    One-shot: compute Grad-CAM and overlay for the DR head (output_index=0).
    img_array: (1, H, W, 3) normalized. Returns overlay image (H, W, 3).
    """
    gradcam_model, _ = make_gradcam_model(model)
    heatmap = compute_heatmap(gradcam_model, img_array, output_index=output_index)
    return overlay_heatmap_on_image(img_array, heatmap, alpha=alpha)
