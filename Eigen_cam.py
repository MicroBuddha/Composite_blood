import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# === Helper functions ===
def add_gaussian_noise(image, mean=0, std=25):
    gaussian = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + gaussian
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_poisson_noise(image):
    image_float = image.astype(np.float32) / 255.0
    noisy = np.random.poisson(image_float * 255) / 255.0
    noisy = np.clip(noisy * 255, 0, 255).astype(np.uint8)
    return noisy

# === Load YOLOv11 model ===
model_path = "/home/buddhadev/Desktop/afnan_blood/combined_dataset_v2/train_yolov11m/weights/best.pt"
model = YOLO(model_path)

# === Load and prepare original image ===
img_path = "/home/buddhadev/Desktop/afnan_blood/combined_dataset_v2/test/transformed_images_by_transform/original/gen_test_0019.jpg"
image_bgr = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_rgb_resized = cv2.resize(image_rgb, (640, 640))

# === Create Grayscale version (3-channel)
gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
gray_resized = cv2.resize(gray_3ch, (640, 640))

# === Create Colored Noise image (Gaussian + Poisson)
color_gaussian = add_gaussian_noise(image_rgb)
color_gaussian_poisson = add_poisson_noise(color_gaussian)
color_gaussian_poisson_resized = cv2.resize(color_gaussian_poisson, (640, 640))

# === Configure CAM ===
target_layers = [model.model.model[-2]]  # Last conv layer before head
cam = EigenCAM(model, target_layers, task='od')

# === Function to generate CAM image ===
def generate_cam(image_rgb):
    rgb_input = image_rgb.copy()
    norm_img = np.float32(image_rgb) / 255.0
    grayscale_cam = cam(rgb_input)[0, :, :]
    cam_image = show_cam_on_image(norm_img, grayscale_cam, use_rgb=True)
    return cam_image

# === Generate CAMs for each version ===
cam_original = generate_cam(image_rgb_resized)
cam_gray = generate_cam(gray_resized)
cam_noise = generate_cam(color_gaussian_poisson_resized)

# === Prepare list of images and titles ===
images = [image_rgb_resized, cam_original, cam_gray, cam_noise]
titles = ["Colour model -Original (No CAM)", "Original + Eigen-CAM", "Grayscale + Eigen-CAM", "Colored Noise + Eigen-CAM"]

# === Create 2x2 Grid Plot ===
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i])
    ax.set_title(titles[i], fontsize=18)
    ax.axis("off")
plt.tight_layout()

# === Save the figure ===
fig.savefig("eigen_cam_2x2_grid_color.png", dpi=300)
plt.show()
