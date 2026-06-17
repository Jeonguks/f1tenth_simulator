import os
import yaml
import cv2
import numpy as np

YAML_PATH = "map/godmap1.yaml"
INPUT_CSV = "result_gui_graph/raceline_with_speed_px.csv"
OUTPUT_CSV = "result_gui_graph/waypoints.csv"

with open(YAML_PATH, "r") as f:
    map_info = yaml.safe_load(f)

resolution = float(map_info["resolution"])
origin_x, origin_y, _ = map_info["origin"]

image_path = os.path.join(os.path.dirname(YAML_PATH), map_info["image"])

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(image_path)

height, width = img.shape

data = np.loadtxt(INPUT_CSV, delimiter=",", skiprows=1)
if data.ndim == 1:
    data = data.reshape(1, -1)

px = data[:, 0]
py = data[:, 1]
v = data[:, 2]

x_world = origin_x + (px + 0.5) * resolution
y_world = origin_y + (height - py - 0.5) * resolution

world_data = np.column_stack([x_world, y_world, v])

np.savetxt(
    OUTPUT_CSV,
    world_data,
    delimiter=",",
    fmt="%.6f"
)

print("저장 완료:", OUTPUT_CSV)
print("image size:", width, height)
print("resolution:", resolution)
print("origin:", origin_x, origin_y)