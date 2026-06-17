import os
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt

YAML_PATH = "map/godmap1.yaml"
CSV_PATH = "result_gui_graph/waypoints.csv"
OUT_IMAGE = "result_gui_graph/raceline_2d_overlay.png"

os.makedirs(os.path.dirname(OUT_IMAGE), exist_ok=True)

# =========================
# YAML 로드
# =========================
with open(YAML_PATH, "r") as f:
    map_info = yaml.safe_load(f)

resolution = float(map_info["resolution"])
origin_x, origin_y, _ = map_info["origin"]

image_path = os.path.join(os.path.dirname(YAML_PATH), map_info["image"])

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(image_path)

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(CSV_PATH)

h, w = img.shape

# =========================
# raceline
# =========================
data = np.loadtxt(CSV_PATH, delimiter=",")
if data.ndim == 1:
    data = data.reshape(1, -1)

x = data[:, 0]
y = data[:, 1]
v = data[:, 2]

# =========================
# plot
# =========================
plt.figure(figsize=(10, 8))

# 맵을 world 좌표 기준으로 표시
plt.imshow(
    img,
    cmap="gray",
    extent=[
        origin_x,
        origin_x + w * resolution,
        origin_y,
        origin_y + h * resolution
    ],
    origin="upper"
)

# raceline
plt.plot(x, y, color="red", linewidth=2)
sc = plt.scatter(x, y, c=v, cmap="jet", s=10)

plt.colorbar(sc, label="velocity")

plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Raceline Overlay on Map")

plt.axis("equal")
plt.grid(False)

plt.savefig(OUT_IMAGE, dpi=200)
plt.show()

print("저장 완료:", OUT_IMAGE)