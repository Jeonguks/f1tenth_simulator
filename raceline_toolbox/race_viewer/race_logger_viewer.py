import yaml
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button


# config ################
csv_path = "race_log.csv"
map_src = "../map/godmap1.png"
map_config = "../map/godmap1.yaml"
waypoint = "../result_gui_graph/waypoints.csv"
#########################


class RaceVisualizer:
    def __init__(self):
        self.df = pd.read_csv(csv_path)
        self.wp = pd.read_csv(waypoint, header=None)

        self.wp_x = self.wp.iloc[:, 0].to_numpy()
        self.wp_y = self.wp.iloc[:, 1].to_numpy()

        self.img, self.extent = self.load_ros_map(map_src, map_config)

        self.lap_ids = sorted(self.df["lap_id"].unique())
        self.current_lap_index = 0

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.subplots_adjust(bottom=0.22)

        self.draw_all()

        self.create_buttons()

    def load_ros_map(self, map_image_path, map_yaml_path):
        with open(map_yaml_path, "r") as f:
            map_yaml = yaml.safe_load(f)

        resolution = float(map_yaml["resolution"])
        origin_x = float(map_yaml["origin"][0])
        origin_y = float(map_yaml["origin"][1])

        img = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Map image not found: {map_image_path}")

        h, w = img.shape
        img = np.flipud(img)

        extent = [
            origin_x,
            origin_x + w * resolution,
            origin_y,
            origin_y + h * resolution,
        ]

        return img, extent

    def normalize_speed_to_1_10(self, speed_series):
        speed = speed_series.astype(float).to_numpy()

        min_v = np.nanmin(speed)
        max_v = np.nanmax(speed)

        if max_v - min_v < 1e-9:
            return np.ones_like(speed)

        return 1.0 + 9.0 * (speed - min_v) / (max_v - min_v)

    def clear_and_draw_base_map(self):
        self.ax.clear()

        self.ax.imshow(
            self.img,
            cmap="gray",
            extent=self.extent,
            origin="lower",
            zorder=0
        )

        self.ax.axis("equal")
        self.ax.grid(True, zorder=1)
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")

    def plot_speed_colored_trajectory(self, df):
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()

        if len(x) < 2:
            return None

        speed_norm = self.normalize_speed_to_1_10(df["speed"])

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(
            segments,
            cmap="viridis",
            linewidth=2.5,
            zorder=2
        )

        lc.set_array(speed_norm[:-1])
        lc.set_clim(1, 10)

        self.ax.add_collection(lc)
        return lc

    def plot_waypoints(self):
        self.ax.plot(
            self.wp_x,
            self.wp_y,
            linestyle="--",
            linewidth=2,
            color="cyan",
            label="waypoint",
            zorder=3
        )

    def plot_collisions(self, df):
        collision_df = df[df["collision"] == 1]

        if len(collision_df) > 0:
            self.ax.scatter(
                collision_df["x"],
                collision_df["y"],
                marker="x",
                s=120,
                linewidths=3,
                color="red",
                label="collision",
                zorder=10
            )

    def draw_all(self, event=None):
        self.clear_and_draw_base_map()

        self.plot_speed_colored_trajectory(self.df)
        self.plot_waypoints()
        self.plot_collisions(self.df)

        self.ax.set_title("All Trajectory + Waypoints + Collisions")
        self.ax.legend()
        self.fig.canvas.draw_idle()

    def draw_waypoints_only(self, event=None):
        self.clear_and_draw_base_map()

        self.plot_waypoints()

        self.ax.set_title("Waypoints Only")
        self.ax.legend()
        self.fig.canvas.draw_idle()

    def draw_current_lap(self, event=None):
        self.clear_and_draw_base_map()

        if len(self.lap_ids) == 0:
            self.ax.set_title("No lap data")
            self.fig.canvas.draw_idle()
            return

        lap_id = self.lap_ids[self.current_lap_index]
        lap_df = self.df[self.df["lap_id"] == lap_id]

        self.plot_speed_colored_trajectory(lap_df)
        self.plot_waypoints()
        self.plot_collisions(lap_df)

        self.ax.set_title(f"Lap View - lap_id = {lap_id}")
        self.ax.legend()
        self.fig.canvas.draw_idle()

    def next_lap(self, event=None):
        if len(self.lap_ids) == 0:
            return

        self.current_lap_index += 1

        if self.current_lap_index >= len(self.lap_ids):
            self.current_lap_index = 0

        self.draw_current_lap()

    def create_buttons(self):
        ax_all = plt.axes([0.08, 0.05, 0.18, 0.06])
        ax_wp = plt.axes([0.30, 0.05, 0.18, 0.06])
        ax_lap = plt.axes([0.52, 0.05, 0.18, 0.06])
        ax_next = plt.axes([0.74, 0.05, 0.18, 0.06])

        self.btn_all = Button(ax_all, "All")
        self.btn_wp = Button(ax_wp, "Waypoint")
        self.btn_lap = Button(ax_lap, "Lap")
        self.btn_next = Button(ax_next, "Next Lap")

        self.btn_all.on_clicked(self.draw_all)
        self.btn_wp.on_clicked(self.draw_waypoints_only)
        self.btn_lap.on_clicked(self.draw_current_lap)
        self.btn_next.on_clicked(self.next_lap)

    def show(self):
        plt.show()


if __name__ == "__main__":
    visualizer = RaceVisualizer()
    visualizer.show()