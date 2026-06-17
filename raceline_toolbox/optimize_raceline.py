import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import sys
sys.path.append("./build")  # raceline_core.so 위치

import cv2
import numpy as np
from scipy.interpolate import splprep, splev

import raceline_core


MAP_PATH = "map/godmap1.png"
CENTER_CSV = "result/selected_loop.csv"
OUT_DIR = "result_gui_graph/"
os.makedirs(OUT_DIR, exist_ok=True)

BINARY_THRESHOLD = 250
MAX_RAY_DIST = 120

DEFAULT_SPEED = 1.0


def load_centerline(path):
    pts = np.loadtxt(path, delimiter=",")
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    return pts[:, :2].astype(np.float64)


def reduce_centerline_points(points, min_dist):
    if len(points) == 0:
        return points

    out = [points[0]]
    last = points[0]

    for p in points[1:]:
        if np.linalg.norm(p - last) >= min_dist:
            out.append(p)
            last = p

    return np.array(out, dtype=np.float64)


def close_loop(points):
    if len(points) == 0:
        return points

    if np.linalg.norm(points[0] - points[-1]) > 1e-6:
        return np.vstack([points, points[0]])

    return points


def resample_by_count(points, n):
    points = close_loop(points)

    seg = np.diff(points, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)

    valid = seg_len > 1e-6
    p0 = points[:-1][valid]
    p1 = points[1:][valid]

    if len(p0) < 2:
        return points[:-1]

    seg = p1 - p0
    seg_len = np.linalg.norm(seg, axis=1)

    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]

    ds = np.linspace(0, total, n, endpoint=False)

    out = []
    for d in ds:
        i = np.searchsorted(cum, d, side="right") - 1
        i = max(0, min(i, len(seg_len) - 1))

        t = (d - cum[i]) / max(seg_len[i], 1e-6)
        out.append(p0[i] + t * seg[i])

    return np.array(out, dtype=np.float64)


def make_track():
    img = cv2.imread(MAP_PATH, 0)
    if img is None:
        raise FileNotFoundError(MAP_PATH)

    _, track = cv2.threshold(img, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    return img, track


def inside(track, p):
    x = int(round(p[0]))
    y = int(round(p[1]))

    if x < 0 or y < 0 or y >= track.shape[0] or x >= track.shape[1]:
        return False

    return track[y, x] > 0


def ray(track, p, direction):
    last = 0.0

    for i in range(MAX_RAY_DIST + 1):
        q = p + direction * i

        if inside(track, q):
            last = float(i)
        else:
            break

    return last


def tangent_normal(points):
    prev = np.roll(points, 1, axis=0)
    nxt = np.roll(points, -1, axis=0)

    t = nxt - prev
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-6)

    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    return n


def build_candidates(track, center, normals, margin, maxcand):
    cand = []
    offsets = []
    valid = []

    for p, n in zip(center, normals):
        left = ray(track, p, n)
        right = ray(track, p, -n)

        mn = -right + margin
        mx = left - margin

        if mn >= mx or maxcand <= 1:
            offs = np.array([0.0], dtype=np.float64)
        else:
            offs = np.linspace(mn, mx, maxcand, dtype=np.float64)

            if not np.any(np.isclose(offs, 0.0, atol=1e-6)):
                offs = np.append(offs, 0.0)
                offs.sort()

        pts = []
        v = []

        for o in offs:
            q = p + n * o
            pts.append(q)
            v.append(inside(track, q))

        cand.append(np.array(pts, dtype=np.float64))
        offsets.append(np.array(offs, dtype=np.float64))
        valid.append(np.array(v, dtype=bool))

    return cand, offsets, valid


def pad_candidates(cand, valid, offsets):
    n = len(cand)
    m = max(len(c) for c in cand)

    cand_np = np.zeros((n, m, 2), dtype=np.float64)
    valid_np = np.zeros((n, m), dtype=bool)
    offsets_np = np.zeros((n, m), dtype=np.float64)

    for i in range(n):
        size = len(cand[i])
        cand_np[i, :size, :] = cand[i]
        valid_np[i, :size] = valid[i]
        offsets_np[i, :size] = offsets[i]

    return cand_np, valid_np, offsets_np


def build_loop_cpp(cand, valid, offsets, smooth_w, offset_w):
    cand_np, valid_np, offsets_np = pad_candidates(cand, valid, offsets)

    selected_indices = raceline_core.shortest_loop_dp(
        cand_np,
        valid_np,
        offsets_np,
        float(smooth_w),
        float(offset_w)
    )

    selected_indices = np.asarray(selected_indices, dtype=np.int32)

    path = np.array([
        cand[i][selected_indices[i]]
        for i in range(len(cand))
    ], dtype=np.float64)

    return path, selected_indices


def spline_closed(points, smooth, out_count=None):
    if len(points) < 4:
        return points

    if out_count is None:
        out_count = len(points)

    pts = close_loop(points)

    try:
        tck, _ = splprep(
            [pts[:, 0], pts[:, 1]],
            s=float(smooth),
            per=True
        )

        u = np.linspace(0, 1, out_count, endpoint=False)
        x, y = splev(u, tck)

        return np.column_stack([x, y]).astype(np.float64)

    except Exception as e:
        print("[WARN] spline 실패:", e)
        return points


def estimate_curvature(p0, p1, p2):
    a = np.linalg.norm(p1 - p0)
    b = np.linalg.norm(p2 - p1)
    c = np.linalg.norm(p2 - p0)

    area = abs(np.cross(p1 - p0, p2 - p0)) / 2.0

    if area < 1e-6 or a * b * c < 1e-6:
        return 0.0

    return 4.0 * area / (a * b * c)


def segment_lengths(path):
    nxt = np.roll(path, -1, axis=0)
    return np.linalg.norm(nxt - path, axis=1)


def calc_velocity_profile(
    path,
    v_min=1.0,
    v_max=6.0,
    lat_acc_max=4.0,
    accel_max=1.5,
    decel_max=2.5,
):
    n = len(path)

    curvature = np.zeros(n, dtype=np.float64)

    for i in range(n):
        p0 = path[i - 1]
        p1 = path[i]
        p2 = path[(i + 1) % n]
        curvature[i] = abs(estimate_curvature(p0, p1, p2))

    for _ in range(3):
        curvature = (
            0.25 * np.roll(curvature, 1)
            + 0.50 * curvature
            + 0.25 * np.roll(curvature, -1)
        )

    velocity = np.sqrt(lat_acc_max / (curvature + 1e-6))
    velocity = np.clip(velocity, v_min, v_max)

    ds = segment_lengths(path)

    for _ in range(5):
        for i in range(n):
            j = (i + 1) % n
            v_allowed = np.sqrt(velocity[i] ** 2 + 2.0 * accel_max * ds[i])
            if velocity[j] > v_allowed:
                velocity[j] = v_allowed

    for _ in range(5):
        for i in range(n - 1, -1, -1):
            j = (i - 1) % n
            v_allowed = np.sqrt(velocity[i] ** 2 + 2.0 * decel_max * ds[j])
            if velocity[j] > v_allowed:
                velocity[j] = v_allowed

    for _ in range(2):
        velocity = (
            0.20 * np.roll(velocity, 1)
            + 0.60 * velocity
            + 0.20 * np.roll(velocity, -1)
        )

    velocity = np.clip(velocity, v_min, v_max)

    return velocity, curvature


def draw_points(img, pts, color, radius=2):
    for p in pts:
        x = int(round(p[0]))
        y = int(round(p[1]))

        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x, y), radius, color, -1)


def draw_loop(img, path, color=(255, 0, 0), thickness=2):
    if len(path) < 2:
        return

    for i in range(len(path)):
        p1 = path[i]
        p2 = path[(i + 1) % len(path)]

        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        x2, y2 = int(round(p2[0])), int(round(p2[1]))

        cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_velocity_points(img, path, velocity):
    vmin = np.min(velocity)
    vmax = np.max(velocity)
    denom = max(vmax - vmin, 1e-6)

    for p, v in zip(path, velocity):
        ratio = (v - vmin) / denom

        r = int(255 * (1.0 - ratio))
        g = int(255 * ratio)
        b = 0

        x = int(round(p[0]))
        y = int(round(p[1]))

        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x, y), 3, (b, g, r), -1)


img, track = make_track()
center_raw = load_centerline(CENTER_CSV)

cv2.namedWindow("control", cv2.WINDOW_NORMAL)
cv2.namedWindow("view", cv2.WINDOW_NORMAL)

cv2.createTrackbar("points", "control", 220, 800, lambda x: None)
cv2.createTrackbar("reduce", "control", 10, 40, lambda x: None)
cv2.createTrackbar("margin", "control", 7, 30, lambda x: None)
cv2.createTrackbar("cand", "control", 25, 100, lambda x: None)
cv2.createTrackbar("smooth x10", "control", 10, 200, lambda x: None)
cv2.createTrackbar("offset x100", "control", 5, 100, lambda x: None)
cv2.createTrackbar("spline", "control", 5, 100, lambda x: None)

cv2.createTrackbar("vmin x10", "control", 10, 100, lambda x: None)
cv2.createTrackbar("vmax x10", "control", 60, 150, lambda x: None)
cv2.createTrackbar("latacc x10", "control", 40, 150, lambda x: None)
cv2.createTrackbar("accel x10", "control", 15, 100, lambda x: None)
cv2.createTrackbar("decel x10", "control", 25, 150, lambda x: None)

last_params = None
cached = None

print("q 또는 ESC: 종료")
print("s: 저장")
print("색상: 빨강=center, 노랑=candidate, 파랑=raceline, 빨강~초록=velocity")

while True:
    n_points = max(cv2.getTrackbarPos("points", "control"), 3)
    reduce_v = max(cv2.getTrackbarPos("reduce", "control"), 1)
    margin_v = cv2.getTrackbarPos("margin", "control")
    maxcand_v = max(cv2.getTrackbarPos("cand", "control"), 1)

    smooth_w = cv2.getTrackbarPos("smooth x10", "control") / 10.0
    offset_w = cv2.getTrackbarPos("offset x100", "control") / 100.0
    spline_s = cv2.getTrackbarPos("spline", "control")

    v_min = max(cv2.getTrackbarPos("vmin x10", "control") / 10.0, 0.1)
    v_max = max(cv2.getTrackbarPos("vmax x10", "control") / 10.0, v_min)
    latacc = max(cv2.getTrackbarPos("latacc x10", "control") / 10.0, 0.1)
    accel = max(cv2.getTrackbarPos("accel x10", "control") / 10.0, 0.1)
    decel = max(cv2.getTrackbarPos("decel x10", "control") / 10.0, 0.1)

    params = (
        n_points,
        reduce_v,
        margin_v,
        maxcand_v,
        smooth_w,
        offset_w,
        spline_s,
        v_min,
        v_max,
        latacc,
        accel,
        decel,
    )

    if params != last_params:
        reduced = reduce_centerline_points(center_raw, reduce_v)
        center = resample_by_count(reduced, n_points)

        normals = tangent_normal(center)

        cand, offsets, valid = build_candidates(
            track,
            center,
            normals,
            margin_v,
            maxcand_v
        )

        loop_path, selected_indices = build_loop_cpp(
            cand,
            valid,
            offsets,
            smooth_w,
            offset_w
        )

        spline_path = spline_closed(
            loop_path,
            smooth=spline_s,
            out_count=len(loop_path)
        )

        velocity, curvature = calc_velocity_profile(
            spline_path,
            v_min=v_min,
            v_max=v_max,
            lat_acc_max=latacc,
            accel_max=accel,
            decel_max=decel,
        )

        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for c, v in zip(cand, valid):
            for p, ok in zip(c, v):
                if ok:
                    draw_points(vis, [p], (0, 255, 255), radius=1)

        draw_points(vis, center, (0, 0, 255), radius=2)

        draw_loop(vis, spline_path, color=(255, 0, 0), thickness=2)
        draw_velocity_points(vis, spline_path, velocity)

        text = (
            f"N={len(center)}, cand={maxcand_v}, margin={margin_v}, "
            f"smooth={smooth_w:.1f}, offset={offset_w:.2f}, spline={spline_s}, "
            f"v={np.min(velocity):.2f}~{np.max(velocity):.2f}"
        )

        cv2.putText(
            vis,
            text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1
        )

        cached = {
            "center": center,
            "cand": cand,
            "offsets": offsets,
            "valid": valid,
            "loop_path": loop_path,
            "spline_path": spline_path,
            "selected_indices": selected_indices,
            "velocity": velocity,
            "curvature": curvature,
            "vis": vis,
        }

        last_params = params

    cv2.imshow("view", cached["vis"])

    key = cv2.waitKey(20) & 0xFF

    if key == ord("s"):
        spline_path = cached["spline_path"]
        velocity = cached["velocity"]
        curvature = cached["curvature"]

        out = np.column_stack([
            spline_path[:, 0],
            spline_path[:, 1],
            velocity,
            curvature
        ])

        np.savetxt(
            OUT_DIR + "raceline_with_speed_px.csv",
            out,
            delimiter=",",
            fmt="%.6f",
            header="x,y,velocity,curvature",
            comments=""
        )

        cv2.imwrite(
            OUT_DIR + "raceline_with_speed_preview.png",
            cached["vis"]
        )

        print("저장 완료:")
        print(f"- {OUT_DIR}raceline_with_speed_px.csv")
        print(f"- {OUT_DIR}raceline_with_speed_preview.png")

    elif key == ord("q") or key == 27:
        break

cv2.destroyAllWindows()