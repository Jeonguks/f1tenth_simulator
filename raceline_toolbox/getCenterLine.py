import os
import cv2
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize

# =========================
# 설정
# =========================
MAP_PATH = "map/godmap1.png"
OUT_DIR = "result/"
BINARY_THRESHOLD = 250
MIN_BRANCH_LENGTH = 40

OUT_CSV = OUT_DIR + "selected_loop.csv"
OUT_PREVIEW = OUT_DIR + "selected_loop_preview.png"

os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# graph 함수
# =========================
def build_graph(skel_img):
    binary = skel_img > 0
    h, w = binary.shape
    G = nx.Graph()

    ys, xs = np.where(binary)

    for x, y in zip(xs, ys):
        G.add_node((x, y))

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                nx_, ny_ = x + dx, y + dy

                if 0 <= nx_ < w and 0 <= ny_ < h and binary[ny_, nx_]:
                    weight = float(np.hypot(dx, dy))
                    G.add_edge((x, y), (nx_, ny_), weight=weight)

    return G


def prune_branches(G, min_length=20):
    G = G.copy()

    changed = True
    while changed:
        changed = False

        endpoints = [n for n in G.nodes if G.degree[n] == 1]

        for ep in endpoints:
            if ep not in G or G.degree[ep] != 1:
                continue

            path = [ep]
            prev = None
            cur = ep

            while True:
                neighbors = [n for n in G.neighbors(cur) if n != prev]

                if len(neighbors) == 0:
                    break

                nxt = neighbors[0]
                path.append(nxt)

                prev, cur = cur, nxt

                if G.degree[cur] != 2:
                    break

            if len(path) < min_length:
                remove_nodes = path[:-1]
                G.remove_nodes_from(remove_nodes)
                changed = True
                break

    return G


def graph_to_image(G, shape):
    out = np.zeros(shape, dtype=np.uint8)
    for x, y in G.nodes:
        out[y, x] = 255
    return out


# =========================
# 맵 전처리
# =========================
map_src = cv2.imread(MAP_PATH, cv2.IMREAD_GRAYSCALE)
if map_src is None:
    raise FileNotFoundError(MAP_PATH)

_, binary = cv2.threshold(
    map_src,
    BINARY_THRESHOLD,
    255,
    cv2.THRESH_BINARY
)

K = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

bin_track = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, K)
bin_track = cv2.morphologyEx(bin_track, cv2.MORPH_OPEN, K)

# =========================
# skeletonize
# =========================
track_bool = bin_track > 0
skeleton = skeletonize(track_bool)
skeleton_img = (skeleton.astype(np.uint8)) * 255

# =========================
# 가지치기
# =========================
G = build_graph(skeleton_img)
G_pruned = prune_branches(G, min_length=MIN_BRANCH_LENGTH)
skeleton_pruned = graph_to_image(G_pruned, skeleton_img.shape)

# =========================
# 중간 결과 저장
# =========================
cv2.imwrite(OUT_DIR + "binary.png", binary)
cv2.imwrite(OUT_DIR + "bin_track.png", bin_track)
cv2.imwrite(OUT_DIR + "skeleton.png", skeleton_img)
cv2.imwrite(OUT_DIR + "skeleton_pruned.png", skeleton_pruned)


# =========================
# 경유점 선택해서 루프 생성
# =========================
G_loop = build_graph(skeleton_pruned)

binary_pruned = skeleton_pruned > 0
skeleton_points = np.column_stack(np.where(binary_pruned))  # y, x

if len(skeleton_points) == 0:
    raise RuntimeError("가지치기 후 skeleton 점이 없음")

clicked_points = []
snapped_points = []


def snap_to_skeleton(x, y):
    d = (skeleton_points[:, 1] - x) ** 2 + (skeleton_points[:, 0] - y) ** 2
    idx = np.argmin(d)
    sy, sx = skeleton_points[idx]
    return int(sx), int(sy)


def make_vis():
    vis_img = cv2.cvtColor(map_src, cv2.COLOR_GRAY2BGR)

    # 주행 가능 영역: 어두운 회색
    vis_img[bin_track > 0] = (70, 70, 70)

    # 가지치기된 skeleton: 빨간색
    vis_img[binary_pruned] = (0, 0, 255)

    return vis_img


vis = make_vis()


def mouse_callback(event, x, y, flags, param):
    global vis

    if event == cv2.EVENT_LBUTTONDOWN:
        snapped = snap_to_skeleton(x, y)

        clicked_points.append((x, y))
        snapped_points.append(snapped)

        cv2.circle(vis, (x, y), 4, (255, 0, 0), -1)
        cv2.circle(vis, snapped, 5, (0, 255, 0), -1)

        cv2.putText(
            vis,
            str(len(snapped_points)),
            (snapped[0] + 5, snapped[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )


cv2.namedWindow("select waypoints", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("select waypoints", mouse_callback)

print("왼쪽 클릭: 경유점 선택")
print("s: 선택한 경유점들을 연결해서 폐곡선 저장")
print("r: 선택 초기화")
print("q 또는 ESC: 종료")
print("중간 결과 저장 완료:")
print(f"- {OUT_DIR}binary.png")
print(f"- {OUT_DIR}bin_track.png")
print(f"- {OUT_DIR}skeleton.png")
print(f"- {OUT_DIR}skeleton_pruned.png")

while True:
    cv2.imshow("select waypoints", vis)
    key = cv2.waitKey(20) & 0xFF

    if key == ord("r"):
        clicked_points.clear()
        snapped_points.clear()
        vis = make_vis()
        print("초기화 완료")

    elif key == ord("s"):
        if len(snapped_points) < 2:
            print("경유점이 최소 2개 필요함")
            continue

        route_points = snapped_points + [snapped_points[0]]
        full_path = []
        valid = True

        for i in range(len(route_points) - 1):
            start = route_points[i]
            goal = route_points[i + 1]

            try:
                segment = nx.shortest_path(
                    G_loop,
                    source=start,
                    target=goal,
                    weight="weight"
                )
            except nx.NetworkXNoPath:
                print(f"경로 없음: {start} -> {goal}")
                valid = False
                break
            except nx.NodeNotFound:
                print(f"노드 없음: {start} 또는 {goal}")
                valid = False
                break

            if i > 0:
                segment = segment[1:]

            full_path.extend(segment)

        if not valid or len(full_path) == 0:
            print("루프 생성 실패")
            continue

        path_arr = np.array(full_path, dtype=np.int32)
        np.savetxt(OUT_CSV, path_arr, delimiter=",", fmt="%d")

        result = cv2.cvtColor(map_src, cv2.COLOR_GRAY2BGR)
        result[bin_track > 0] = (50, 50, 50)
        result[binary_pruned] = (80, 80, 80)

        for x, y in full_path:
            result[y, x] = (255, 255, 255)

        for p in snapped_points:
            cv2.circle(result, p, 5, (0, 255, 0), -1)

        cv2.imshow("selected loop", result)
        cv2.imwrite(OUT_PREVIEW, result)

        print(f"저장 완료: {OUT_CSV}")
        print(f"미리보기 저장: {OUT_PREVIEW}")
        print(f"경유점 수: {len(snapped_points)}")
        print(f"루프 픽셀 수: {len(full_path)}")

    elif key == ord("q") or key == 27:
        break

cv2.destroyAllWindows()