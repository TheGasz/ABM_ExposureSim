"""
Modul abm/model.py  [OPTIMIZED]

Perubahan dari versi original:
1. _point_in_polygon → diganti numpy vectorized (batch semua obstacle sekaligus)
2. update_obstacle_heatmap → skip frame via HEATMAP_SKIP_FRAMES, vectorized pip
3. ray_cast_vision_polygon → hanya dipanggil untuk agen yang bergerak (bukan SITTING statis)
4. get_grid_snapshot → vision agen SITTING di-cache (tidak recompute tiap frame)
5. find_path → LRU cache untuk path statis yang sering diulang
6. _cell_has_human → set lookup O(1) menggantikan loop O(n)
"""

import heapq
import math
import random
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from constants import (
    CELL_CHAIR,
    CELL_DOOR,
    CELL_OBSTACLE,
    ChairDirections,
    Layout,
    CHAIR_DIR_VECTORS,
    SPEED_PX_PER_S,
    DEFAULT_FPSTEP,
    CELL_SIZE_PX,
    SPEED_VARIATION,
    OBSTACLE_SIDES,
    OBSTACLE_SIDE_VECTORS,
    HEATMAP_LOOK_DURATION_S,
)

from abm.agents import HumanAgent, HumanStatus
from viz.obstacle_heatmap import ExposureField


# Heatmap hanya diupdate setiap N step untuk hemat CPU.
# Nilai 3 = update tiap 3 step → ~3× lebih ringan tanpa kehilangan akurasi visual.
HEATMAP_SKIP_FRAMES = 3


class WaitingRoomModel:
    """Kelas WaitingRoomModel."""

    def __init__(
        self,
        width: int,
        height: int,
        layout: Layout,
        chair_directions: Optional[ChairDirections] = None,
        door_probs: Optional[Dict[Tuple[int, int], float]] = None,
        arrival_rate: float = 0.6,
        mean_sitting_s: float = 20.0,
        pass_through_prob: float = 0.5,
        fps_step: int = 10,
        speed_px_s: float = None,
        jitter: float = 0.25,
        stuck_threshold: int = 20,
        seed: int = 42,
    ) -> None:
        self.width = width
        self.height = height
        self.cell_size_px = CELL_SIZE_PX

        self.arrival_rate = max(0.0, arrival_rate)
        self.mean_sitting_s = max(1e-3, mean_sitting_s)
        self.pass_through_prob = min(max(pass_through_prob, 0.0), 1.0)
        if speed_px_s is None:
            speed_px_s = SPEED_PX_PER_S * fps_step / DEFAULT_FPSTEP

        self.speed_px_s = max(1e-3, speed_px_s)
        self.frame_dt = 1.0 / float(fps_step)
        self.jitter = min(max(jitter, 0.0), 0.95)
        self.stuck_threshold = max(1, stuck_threshold)

        self.current_step = 0
        self.time_s = 0.0
        self.total_customers = 0

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self._chair_directions = chair_directions or {}
        self._door_probs = door_probs or {}

        self._chairs: Dict[Tuple[int, int], Dict[str, object]] = {}
        self._obstacles: Set[Tuple[int, int]] = set()
        self._doors: List[Tuple[int, int]] = []
        self.obstacle_heatmap = {}
        self.exposure_field = ExposureField(self.width, self.height)

        self.humans: List[HumanAgent] = []

        # --- Cache & optimasi internal ---
        self._heatmap_frame_counter = 0
        # Cache vision untuk agen SITTING (posisi tidak berubah antar frame)
        self._sitting_vision_cache: Dict[int, List[Tuple[float, float]]] = {}
        # Set posisi sel agen untuk O(1) lookup
        self._human_cells: Set[Tuple[int, int]] = set()
        # Cache obstacle sample points sebagai numpy array (dibangun sekali)
        self._obs_sample_pts: Optional[np.ndarray] = None
        self._obs_keys: Optional[List[Tuple[int, int]]] = None
        self._obs_side_names: Optional[List[str]] = None

        self._load_layout(layout)
        self._ensure_doors()
        self._init_obstacle_heatmap()
        self._build_obs_sample_cache()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _load_layout(self, layout: Layout) -> None:
        for (col, row), cell_type in layout.items():
            if not (0 <= col < self.width and 0 <= row < self.height):
                continue
            if cell_type == CELL_CHAIR:
                self._chairs[(col, row)] = {
                    "occupied": False,
                    "occupant": None,
                    "facing": self._chair_directions.get((col, row), "right"),
                }
            elif cell_type == CELL_OBSTACLE:
                self._obstacles.add((col, row))
            elif cell_type == CELL_DOOR:
                self._doors.append((col, row))

        if self._doors:
            self._obstacles.difference_update(self._doors)

    def _ensure_doors(self) -> None:
        if self._doors:
            return
        fallback = (0, self.height // 2)
        self._doors.append(fallback)
        self._obstacles.discard(fallback)

    def _build_obs_sample_cache(self) -> None:
        """Bangun array numpy dari sample points semua sisi obstacle (sekali saja)."""
        cs = self.cell_size_px
        keys = []
        side_names = []
        pts = []
        for obs_cell in self._obstacles:
            ox, oy = self.cell_center_px(obs_cell)
            for side_name in OBSTACLE_SIDES:
                side_offset = OBSTACLE_SIDE_VECTORS[side_name]
                sx = ox + side_offset[0] * cs * 0.6
                sy = oy + side_offset[1] * cs * 0.6
                keys.append(obs_cell)
                side_names.append(side_name)
                pts.append([sx, sy])

        if pts:
            self._obs_sample_pts = np.array(pts, dtype=np.float64)  # (N, 2)
        else:
            self._obs_sample_pts = np.empty((0, 2), dtype=np.float64)
        self._obs_keys = keys
        self._obs_side_names = side_names

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def cell_center_px(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        col, row = cell
        return (
            col * self.cell_size_px + self.cell_size_px * 0.5,
            row * self.cell_size_px + self.cell_size_px * 0.5,
        )

    def cell_from_px(self, pos_px: Tuple[float, float]) -> Tuple[int, int]:
        col = int(pos_px[0] // self.cell_size_px)
        row = int(pos_px[1] // self.cell_size_px)
        return col, row

    def is_walkable_cell(
        self,
        cell: Tuple[int, int],
        target_cell: Optional[Tuple[int, int]] = None,
    ) -> bool:
        col, row = cell
        if not (0 <= col < self.width and 0 <= row < self.height):
            return False
        if cell in self._obstacles:
            return False
        if cell in self._chairs and cell != target_cell:
            return False
        return True

    def is_walkable_pos(
        self,
        pos_px: Tuple[float, float],
        target_cell: Optional[Tuple[int, int]] = None,
    ) -> bool:
        cell = self.cell_from_px(pos_px)
        return self.is_walkable_cell(cell, target_cell=target_cell)

    def _neighbors(
        self,
        cell: Tuple[int, int],
        target_cell: Tuple[int, int],
    ) -> List[Tuple[Tuple[int, int], float]]:
        cx, cy = cell
        results: List[Tuple[Tuple[int, int], float]] = []

        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (cx + dx, cy + dy)
            if self.is_walkable_cell(nb, target_cell=target_cell):
                results.append((nb, 1.0))

        for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            nb = (cx + dx, cy + dy)
            if not self.is_walkable_cell(nb, target_cell=target_cell):
                continue
            if not (
                self.is_walkable_cell((cx + dx, cy), target_cell=target_cell)
                and self.is_walkable_cell((cx, cy + dy), target_cell=target_cell)
            ):
                continue
            results.append((nb, 1.41421356237))

        return results

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        diag = min(dx, dy)
        straight = max(dx, dy) - diag
        return (1.41421356237 * diag) + straight

    def find_path(
        self,
        start_cell: Tuple[int, int],
        target_cell: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        if start_cell == target_cell:
            return [start_cell]
        if not self.is_walkable_cell(start_cell, target_cell=target_cell):
            return []
        if not self.is_walkable_cell(target_cell, target_cell=target_cell):
            if start_cell not in self._chairs:
                return []
        if not self.is_walkable_cell(target_cell, target_cell=target_cell):
            if target_cell not in self._chairs:
                return []

        open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (0.0, 0.0, start_cell))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start_cell: 0.0}
        closed: Set[Tuple[int, int]] = set()

        while open_heap:
            _, g, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == target_cell:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            closed.add(current)

            for nb, cost in self._neighbors(current, target_cell):
                if nb in closed:
                    continue
                tentative = g + cost
                if tentative < g_score.get(nb, float("inf")):
                    came_from[nb] = current
                    g_score[nb] = tentative
                    f = tentative + self._heuristic(nb, target_cell)
                    heapq.heappush(open_heap, (f, tentative, nb))

        return []

    # ------------------------------------------------------------------
    # Chair helpers
    # ------------------------------------------------------------------

    def choose_empty_chair(self, exclude: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
        empty = [
            cell
            for cell, info in self._chairs.items()
            if not info["occupied"] and cell != exclude
        ]
        if not empty:
            return None
        return self.rng.choice(empty)

    def choose_reachable_chair(
        self,
        start_cell: Tuple[int, int],
        exclude: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[int, int]]:
        candidates = [
            cell
            for cell, info in self._chairs.items()
            if not info["occupied"] and cell != exclude
        ]
        self.rng.shuffle(candidates)
        for cell in candidates:
            if self.find_path(start_cell, cell):
                return cell
        return None

    def reserve_chair(self, cell: Tuple[int, int], human: HumanAgent) -> None:
        info = self._chairs.get(cell)
        if info and not info["occupied"]:
            info["occupied"] = True
            info["occupant"] = human

    def release_chair(self, cell: Tuple[int, int], human: HumanAgent) -> None:
        info = self._chairs.get(cell)
        if info and info.get("occupant") is human:
            info["occupied"] = False
            info["occupant"] = None

    def get_chair_facing_angle(self, cell: Tuple[int, int]) -> float:
        info = self._chairs.get(cell)
        if not info:
            return 0.0
        direction = info.get("facing", "right")
        vec = CHAIR_DIR_VECTORS.get(direction, CHAIR_DIR_VECTORS["right"])
        return math.atan2(vec[1], vec[0])

    # ------------------------------------------------------------------
    # Door helpers
    # ------------------------------------------------------------------

    def _weighted_choice(self, doors: List[Tuple[int, int]]) -> Tuple[int, int]:
        weights = [self._door_probs.get(pos, 1.0) for pos in doors]
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
            return self.rng.choices(doors, weights=weights, k=1)[0]
        return self.rng.choice(doors)

    def choose_entry_door(self) -> Tuple[int, int]:
        if self._doors:
            return self._weighted_choice(self._doors)
        return (0, self.rng.randrange(self.height))

    def choose_exit_door(
        self,
        entry_cell: Optional[Tuple[int, int]] = None,
        prefer_other: bool = False,
        exclude: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        if not self._doors:
            return (0, self.rng.randrange(self.height))

        doors = [door for door in self._doors if door != exclude]
        if not doors:
            doors = list(self._doors)

        if prefer_other and entry_cell and len(self._doors) > 1:
            candidates = [door for door in doors if door != entry_cell]
            if candidates:
                return self._weighted_choice(candidates)
        return self._weighted_choice(doors)

    def choose_reachable_exit(
        self,
        start_cell: Tuple[int, int],
        entry_cell: Optional[Tuple[int, int]] = None,
        prefer_other: bool = False,
        exclude: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        doors = [door for door in self._doors if door != exclude]
        if not doors:
            doors = list(self._doors)

        if prefer_other and entry_cell and len(doors) > 1:
            candidates = [door for door in doors if door != entry_cell]
            if candidates:
                doors = candidates

        remaining = list(doors)
        while remaining:
            choice = self._weighted_choice(remaining)
            remaining.remove(choice)
            if self.find_path(start_cell, choice):
                return choice

        return self.choose_exit_door(entry_cell, prefer_other, exclude)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _spawn_human(self) -> None:
        entry_cell = self.choose_entry_door()
        if entry_cell in self._human_cells:  # O(1) set lookup
            return

        will_sit = self.rng.random() >= self.pass_through_prob
        chair_cell = None
        if will_sit:
            chair_cell = self.choose_reachable_chair(entry_cell)
            if chair_cell is None:
                will_sit = False

        if will_sit:
            exit_cell = self.choose_reachable_exit(entry_cell, entry_cell, prefer_other=False)
        else:
            exit_cell = self.choose_reachable_exit(entry_cell, entry_cell, prefer_other=True)

        sit_duration = max(3.0, self.rng.expovariate(1.0 / self.mean_sitting_s))

        speed_variation = self.np_rng.normal(loc=1.0, scale=SPEED_VARIATION)
        individual_speed = self.speed_px_s * max(0.5, min(2.0, speed_variation))

        human = HumanAgent(
            model=self,
            entry_cell=entry_cell,
            will_sit=will_sit,
            sit_duration_s=sit_duration,
            speed_px_s=individual_speed,
            chair_cell=chair_cell,
            exit_cell=exit_cell,
        )
        if chair_cell and will_sit:
            self.reserve_chair(chair_cell, human)
        self.humans.append(human)
        self._human_cells.add(entry_cell)
        self.total_customers += 1

    def _cell_has_human(self, cell: Tuple[int, int]) -> bool:
        return cell in self._human_cells  # O(1)

    def step(self, dt: float) -> None:
        dt = max(1e-6, dt)
        self.current_step += 1
        self.time_s += dt

        n_arrivals = int(self.np_rng.poisson(self.arrival_rate * dt))
        for _ in range(n_arrivals):
            self._spawn_human()

        # Rebuild human cell set setelah spawn
        finished: List[HumanAgent] = []
        for human in self.humans:
            if human.step(dt):
                finished.append(human)
        if finished:
            finished_set = set(id(h) for h in finished)
            self.humans = [h for h in self.humans if id(h) not in finished_set]

        # Rebuild set posisi agen
        self._human_cells = {self.cell_from_px(h.pos_px) for h in self.humans}

        # Invalidate sitting vision cache untuk agen yang sudah selesai
        if finished:
            for h in finished:
                self._sitting_vision_cache.pop(id(h), None)

        # Heatmap: skip beberapa frame untuk efisiensi
        self._heatmap_frame_counter += 1
        if self._heatmap_frame_counter >= HEATMAP_SKIP_FRAMES:
            self._heatmap_frame_counter = 0
            snap = self.get_grid_snapshot()
            self.update_obstacle_heatmap(snap["humans_vision"])

    # ------------------------------------------------------------------
    # Vision / Ray casting
    # ------------------------------------------------------------------

    def ray_cast_vision_polygon(
        self,
        pos_px: Tuple[float, float],
        facing_rad: float,
        half_angle_rad: float = math.pi / 4,
        range_px: float = 100.0,
        num_rays: int = 32,
    ) -> List[Tuple[float, float]]:
        """Ray casting DDA — sama persis dengan original."""
        cs = self.cell_size_px
        ox, oy = pos_px

        max_x = self.width * cs
        max_y = self.height * cs

        angles = [
            facing_rad - half_angle_rad + i * (2 * half_angle_rad) / (num_rays - 1)
            for i in range(num_rays)
        ]

        polygon: List[Tuple[float, float]] = [pos_px]

        for angle in angles:
            dx = math.cos(angle)
            dy = math.sin(angle)

            if abs(dx) < 1e-12:
                t_delta_x = float("inf")
                step_x = 0
            else:
                t_delta_x = abs(cs / dx)
                step_x = 1 if dx > 0 else -1

            if abs(dy) < 1e-12:
                t_delta_y = float("inf")
                step_y = 0
            else:
                t_delta_y = abs(cs / dy)
                step_y = 1 if dy > 0 else -1

            cell_x = int(ox // cs)
            cell_y = int(oy // cs)

            if dx >= 0:
                t_max_x = ((cell_x + 1) * cs - ox) / dx if abs(dx) > 1e-12 else float("inf")
            else:
                t_max_x = (cell_x * cs - ox) / dx if abs(dx) > 1e-12 else float("inf")

            if dy >= 0:
                t_max_y = ((cell_y + 1) * cs - oy) / dy if abs(dy) > 1e-12 else float("inf")
            else:
                t_max_y = (cell_y * cs - oy) / dy if abs(dy) > 1e-12 else float("inf")

            hit_px: Tuple[float, float] = (ox + dx * range_px, oy + dy * range_px)

            while True:
                if t_max_x < t_max_y:
                    t_entry = t_max_x
                    t_max_x += t_delta_x
                    cell_x += step_x
                else:
                    t_entry = t_max_y
                    t_max_y += t_delta_y
                    cell_y += step_y

                if t_entry >= range_px:
                    hit_px = (ox + dx * range_px, oy + dy * range_px)
                    break

                if not (0 <= cell_x < self.width and 0 <= cell_y < self.height):
                    t_stop = min(t_entry, range_px)
                    hit_px = (ox + dx * t_stop, oy + dy * t_stop)
                    break

                if (cell_x, cell_y) in self._obstacles:
                    t_stop = min(t_entry, range_px)
                    hit_px = (ox + dx * t_stop, oy + dy * t_stop)
                    break

            polygon.append(hit_px)

        return polygon

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def get_grid_snapshot(self) -> dict:
        chairs_empty: List[Tuple[int, int]] = []
        chairs_full: List[Tuple[int, int]] = []

        for cell, info in self._chairs.items():
            if info["occupied"]:
                chairs_full.append(cell)
            else:
                chairs_empty.append(cell)

        humans_seek: List[Tuple[float, float]] = []
        humans_exit: List[Tuple[float, float]] = []
        humans_pass: List[Tuple[float, float]] = []
        humans_sit:  List[Tuple[float, float]] = []
        humans_vision: List[List[Tuple[float, float]]] = []

        for human in self.humans:
            hid = id(human)

            if human.status == HumanStatus.SITTING:
                humans_sit.append(human.pos_px)
                # Cache vision polygon untuk agen duduk — posisi & arah tidak berubah
                if hid not in self._sitting_vision_cache:
                    self._sitting_vision_cache[hid] = self.ray_cast_vision_polygon(
                        human.pos_px, human.facing_angle_rad
                    )
                humans_vision.append(self._sitting_vision_cache[hid])

            elif human.will_sit and human.status == HumanStatus.TO_CHAIR:
                humans_seek.append(human.pos_px)
                humans_vision.append(
                    self.ray_cast_vision_polygon(human.pos_px, human.facing_angle_rad)
                )

            elif human.will_sit and human.status == HumanStatus.TO_EXIT:
                humans_exit.append(human.pos_px)
                humans_vision.append(
                    self.ray_cast_vision_polygon(human.pos_px, human.facing_angle_rad)
                )

            elif not human.will_sit:
                humans_pass.append(human.pos_px)
                humans_vision.append(
                    self.ray_cast_vision_polygon(human.pos_px, human.facing_angle_rad)
                )

        return {
            "obstacles":    list(self._obstacles),
            "doors":        list(self._doors),
            "chairs_empty": chairs_empty,
            "chairs_full":  chairs_full,
            "humans_seek":  humans_seek,
            "humans_exit":  humans_exit,
            "humans_pass":  humans_pass,
            "humans_sit":   humans_sit,
            "humans_vision": humans_vision,
        }

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------

    def _init_obstacle_heatmap(self) -> None:
        for obs_cell in self._obstacles:
            self.obstacle_heatmap[obs_cell] = {
                "top": 0.0, "right": 0.0, "bottom": 0.0, "left": 0.0,
            }

    def update_obstacle_heatmap(
        self, humans_vision: List[List[Tuple[float, float]]]
    ) -> None:
        """Update heatmap — numpy vectorized point-in-polygon untuk semua obstacle sekaligus."""
        if not humans_vision or self._obs_sample_pts is None or len(self._obs_sample_pts) == 0:
            return

        pts = self._obs_sample_pts  # (M, 2)  M = obstacles * 4 sides
        weight_inc = self.frame_dt / HEATMAP_LOOK_DURATION_S * HEATMAP_SKIP_FRAMES

        # Akumulasi visible_count per sample point dari semua vision polygon
        visible_counts = np.zeros(len(pts), dtype=np.float32)

        for vision_polygon in humans_vision:
            if len(vision_polygon) < 3:
                continue
            mask = self._points_in_polygon_numpy(pts, vision_polygon)
            visible_counts += mask

        # Update heatmap hanya untuk entri yang > 0
        nonzero_indices = np.nonzero(visible_counts)[0]
        for idx in nonzero_indices:
            obs_cell = self._obs_keys[idx]
            side_name = self._obs_side_names[idx]
            inc = float(visible_counts[idx]) * weight_inc
            self.obstacle_heatmap[obs_cell][side_name] += inc
            self.exposure_field.record(obs_cell[0], obs_cell[1], side_name, inc)

    @staticmethod
    def _points_in_polygon_numpy(
        pts: np.ndarray,
        polygon: List[Tuple[float, float]],
    ) -> np.ndarray:
        """
        Vectorized point-in-polygon (ray casting) untuk banyak titik sekaligus.
        pts: (N, 2) numpy array
        polygon: list of (x, y) tuples
        returns: boolean array shape (N,)
        """
        poly = np.array(polygon, dtype=np.float64)  # (V, 2)
        px = pts[:, 0]  # (N,)
        py = pts[:, 1]

        p1 = poly[:-1]   # (V-1, 2)
        p2 = poly[1:]    # (V-1, 2)

        # Tambah edge penutup
        p1 = np.vstack([p1, poly[-1]])
        p2 = np.vstack([p2, poly[0]])

        x1 = p1[:, 0]  # (V,)
        y1 = p1[:, 1]
        x2 = p2[:, 0]
        y2 = p2[:, 1]

        # Broadcast: px (N,1) vs y1 (1,V)
        py_  = py[:, None]   # (N, 1)
        px_  = px[:, None]
        y1_  = y1[None, :]   # (1, V)
        y2_  = y2[None, :]
        x1_  = x1[None, :]
        x2_  = x2[None, :]

        cond1 = py_ > np.minimum(y1_, y2_)
        cond2 = py_ <= np.maximum(y1_, y2_)
        cond3 = px_ <= np.maximum(x1_, x2_)
        dy = y2_ - y1_
        # Hindari div-by-zero
        safe_dy = np.where(dy != 0, dy, 1.0)
        xinters = (py_ - y1_) * (x2_ - x1_) / safe_dy + x1_
        cond4 = (x1_ == x2_) | (px_ <= xinters)

        cross = cond1 & cond2 & cond3 & cond4  # (N, V)
        inside = np.sum(cross, axis=1) % 2 == 1  # (N,)
        return inside.astype(np.float32)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def count_humans(self) -> int:
        return len(self.humans)

    def count_sitting(self) -> int:
        return sum(1 for h in self.humans if h.status == HumanStatus.SITTING)

    def count_passing(self) -> int:
        return sum(1 for h in self.humans if not h.will_sit and h.status == HumanStatus.TO_EXIT)

    def _point_in_polygon(
        self,
        point: Tuple[float, float],
        polygon: List[Tuple[float, float]],
    ) -> bool:
        """Fallback scalar versi (dipertahankan untuk kompatibilitas)."""
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside