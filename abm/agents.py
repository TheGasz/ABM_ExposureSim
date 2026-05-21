"""
abm/agents.py
=============
Definisi seluruh kelas Agent untuk simulasi ABM Ruang Tunggu.

Hierarki:
    Agent (Mesa)
    ├── ObstacleAgent   — tiang/pot, diam, memblokir ray & bisa ditempel iklan
    ├── DoorAgent       — pintu masuk, marker posisi spawn pelanggan
    ├── ChairAgent      — kursi, diam, punya status kosong/terisi + arah hadap
    └── CustomerAgent   — pelanggan, bergerak, memancarkan sorot mata (raycasting)
"""

import heapq
import math
from collections import deque
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from mesa import Agent

from constants import CELL_OBSTACLE, CELL_CHAIR, CELL_DOOR, CHAIR_DIR_VECTORS

if TYPE_CHECKING:
    from abm.model import WaitingRoomModel


class CustomerStatus(Enum):
    SEEKING = "seeking"
    MOVING  = "moving"
    SITTING = "sitting"
    LEAVING = "leaving"


class ObstacleAgent(Agent):
    """Tiang / pot / tembok pendek. Statis, memblokir ray."""
    def __init__(self, model):
        super().__init__(model)
    def step(self):
        pass


class DoorAgent(Agent):
    """Pintu masuk ruangan. Statis, menandai spawn point."""
    def __init__(self, model):
        super().__init__(model)
    def step(self):
        pass


class ChairAgent(Agent):
    """Kursi dengan arah hadap (facing)."""
    def __init__(self, model, facing: str = "right"):
        super().__init__(model)
        self.occupied: bool = False
        self.occupant: Optional["CustomerAgent"] = None
        self.facing: str = facing

    def step(self):
        pass


class CustomerAgent(Agent):
    """Pelanggan: masuk, cari kursi, duduk, sorot mata, pergi."""

    def __init__(self, model, sitting_duration: int, gaze_range: int):
        super().__init__(model)
        self.status = CustomerStatus.SEEKING
        self.target_chair: Optional[ChairAgent] = None
        self.sitting_timer = sitting_duration
        self.gaze_range = gaze_range
        self._path: List[Tuple[int, int]] = []
        self._path_idx: int = 0
        self.gaze_floor_cells: List[Tuple[int, int]] = []
        self.gaze_obstacle_cells: List[Tuple[int, int]] = []

    # -- Navigation helpers --

    def _is_walkable(self, pos):
        grid = self.model.grid
        if not (0 <= pos[0] < grid.width and 0 <= pos[1] < grid.height):
            return False
        contents = grid.get_cell_list_contents([pos])
        return not any(isinstance(a, ObstacleAgent) for a in contents)

    def _can_move_diagonal(self, x, y, dx, dy):
        if dx == 0 or dy == 0:
            return True
        a_blocked = not self._is_walkable((x + dx, y))
        b_blocked = not self._is_walkable((x, y + dy))
        return not (a_blocked and b_blocked)

    def _find_path(self, start, goal):
        """A* pathfinding menghindari obstacle, cek diagonal gap."""
        grid = self.model.grid

        def h(a, b):
            return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            cx, cy = current
            for nb in grid.get_neighborhood(current, moore=True, include_center=False):
                nx, ny = nb
                ddx, ddy = nx - cx, ny - cy
                if not self._is_walkable(nb) and nb != goal:
                    continue
                if not self._can_move_diagonal(cx, cy, ddx, ddy):
                    continue
                cost = 1.414 if (ddx != 0 and ddy != 0) else 1.0
                tg = g_score[current] + cost
                if tg < g_score.get(nb, float('inf')):
                    came_from[nb] = current
                    g_score[nb] = tg
                    heapq.heappush(open_set, (tg + h(nb, goal), nb))
        return []

    def _find_nearest_empty_chair(self):
        """BFS + A* untuk menemukan kursi kosong terdekat yang reachable."""
        grid = self.model.grid
        visited = {self.pos}
        queue = deque([self.pos])
        while queue:
            cur = queue.popleft()
            for agent in grid.get_cell_list_contents([cur]):
                if isinstance(agent, ChairAgent) and not agent.occupied:
                    path = self._find_path(self.pos, agent.pos)
                    if path:
                        self._path = path
                        self._path_idx = 1
                        return agent
            for nb in grid.get_neighborhood(cur, moore=True, include_center=False):
                if nb not in visited and self._is_walkable(nb):
                    visited.add(nb)
                    queue.append(nb)
        return None

    def _step_toward(self, target):
        """Gerak satu langkah mengikuti path A*."""
        if self._path and self._path_idx < len(self._path):
            next_pos = self._path[self._path_idx]
            contents = self.model.grid.get_cell_list_contents([next_pos])
            is_target = self.target_chair and next_pos == self.target_chair.pos
            has_cust = any(isinstance(a, CustomerAgent) and a is not self for a in contents)
            if is_target or (self._is_walkable(next_pos) and not has_cust):
                self.model.grid.move_agent(self, next_pos)
                self._path_idx += 1
                return
            return  # Tunggu jika terblokir customer

        # Fallback: hitung path baru
        new_path = self._find_path(self.pos, target)
        if new_path and len(new_path) > 1:
            self._path = new_path
            self._path_idx = 1
            self._step_toward(target)

    # -- Raycasting --

    def _cast_gaze(self):
        """Pancarkan 3 ray sorot mata mengikuti arah hadap kursi."""
        self.gaze_floor_cells = []
        self.gaze_obstacle_cells = []
        ax, ay = self.pos

        if self.target_chair and hasattr(self.target_chair, 'facing'):
            dir_vec = CHAIR_DIR_VECTORS.get(self.target_chair.facing, (1, 0))
            dx_main, dy_main = float(dir_vec[0]), float(dir_vec[1])
        else:
            cx = self.model.grid.width // 2
            cy = self.model.grid.height // 2
            dx_main = float(cx - ax)
            dy_main = float(cy - ay)
            length = math.hypot(dx_main, dy_main)
            if length < 1e-9:
                return
            dx_main /= length
            dy_main /= length

        for angle_deg in [-45, 0, 45]:
            rad = math.radians(angle_deg)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            rdx = dx_main * cos_a - dy_main * sin_a
            rdy = dx_main * sin_a + dy_main * cos_a
            self._trace_ray(ax, ay, rdx, rdy)

    def _trace_ray(self, ox, oy, dx, dy):
        """Telusuri satu ray, update heatmap."""
        grid = self.model.grid
        f_heat = self.model.floor_heatmap
        o_heat = self.model.obstacle_heatmap
        prev = (-9999, -9999)

        for i in range(1, self.gaze_range + 1):
            nx = int(round(ox + dx * i))
            ny = int(round(oy + dy * i))
            if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                break
            cell = (nx, ny)
            if cell == prev:
                continue
            prev = cell
            contents = grid.get_cell_list_contents([cell])
            if any(isinstance(a, ObstacleAgent) for a in contents):
                o_heat[nx][ny] += 1
                self.gaze_obstacle_cells.append(cell)
                break
            f_heat[nx][ny] += 1
            self.gaze_floor_cells.append(cell)

    # -- Step utama --

    def step(self):
        if self.status == CustomerStatus.SEEKING:
            chair = self._find_nearest_empty_chair()
            if chair:
                self.target_chair = chair
                chair.occupied = True
                chair.occupant = self
                self.status = CustomerStatus.MOVING

        elif self.status == CustomerStatus.MOVING:
            if self.pos == self.target_chair.pos:
                self.status = CustomerStatus.SITTING
            else:
                self._step_toward(self.target_chair.pos)

        elif self.status == CustomerStatus.SITTING:
            self._cast_gaze()
            self.sitting_timer -= 1
            if self.sitting_timer <= 0:
                if self.target_chair:
                    self.target_chair.occupied = False
                    self.target_chair.occupant = None
                self.status = CustomerStatus.LEAVING
                self.gaze_floor_cells = []
                self.gaze_obstacle_cells = []

        elif self.status == CustomerStatus.LEAVING:
            self.model.grid.remove_agent(self)
            self.remove()