"""
abm/model.py
============
Main simulation model for the waiting room.
"""

import heapq
import math
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from constants import (
    CELL_CHAIR,
    CELL_DOOR,
    CELL_OBSTACLE,
    ChairDirections,
    Layout,
    SPEED_PX_PER_S,
    CELL_SIZE_PX,
)
from abm.agents import HumanAgent, HumanStatus


class WaitingRoomModel:
    """Lightweight, grid-backed model with continuous movement in pixels."""

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
        speed_px_s: float = SPEED_PX_PER_S,
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
        self.speed_px_s = max(1e-3, speed_px_s)
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

        self.humans: List[HumanAgent] = []

        self._load_layout(layout)
        self._ensure_doors()

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
        if self._cell_has_human(entry_cell):
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

        human = HumanAgent(
            model=self,
            entry_cell=entry_cell,
            will_sit=will_sit,
            sit_duration_s=sit_duration,
            speed_px_s=self.speed_px_s,
            chair_cell=chair_cell,
            exit_cell=exit_cell,
        )
        # Reserve chair after instantiation to prevent double booking.
        if chair_cell and will_sit:
            self.reserve_chair(chair_cell, human)
        self.humans.append(human)
        self.total_customers += 1

    def _cell_has_human(self, cell: Tuple[int, int]) -> bool:
        for human in self.humans:
            if self.cell_from_px(human.pos_px) == cell:
                return True
        return False

    def step(self, dt: float) -> None:
        dt = max(1e-6, dt)
        self.current_step += 1
        self.time_s += dt

        n_arrivals = int(self.np_rng.poisson(self.arrival_rate * dt))
        for _ in range(n_arrivals):
            self._spawn_human()

        finished: List[HumanAgent] = []
        for human in self.humans:
            if human.step(dt):
                finished.append(human)
        if finished:
            self.humans = [h for h in self.humans if h not in finished]

    # ------------------------------------------------------------------
    # Metrics and snapshot
    # ------------------------------------------------------------------

    def count_humans(self) -> int:
        return len(self.humans)

    def count_sitting(self) -> int:
        return sum(1 for h in self.humans if h.status == HumanStatus.SITTING)

    def count_passing(self) -> int:
        return sum(1 for h in self.humans if not h.will_sit and h.status == HumanStatus.TO_EXIT)

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
        humans_sit: List[Tuple[float, float]] = []

        for human in self.humans:
            if human.status == HumanStatus.SITTING:
                humans_sit.append(human.pos_px)
            elif human.will_sit and human.status == HumanStatus.TO_CHAIR:
                humans_seek.append(human.pos_px)
            elif human.will_sit and human.status == HumanStatus.TO_EXIT:
                humans_exit.append(human.pos_px)
            elif not human.will_sit:
                humans_pass.append(human.pos_px)

        return {
            "obstacles": list(self._obstacles),
            "doors": list(self._doors),
            "chairs_empty": chairs_empty,
            "chairs_full": chairs_full,
            "humans_seek": humans_seek,
            "humans_exit": humans_exit,
            "humans_pass": humans_pass,
            "humans_sit": humans_sit,
        }
