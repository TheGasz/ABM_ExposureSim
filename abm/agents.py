"""
Modul abm/agents.py
"""

import math
from enum import Enum
from typing import List, Optional, Tuple, TYPE_CHECKING

from constants import ARRIVE_THRESHOLD_PX

if TYPE_CHECKING:
    from abm.model import WaitingRoomModel


class HumanStatus(Enum):
    TO_CHAIR = "to_chair"
    SITTING = "sitting"
    TO_EXIT = "to_exit"


class HumanAgent:
    """Kelas HumanAgent."""

    def __init__(
        self,
        model: "WaitingRoomModel",
        entry_cell: Tuple[int, int],
        will_sit: bool,
        sit_duration_s: float,
        speed_px_s: float,
        chair_cell: Optional[Tuple[int, int]] = None,
        exit_cell: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.model = model
        self.entry_cell = entry_cell
        self.will_sit = will_sit
        self.chair_cell = chair_cell
        self.exit_cell = exit_cell

        self.pos_px = model.cell_center_px(entry_cell)
        self.status = HumanStatus.TO_CHAIR if will_sit else HumanStatus.TO_EXIT
        self.sit_remaining_s = max(0.0, sit_duration_s)
        self.speed_px_s = speed_px_s

        self.path_cells: List[Tuple[int, int]] = []
        self.path_idx: int = 0
        self.path_target: Optional[Tuple[int, int]] = None
        self.last_cell = model.cell_from_px(self.pos_px)
        # Pusat sel terakhir yang secara fisik dicapai oleh agen.
        # Digunakan sebagai titik awal replanning untuk mencegah perpindahan mendadak (teleport).
        self.last_reached_cell: Tuple[int, int] = model.cell_from_px(self.pos_px)
        self.stuck_steps = 0
        # Arah hadap agen dalam radian (0 = kanan, pi/2 = bawah, dst).
        # Diperbarui secara dinamis saat pergerakan.
        self.facing_angle_rad: float = 0.0

        if self.status == HumanStatus.TO_CHAIR and self.chair_cell:
            self._plan_path(self.chair_cell)
        elif self.exit_cell:
            self._plan_path(self.exit_cell)

    def step(self, dt: float) -> bool:
        """Metode step."""
        if self.status == HumanStatus.TO_CHAIR:
            if not self.chair_cell:
                start_cell = self.last_reached_cell
                self.chair_cell = self.model.choose_reachable_chair(start_cell)
                if self.chair_cell:
                    self.model.reserve_chair(self.chair_cell, self)
                    self._plan_path(self.chair_cell)
                else:
                    self.status = HumanStatus.TO_EXIT
                    self.exit_cell = self.exit_cell or self.model.choose_reachable_exit(
                        start_cell, self.entry_cell, prefer_other=True
                    )
                    if self.exit_cell:
                        self._plan_path(self.exit_cell)
            if self.chair_cell and self._move_along_path(self.chair_cell, dt):
                self.pos_px = self.model.cell_center_px(self.chair_cell)
                self.last_reached_cell = self.chair_cell
                self.status = HumanStatus.SITTING
                self.facing_angle_rad = self.model.get_chair_facing_angle(self.chair_cell)

        elif self.status == HumanStatus.SITTING:
            self.sit_remaining_s -= dt
            if self.sit_remaining_s <= 0:
                if self.chair_cell:
                    self.model.release_chair(self.chair_cell, self)
                    # Mencari sel kosong terdekat dari kursi untuk proses berdiri.
                    start_cell = self._find_adjacent_walkable_cell(self.chair_cell)
                    if not start_cell:
                        start_cell = self.last_reached_cell
                    self.last_reached_cell = start_cell
                    self.pos_px = self.model.cell_center_px(start_cell) 
                    self.chair_cell = None
                else:
                    start_cell = self.last_reached_cell
                    
                self.exit_cell = self.exit_cell or self.model.choose_reachable_exit(
                    start_cell, self.entry_cell, prefer_other=False
                )
                if self.exit_cell:
                    self._plan_path(self.exit_cell)
                self.status = HumanStatus.TO_EXIT

        elif self.status == HumanStatus.TO_EXIT:
            if not self.exit_cell:
                start_cell = self.last_reached_cell
                self.exit_cell = self.model.choose_reachable_exit(
                    start_cell, self.entry_cell, prefer_other=True
                )
                if self.exit_cell:
                    self._plan_path(self.exit_cell)
            if self.exit_cell and self._move_along_path(self.exit_cell, dt):
                return True

        self._update_stuck_state()
        self._recover_if_stuck()
        return False

    def _update_stuck_state(self) -> None:
        cell = self.model.cell_from_px(self.pos_px)
        if cell == self.last_cell:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0
            self.last_cell = cell

    def _recover_if_stuck(self) -> None:
        if self.stuck_steps < self.model.stuck_threshold:
            return
        self.stuck_steps = 0
        start_cell = self.last_reached_cell

        if self.status == HumanStatus.TO_CHAIR:
            new_chair = self.model.choose_reachable_chair(start_cell, exclude=self.chair_cell)
            if new_chair:
                if self.chair_cell:
                    self.model.release_chair(self.chair_cell, self)
                self.chair_cell = new_chair
                self.model.reserve_chair(new_chair, self)
                self._plan_path(new_chair)
                return

            if self.chair_cell:
                self.model.release_chair(self.chair_cell, self)
                self.chair_cell = None
            self.status = HumanStatus.TO_EXIT
            self.exit_cell = self.model.choose_reachable_exit(
                start_cell, self.entry_cell, prefer_other=True, exclude=self.exit_cell
            )
            if self.exit_cell:
                self._plan_path(self.exit_cell)
            return

        if self.exit_cell and self._plan_path(self.exit_cell):
            return

        self.exit_cell = self.model.choose_reachable_exit(
            start_cell, self.entry_cell, prefer_other=True, exclude=self.exit_cell
        )
        if self.exit_cell:
            self._plan_path(self.exit_cell)

    def _plan_path(self, target_cell: Tuple[int, int]) -> bool:
        # Menggunakan last_reached_cell sebagai titik awal menggantikan posisi float saat ini
        # untuk mencegah artefak pergerakan instan (teleport) selama kalkulasi ulang path.
        start_cell = self.last_reached_cell
        path = self.model.find_path(start_cell, target_cell)
        if not path:
            # Cadangan: mencoba pencarian rute dari posisi float saat ini.
            start_cell = self.model.cell_from_px(self.pos_px)
            path = self.model.find_path(start_cell, target_cell)
        if not path:
            return False
        self.path_cells = path
        self.path_idx = 0
        self.path_target = target_cell
        return True

    def _move_along_path(self, target_cell: Tuple[int, int], dt: float) -> bool:
        if self.path_target != target_cell or not self.path_cells:
            if not self._plan_path(target_cell):
                return False

        remaining = self.speed_px_s * dt
        while remaining > 0 and self.path_idx < len(self.path_cells):
            next_cell = self.path_cells[self.path_idx]
            next_px = self.model.cell_center_px(next_cell)
            dx = next_px[0] - self.pos_px[0]
            dy = next_px[1] - self.pos_px[1]
            dist = math.hypot(dx, dy)

            if dist <= remaining:
                # Agen dapat mencapai atau melewati pusat sel pada langkah waktu saat ini.
                # Sesuaikan ke pusat sel dan catat sebagai titik tercapai.
                if dist > 1e-9:
                    self.facing_angle_rad = math.atan2(dy, dx)
                self.pos_px = next_px
                self.last_reached_cell = next_cell
                remaining -= dist
                self.path_idx += 1
            else:
                # Agen bergerak secara parsial menuju sel berikutnya.
                ratio = remaining / dist
                candidate = (self.pos_px[0] + dx * ratio, self.pos_px[1] + dy * ratio)
                if self.model.is_walkable_pos(candidate, target_cell):
                    self.pos_px = candidate
                if dist > 1e-9:
                    self.facing_angle_rad = math.atan2(dy, dx)
                # last_reached_cell tidak diubah hingga pusat sel baru benar-benar tercapai.
                remaining = 0.0

        return self.path_idx >= len(self.path_cells)
    
    def _find_adjacent_walkable_cell(self, cell: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Metode _find_adjacent_walkable_cell."""
        col, row = cell
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            adjacent = (col + dx, row + dy)
            if self.model.is_walkable_cell(adjacent):
                return adjacent
        return None