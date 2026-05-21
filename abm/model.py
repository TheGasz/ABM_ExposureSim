"""
abm/model.py
============
Model ABM utama: WaitingRoomModel.

Tanggung jawab model:
  • Menginisialisasi grid Mesa (MultiGrid) dari layout editor.
  • Menyimpan dua heatmap akumulatif (floor & obstacle).
  • Mengelola kedatangan pelanggan stokastik (distribusi Poisson).
  • Menjalankan satu step simulasi: spawn + step semua CustomerAgent.
  • Mendukung pintu masuk (CELL_DOOR) yang bisa diatur posisinya.
  • Mendukung arah hadap kursi (chair facing).
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from mesa import Model
from mesa.space import MultiGrid

from constants import CELL_CHAIR, CELL_OBSTACLE, CELL_DOOR, Layout, ChairDirections
from abm.agents import ChairAgent, CustomerAgent, CustomerStatus, DoorAgent, ObstacleAgent


class WaitingRoomModel(Model):
    """
    Model Mesa untuk simulasi ruang tunggu.

    Parameter
    ---------
    width, height : int
        Dimensi grid (kolom x baris).
    layout : Layout
        Dict hasil editor: {(col, row): CELL_CHAIR | CELL_OBSTACLE | CELL_DOOR}.
    chair_directions : ChairDirections
        Dict arah hadap kursi: {(col, row): "up"|"down"|"left"|"right"}.
    arrival_rate : float
        Lambda distribusi Poisson untuk jumlah pelanggan per step.
    mean_sitting : float
        Rata-rata durasi duduk dalam step.
    gaze_range : int
        Panjang maksimal ray sorot mata.
    seed : int
        Seed untuk reproducibility.
    """

    def __init__(
        self,
        width: int,
        height: int,
        layout: Layout,
        chair_directions: ChairDirections = None,
        arrival_rate: float = 0.35,
        mean_sitting: float = 18.0,
        gaze_range: int     = 8,
        seed: int           = 42,
    ):
        super().__init__()

        self.width  = width
        self.height = height

        self.arrival_rate = arrival_rate
        self.mean_sitting = mean_sitting
        self.gaze_range   = gaze_range

        self.current_step    = 0
        self.total_customers = 0

        self.grid = MultiGrid(width, height, torus=False)

        self.floor_heatmap    = np.zeros((width, height), dtype=np.float64)
        self.obstacle_heatmap = np.zeros((width, height), dtype=np.float64)

        random.seed(seed)
        np.random.seed(seed)

        self._chairs:    List[ChairAgent]    = []
        self._obstacles: List[ObstacleAgent] = []
        self._doors:     List[DoorAgent]     = []

        self._chair_directions = chair_directions or {}

        self._load_layout(layout)

    def _load_layout(self, layout: Layout) -> None:
        """Tempatkan ChairAgent, ObstacleAgent, DoorAgent dari layout editor."""
        for (col, row), cell_type in layout.items():
            if not (0 <= col < self.width and 0 <= row < self.height):
                continue

            pos = (col, row)

            if cell_type == CELL_CHAIR:
                facing = self._chair_directions.get((col, row), "right")
                agent = ChairAgent(self, facing=facing)
                self.grid.place_agent(agent, pos)
                self._chairs.append(agent)

            elif cell_type == CELL_OBSTACLE:
                agent = ObstacleAgent(self)
                self.grid.place_agent(agent, pos)
                self._obstacles.append(agent)

            elif cell_type == CELL_DOOR:
                agent = DoorAgent(self)
                self.grid.place_agent(agent, pos)
                self._doors.append(agent)

    def _get_spawn_positions(self) -> List[Tuple[int, int]]:
        """
        Kembalikan daftar posisi spawn pelanggan.

        Jika ada CELL_DOOR di layout, gunakan posisi pintu.
        Jika tidak ada pintu (backward compat), fallback ke kolom 0.
        """
        if self._doors:
            return [door.pos for door in self._doors]
        # Fallback: semua sel di kolom 0 yang tidak terblokir
        return [(0, r) for r in range(self.height)]

    def _spawn_customer(self) -> None:
        """Spawn satu CustomerAgent di salah satu pintu masuk."""
        spawn_positions = self._get_spawn_positions()
        if not spawn_positions:
            return

        spawn_pos = random.choice(spawn_positions)

        # Pastikan sel spawn tidak diblokir obstacle atau customer
        contents = self.grid.get_cell_list_contents([spawn_pos])
        if any(isinstance(a, (ObstacleAgent, CustomerAgent)) for a in contents):
            return

        sitting_dur = max(3, int(random.expovariate(1.0 / self.mean_sitting)))
        agent = CustomerAgent(self, sitting_dur, self.gaze_range)
        self.grid.place_agent(agent, spawn_pos)
        self.total_customers += 1

    def step(self) -> None:
        """Satu langkah simulasi: spawn Poisson + step semua CustomerAgent."""
        self.current_step += 1

        n_arrivals = int(np.random.poisson(self.arrival_rate))
        for _ in range(n_arrivals):
            self._spawn_customer()

        customers = [a for a in self.agents if isinstance(a, CustomerAgent)]
        random.shuffle(customers)
        for customer in customers:
            customer.step()

    def count_customers(self) -> int:
        return sum(1 for a in self.agents if isinstance(a, CustomerAgent))

    def get_grid_snapshot(self) -> dict:
        """Kembalikan snapshot posisi semua agen dan sel gaze aktif."""
        obs, ch_e, ch_f = [], [], []
        doors = []
        c_seek, c_move, c_sit = [], [], []
        gaze_floor = set()
        gaze_obstacle = set()
        # Juga track arah hadap kursi untuk visualisasi
        chair_facings = {}

        for contents, (x, y) in self.grid.coord_iter():
            for agent in contents:
                if isinstance(agent, ObstacleAgent):
                    obs.append((x, y))
                elif isinstance(agent, DoorAgent):
                    doors.append((x, y))
                elif isinstance(agent, ChairAgent):
                    (ch_f if agent.occupied else ch_e).append((x, y))
                    chair_facings[(x, y)] = agent.facing
                elif isinstance(agent, CustomerAgent):
                    s = agent.status
                    if s == CustomerStatus.SEEKING:
                        c_seek.append((x, y))
                    elif s == CustomerStatus.MOVING:
                        c_move.append((x, y))
                    elif s == CustomerStatus.SITTING:
                        c_sit.append((x, y))
                        gaze_floor.update(agent.gaze_floor_cells)
                        gaze_obstacle.update(agent.gaze_obstacle_cells)

        return dict(
            obstacles      = obs,
            doors          = doors,
            chairs_empty   = ch_e,
            chairs_full    = ch_f,
            chair_facings  = chair_facings,
            customers_seek = c_seek,
            customers_move = c_move,
            customers_sit  = c_sit,
            gaze_floor     = list(gaze_floor),
            gaze_obstacle  = list(gaze_obstacle),
        )