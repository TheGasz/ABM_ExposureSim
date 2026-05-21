"""
abm/model.py
============
Model ABM utama: WaitingRoomModel.

Tanggung jawab model:
  • Menginisialisasi grid Mesa (MultiGrid) dari layout editor.
  • Menyimpan dua heatmap akumulatif (floor & obstacle).
  • Mengelola kedatangan pelanggan stokastik (distribusi Poisson).
  • Menjalankan satu step simulasi: spawn + step semua CustomerAgent.

Alur data:
    Editor (ui/panel_design.py)
        → layout dict {(col,row): CELL_TYPE}
            → WaitingRoomModel.__init__(_load_layout)
                → ChairAgent / ObstacleAgent ditempatkan di grid
                    → CustomerAgent di-spawn tiap step
                        → raycasting mengisi floor_heatmap / obstacle_heatmap
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from mesa import Model
from mesa.space import MultiGrid

from constants import CELL_CHAIR, CELL_OBSTACLE, Layout
from abm.agents import ChairAgent, CustomerAgent, ObstacleAgent


class WaitingRoomModel(Model):
    """
    Model Mesa untuk simulasi ruang tunggu.

    Parameter
    ---------
    width, height : int
        Dimensi grid (kolom × baris). Ditetapkan dari editor denah.
    layout : Layout
        Dict hasil editor: {(col, row): CELL_CHAIR | CELL_OBSTACLE}.
        Elemen statis ditempatkan sekali di awal, tidak berubah selama simulasi.
    arrival_rate : float
        Lambda distribusi Poisson untuk jumlah pelanggan yang datang per step.
        n_arrivals ~ Poisson(arrival_rate).
    mean_sitting : float
        Rata-rata durasi duduk dalam step.
        sitting_duration ~ Exponential(mean_sitting), minimum 3 step.
    gaze_range : int
        Panjang maksimal ray sorot mata (dalam jumlah sel grid).
    seed : int
        Seed untuk Python random dan numpy.random agar reproducible.

    Atribut Publik
    --------------
    floor_heatmap    : np.ndarray shape (width, height)
        Akumulasi exposure sel lantai — updated tiap CustomerAgent._trace_ray().
    obstacle_heatmap : np.ndarray shape (width, height)
        Akumulasi exposure sel tiang — updated saat ray berhenti di obstacle.
    current_step     : int
        Jumlah step yang sudah dijalankan.
    total_customers  : int
        Total pelanggan yang pernah di-spawn sejak model dibuat.
    """

    def __init__(
        self,
        width: int,
        height: int,
        layout: Layout,
        arrival_rate: float = 0.35,
        mean_sitting: float = 18.0,
        gaze_range: int     = 8,
        seed: int           = 42,
    ):
        super().__init__()

        # Dimensi ruangan (fixed dari editor)
        self.width        = width
        self.height       = height

        # Parameter stokastik
        self.arrival_rate = arrival_rate
        self.mean_sitting = mean_sitting
        self.gaze_range   = gaze_range

        # Counter
        self.current_step    = 0
        self.total_customers = 0

        # Grid Mesa (MultiGrid: bisa multi-agen per sel, torus=False)
        self.grid = MultiGrid(width, height, torus=False)

        # Dua heatmap akumulatif — diisi oleh CustomerAgent._trace_ray()
        self.floor_heatmap    = np.zeros((width, height), dtype=np.float64)
        self.obstacle_heatmap = np.zeros((width, height), dtype=np.float64)

        # Seed random untuk reproducibility
        random.seed(seed)
        np.random.seed(seed)

        # Referensi ke agen statis (untuk debug / query cepat)
        self._chairs:    List[ChairAgent]    = []
        self._obstacles: List[ObstacleAgent] = []

        # Tempatkan kursi & halangan dari layout editor
        self._load_layout(layout)

    # =========================================================================
    # INISIALISASI DARI LAYOUT EDITOR
    # =========================================================================

    def _load_layout(self, layout: Layout) -> None:
        """
        Tempatkan ChairAgent dan ObstacleAgent berdasarkan layout dari editor.

        Hanya sel yang berada dalam batas grid yang diproses.
        Elemen di luar batas (misal akibat resize grid) diabaikan.
        """
        for (col, row), cell_type in layout.items():
            if not (0 <= col < self.width and 0 <= row < self.height):
                continue   # Abaikan elemen di luar batas grid saat ini

            pos = (col, row)

            if cell_type == CELL_CHAIR:
                agent = ChairAgent(self)
                self.grid.place_agent(agent, pos)
                self._chairs.append(agent)

            elif cell_type == CELL_OBSTACLE:
                agent = ObstacleAgent(self)
                self.grid.place_agent(agent, pos)
                self._obstacles.append(agent)

    # =========================================================================
    # SPAWN PELANGGAN
    # =========================================================================

    def _spawn_customer(self) -> None:
        """
        Spawn satu CustomerAgent baru di kolom 0 (pintu masuk kiri).

        Posisi baris dipilih secara acak. Jika sel spawn ditempati
        ObstacleAgent, spawn dibatalkan untuk step ini.

        Durasi duduk diambil dari distribusi Eksponensial:
            sitting_duration ~ max(3, Exponential(mean_sitting))
        Distribusi eksponensial sesuai dengan model antrian nyata (waktu layanan).
        """
        spawn_row = random.randint(0, self.height - 1)
        spawn_pos = (0, spawn_row)

        # Pastikan sel spawn tidak diblokir obstacle
        contents = self.grid.get_cell_list_contents([spawn_pos])
        if any(isinstance(a, ObstacleAgent) for a in contents):
            return   # Pintu terblokir, coba lagi di step berikutnya

        sitting_dur = max(3, int(random.expovariate(1.0 / self.mean_sitting)))
        agent = CustomerAgent(self, sitting_dur, self.gaze_range)
        self.grid.place_agent(agent, spawn_pos)
        self.total_customers += 1

    # =========================================================================
    # STEP MODEL
    # =========================================================================

    def step(self) -> None:
        """
        Satu langkah simulasi:

        1. Kedatangan stokastik:
               n_arrivals ~ Poisson(arrival_rate)
           Untuk setiap arrival, panggil _spawn_customer().

        2. Eksekusi CustomerAgent:
           Urutan acak (random.shuffle) mencegah bias posisi antrian.
           ObstacleAgent dan ChairAgent tidak di-step (statis).

        Catatan Mesa 3.x:
           Tidak menggunakan RandomActivation scheduler.
           agents_by_type bisa bermasalah jika class didefinisikan di scope
           berbeda, sehingga digunakan filter isinstance yang lebih aman.
        """
        self.current_step += 1

        # ── Kedatangan Poisson ────────────────────────────────────────────────
        n_arrivals = int(np.random.poisson(self.arrival_rate))
        for _ in range(n_arrivals):
            self._spawn_customer()

        # ── Step CustomerAgent secara acak ────────────────────────────────────
        customers = [a for a in self.agents if isinstance(a, CustomerAgent)]
        random.shuffle(customers)
        for customer in customers:
            customer.step()

    # =========================================================================
    # QUERY HELPERS
    # =========================================================================

    def count_customers(self) -> int:
        """Jumlah CustomerAgent yang masih aktif di grid."""
        return sum(1 for a in self.agents if isinstance(a, CustomerAgent))

    def get_grid_snapshot(self) -> dict:
        """
        Kembalikan snapshot posisi semua agen dan sel gaze aktif.

        Digunakan oleh viz/sim_plots.py untuk merender kondisi ruangan.

        Returns
        -------
        dict dengan key:
            obstacles       : List[(x,y)] posisi ObstacleAgent
            chairs_empty    : List[(x,y)] ChairAgent yang kosong
            chairs_full     : List[(x,y)] ChairAgent yang terisi
            customers_seek  : List[(x,y)] CustomerAgent status SEEKING
            customers_move  : List[(x,y)] CustomerAgent status MOVING
            customers_sit   : List[(x,y)] CustomerAgent status SITTING
            gaze_floor      : List[(x,y)] sel lantai yang disorot step ini
            gaze_obstacle   : List[(x,y)] sel tiang yang disorot step ini
        """
        obs, ch_e, ch_f           = [], [], []
        c_seek, c_move, c_sit     = [], [], []
        gaze_floor   = set()
        gaze_obstacle= set()

        for contents, (x, y) in self.grid.coord_iter():
            for agent in contents:
                if isinstance(agent, ObstacleAgent):
                    obs.append((x, y))
                elif isinstance(agent, ChairAgent):
                    (ch_f if agent.occupied else ch_e).append((x, y))
                elif isinstance(agent, CustomerAgent):
                    s = agent.status
                    from abm.agents import CustomerStatus
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
            chairs_empty   = ch_e,
            chairs_full    = ch_f,
            customers_seek = c_seek,
            customers_move = c_move,
            customers_sit  = c_sit,
            gaze_floor     = list(gaze_floor),
            gaze_obstacle  = list(gaze_obstacle),
        )