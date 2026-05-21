"""
abm/agents.py
=============
Definisi seluruh kelas Agent untuk simulasi ABM Ruang Tunggu.

Hierarki:
    Agent (Mesa)
    ├── ObstacleAgent   — tiang/pot, diam, memblokir ray & bisa ditempel iklan
    ├── ChairAgent      — kursi, diam, punya status kosong/terisi
    └── CustomerAgent   — pelanggan, bergerak, memancarkan sorot mata (raycasting)

Siklus hidup CustomerAgent:
    SEEKING → MOVING → SITTING → LEAVING

Algoritma utama (ada di CustomerAgent):
    _cast_gaze()   : menentukan 3 arah ray dari posisi pelanggan
    _trace_ray()   : menelusuri satu ray sel-per-sel (Bresenham-inspired)
"""

import math
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from mesa import Agent

from constants import CELL_OBSTACLE, CELL_CHAIR

if TYPE_CHECKING:
    # Hanya untuk type-hint, tidak menyebabkan circular import
    from abm.model import WaitingRoomModel


# =============================================================================
# ENUM STATUS PELANGGAN
# =============================================================================

class CustomerStatus(Enum):
    SEEKING = "seeking"   # Mencari kursi kosong
    MOVING  = "moving"    # Bergerak menuju kursi
    SITTING = "sitting"   # Duduk, memancarkan sorot mata
    LEAVING = "leaving"   # Akan meninggalkan ruangan


# =============================================================================
# OBSTACLE AGENT
# =============================================================================

class ObstacleAgent(Agent):
    """
    Tiang / pot tanaman / tembok pendek.

    Dua fungsi dalam simulasi:
      1. Memblokir ray sorot mata CustomerAgent (line-of-sight blocker).
      2. Sendiri bisa ditempel iklan — saat ray BERHENTI di sel obstacle,
         obstacle_heatmap[x][y] += 1 (diproses dari CustomerAgent._trace_ray).

    Agent ini tidak bergerak dan tidak punya logika step.
    """

    def __init__(self, model: "WaitingRoomModel"):
        super().__init__(model)

    def step(self):
        pass   # Obstacle statis, tidak ada aksi


# =============================================================================
# CHAIR AGENT
# =============================================================================

class ChairAgent(Agent):
    """
    Kursi di dalam ruang tunggu.

    Atribut:
        occupied (bool)            : True jika ada pelanggan yang duduk.
        occupant (CustomerAgent|None): Referensi ke pelanggan yang menduduki kursi.

    Agent ini tidak bergerak. Status occupied diubah oleh CustomerAgent
    saat ia mengklaim kursi (SEEKING→MOVING) atau meninggalkan kursi (LEAVING).
    """

    def __init__(self, model: "WaitingRoomModel"):
        super().__init__(model)
        self.occupied: bool = False
        self.occupant: Optional["CustomerAgent"] = None

    def step(self):
        pass   # Kursi statis, tidak ada aksi


# =============================================================================
# CUSTOMER AGENT
# =============================================================================

class CustomerAgent(Agent):
    """
    Pelanggan yang masuk, mencari kursi, duduk, lalu pergi.

    Siklus Hidup
    ------------
    SEEKING  → Cari kursi kosong terdekat via BFS. Jika ditemukan, klaim
               kursi dan beralih ke MOVING.
    MOVING   → Langkah satu sel per step menuju posisi kursi target.
               Sesampainya, beralih ke SITTING.
    SITTING  → Tiap step: pancarkan 3 ray sorot mata (_cast_gaze).
               Hitung mundur sitting_timer. Saat 0, beralih ke LEAVING.
    LEAVING  → Hapus diri dari grid dan dari model (Mesa 3.x: self.remove()).

    Parameter
    ---------
    sitting_duration : int   — jumlah step yang dihabiskan untuk duduk
    gaze_range       : int   — panjang maksimal ray (dalam jumlah sel)
    """

    def __init__(
        self,
        model: "WaitingRoomModel",
        sitting_duration: int,
        gaze_range: int,
    ):
        super().__init__(model)
        self.status         = CustomerStatus.SEEKING
        self.target_chair: Optional[ChairAgent] = None
        self.sitting_timer  = sitting_duration
        self.gaze_range     = gaze_range

        # Daftar sel yang sedang disorot pada step ini (untuk visualisasi)
        self.gaze_floor_cells:    List[Tuple[int, int]] = []
        self.gaze_obstacle_cells: List[Tuple[int, int]] = []

    # =========================================================================
    # LOGIKA NAVIGASI
    # =========================================================================

    def _find_nearest_empty_chair(self) -> Optional[ChairAgent]:
        """
        BFS dari posisi saat ini untuk menemukan ChairAgent kosong terdekat.

        Mengembalikan ChairAgent pertama yang ditemukan, atau None jika
        tidak ada kursi kosong yang bisa dijangkau.
        """
        grid    = self.model.grid
        visited = {self.pos}
        queue   = [self.pos]

        while queue:
            cur = queue.pop(0)
            for agent in grid.get_cell_list_contents([cur]):
                if isinstance(agent, ChairAgent) and not agent.occupied:
                    return agent
            for nb in grid.get_neighborhood(cur, moore=True, include_center=False):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return None

    def _step_toward(self, target: Tuple[int, int]):
        """
        Gerak satu langkah (Moore neighborhood) ke arah target.

        Mencoba tiga kandidat posisi secara berurutan:
          1. Diagonal penuh  (dx, dy)
          2. Horizontal saja (dx, 0)
          3. Vertikal saja   (0, dy)
        Sel yang mengandung ObstacleAgent atau CustomerAgent lain dihindari,
        kecuali sel itu adalah posisi kursi target.
        """
        x, y   = self.pos
        tx, ty = target
        dx, dy = int(np.sign(tx - x)), int(np.sign(ty - y))

        for nx, ny in [(x + dx, y + dy), (x + dx, y), (x, y + dy)]:
            if not (0 <= nx < self.model.grid.width and
                    0 <= ny < self.model.grid.height):
                continue
            cell     = (nx, ny)
            contents = self.model.grid.get_cell_list_contents([cell])
            is_chair_target = self.target_chair and cell == self.target_chair.pos
            blocked  = any(isinstance(a, (ObstacleAgent, CustomerAgent)) for a in contents)
            if is_chair_target or not blocked:
                self.model.grid.move_agent(self, cell)
                return

    # =========================================================================
    # RAYCASTING — SOROT MATA
    # =========================================================================

    def _cast_gaze(self):
        """
        Pancarkan 3 ray sorot mata dari posisi pelanggan yang sedang duduk.

        Arah ray:
          • 0°  — lurus ke tengah ruangan
          • -45° — miring kiri dari arah tengah
          • +45° — miring kanan dari arah tengah

        Ini mensimulasikan pandangan lurus + peripheral vision pelanggan.
        Setiap ray diproses oleh _trace_ray().

        Side effect:
            self.gaze_floor_cells    diisi dengan sel lantai yang tersorot
            self.gaze_obstacle_cells diisi dengan sel tiang yang "ditatap"
        """
        self.gaze_floor_cells    = []
        self.gaze_obstacle_cells = []

        ax, ay = self.pos
        cx     = self.model.grid.width  // 2
        cy     = self.model.grid.height // 2

        # Vektor menuju pusat ruangan
        dx_main = cx - ax
        dy_main = cy - ay
        length  = math.hypot(dx_main, dy_main)
        if length < 1e-9:
            return  # Pelanggan tepat di tengah, skip

        # Tiga arah: tengah, kiri 45°, kanan 45°
        for angle_deg in [-45, 0, 45]:
            rad   = math.radians(angle_deg)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            # Rotasi vektor arah
            rdx = (dx_main * cos_a - dy_main * sin_a) / length
            rdy = (dx_main * sin_a + dy_main * cos_a) / length
            self._trace_ray(ax, ay, rdx, rdy)

    def _trace_ray(self, ox: int, oy: int, dx: float, dy: float):
        """
        Telusuri satu ray dari titik (ox, oy) ke arah vektor (dx, dy).

        Algoritma (Bresenham-inspired discrete stepping):
        ─────────────────────────────────────────────────
        Untuk setiap langkah i = 1 .. gaze_range:
          1. Hitung sel berikutnya: round(ox + dx*i, oy + dy*i)
          2. Skip sel duplikat (akibat pembulatan float)
          3. Cek isi sel:
             • Ada ObstacleAgent →
                 obstacle_heatmap[nx][ny] += 1  (tiang "dilihat" → kandidat iklan)
                 STOP: ray terblokir
             • Tidak ada obstacle →
                 floor_heatmap[nx][ny] += 1     (lantai/kursi tersorot)
                 Lanjut ke sel berikutnya
          4. Berhenti jika keluar batas grid.

        Side effect:
            Mengupdate model.floor_heatmap dan model.obstacle_heatmap secara langsung.
            Menambahkan sel ke self.gaze_floor_cells / self.gaze_obstacle_cells.
        """
        grid   = self.model.grid
        f_heat = self.model.floor_heatmap
        o_heat = self.model.obstacle_heatmap
        prev   = (-9999, -9999)

        for i in range(1, self.gaze_range + 1):
            nx = int(round(ox + dx * i))
            ny = int(round(oy + dy * i))

            # Keluar batas grid → hentikan ray
            if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                break

            cell = (nx, ny)
            if cell == prev:
                continue   # Duplikat akibat floating-point, skip
            prev = cell

            contents     = grid.get_cell_list_contents([cell])
            has_obstacle = any(isinstance(a, ObstacleAgent) for a in contents)

            if has_obstacle:
                # ── Ray berhenti di tiang/halangan ──────────────────────────
                # Tiang yang "dilihat" adalah kandidat media iklan (stiker/banner)
                o_heat[nx][ny] += 1
                self.gaze_obstacle_cells.append(cell)
                break   # Ray terblokir, tidak lanjut

            # ── Ray menembus sel ini → catat sebagai floor exposure ─────────
            f_heat[nx][ny] += 1
            self.gaze_floor_cells.append(cell)

    # =========================================================================
    # STEP UTAMA
    # =========================================================================

    def step(self):
        """Eksekusi satu langkah simulasi sesuai status saat ini."""

        if self.status == CustomerStatus.SEEKING:
            chair = self._find_nearest_empty_chair()
            if chair:
                # Klaim kursi dan mulai bergerak
                self.target_chair = chair
                chair.occupied    = True
                chair.occupant    = self
                self.status       = CustomerStatus.MOVING

        elif self.status == CustomerStatus.MOVING:
            if self.pos == self.target_chair.pos:
                self.status = CustomerStatus.SITTING   # Sudah sampai
            else:
                self._step_toward(self.target_chair.pos)

        elif self.status == CustomerStatus.SITTING:
            self._cast_gaze()          # Pancarkan sorot mata → update heatmap
            self.sitting_timer -= 1
            if self.sitting_timer <= 0:
                # Waktunya habis → lepas kursi dan bersiap pergi
                if self.target_chair:
                    self.target_chair.occupied = False
                    self.target_chair.occupant = None
                self.status             = CustomerStatus.LEAVING
                self.gaze_floor_cells   = []
                self.gaze_obstacle_cells= []

        elif self.status == CustomerStatus.LEAVING:
            # Hapus dari grid lalu dari model (Mesa 3.x API)
            self.model.grid.remove_agent(self)
            self.remove()