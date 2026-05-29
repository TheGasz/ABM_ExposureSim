import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ============================================
# HIGH-CONTRAST TIME-ON-CAMERA HEATMAP
# ============================================

width, height = 2, 1

nx, ny = 1200, 600

x = np.linspace(0, width, nx)
y = np.linspace(0, height, ny)
X, Y = np.meshgrid(x, y)

# ============================================
# SOURCE POINTS
# ============================================

red_points = [(0.5, 0), (1.5, 0)]
green_points = [(0, 0.5)]
blue_points = [(0.5, 1), (1.5, 1)]
black_points = [(2, 0.5)]

# ============================================
# INFLUENCE FUNCTION
# ============================================

def influence(points, sigma=0.23):
    field = np.zeros_like(X)

    for px, py in points:
        d2 = (X - px)**2 + (Y - py)**2
        field += np.exp(-d2 / (2 * sigma**2))

    return field

R = influence(red_points)
G = influence(green_points)
B = influence(blue_points)
K = influence(black_points)

# ============================================
# NORMALIZATION
# ============================================

total = R + G + B + K + 1e-9

R /= total
G /= total
B /= total
K /= total

# ============================================
# VIBRANT RGB COMPOSITION
# ============================================

# Stronger saturation
R = R**0.7
G = G**0.7
B = B**0.7

# Re-normalize after boosting
rgb_total = R + G + B + 1e-9
R /= rgb_total
G /= rgb_total
B /= rgb_total

# Build image
RGB = np.dstack([R, G, B])

# Dark influence from black source
brightness = 1 - (K * 0.95)
RGB *= brightness[..., np.newaxis]

# Slight smoothing
RGB = gaussian_filter(RGB, sigma=1.2)

# Increase contrast
RGB = np.clip((RGB - 0.1) * 1.4, 0, 1)

# ============================================
# PLOT
# ============================================

fig, ax = plt.subplots(figsize=(13, 6))

ax.imshow(
    RGB,
    origin='lower',
    extent=[0, width, 0, height],
    interpolation='bicubic'
)

# ============================================
# SOURCE MARKERS
# ============================================

marker_size = 220

ax.scatter(
    [0.5, 1.5], [0, 0],
    c='#ff2222',
    s=marker_size,
    edgecolors='white',
    linewidths=2.5,
    zorder=10
)

ax.scatter(
    [0], [0.5],
    c='#00ff66',
    s=marker_size,
    edgecolors='white',
    linewidths=2.5,
    zorder=10
)

ax.scatter(
    [0.5, 1.5], [1, 1],
    c='#3399ff',
    s=marker_size,
    edgecolors='white',
    linewidths=2.5,
    zorder=10
)

ax.scatter(
    [2], [0.5],
    c='black',
    s=marker_size,
    edgecolors='white',
    linewidths=2.5,
    zorder=10
)

# ============================================
# BORDER
# ============================================

ax.plot(
    [0, 2, 2, 0, 0],
    [0, 0, 1, 1, 0],
    color='white',
    linewidth=2.5,
    alpha=0.9
)

# ============================================
# STYLE
# ============================================

ax.set_xlim(0, 2)
ax.set_ylim(0, 1)

ax.set_xticks([])
ax.set_yticks([])

ax.set_facecolor("#0a0a0a")
fig.patch.set_facecolor("#0a0a0a")

ax.set_title(
    "High-Contrast Camera Exposure Heatmap",
    fontsize=24,
    color='white',
    weight='bold',
    pad=20
)

plt.tight_layout()
plt.show()