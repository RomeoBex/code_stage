import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
 
# Paramètres
num_particles_large = 200
box_size = 100
steps = 200
step_size_large = 0.3

# Sous-volume (champ de la caméra)
subvol_origin = np.array([40, 40, 40])
subvol_size = 20

# Positions initiales
np.random.seed(42)
positions_large = np.random.rand(num_particles_large, 3) * box_size

# Vitesses initiales
vel_large = (np.random.rand(num_particles_large, 3) - 0.5) * 2 * step_size_large

# Figure
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)
ax.set_zlim(0, box_size)

# Particules (uniquement les grandes bleues)
scat_large = ax.scatter(*positions_large.T, s=10, c='blue', label="Particules")

# Sous-volume (boîte transparente rouge)
def draw_subvolume():
    x0, y0, z0 = subvol_origin
    x1, y1, z1 = subvol_origin + subvol_size

    verts = [
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1]
    ]

    faces = [
        [verts[0], verts[1], verts[2], verts[3]],  # bas
        [verts[4], verts[5], verts[6], verts[7]],  # haut
        [verts[0], verts[1], verts[5], verts[4]],  # avant
        [verts[2], verts[3], verts[7], verts[6]],  # arrière
        [verts[1], verts[2], verts[6], verts[5]],  # droite
        [verts[3], verts[0], verts[4], verts[7]]   # gauche
    ]

    ax.add_collection3d(Poly3DCollection(faces, alpha=0.2, facecolor='red', edgecolor='darkred'))

draw_subvolume()

# Animation
def update(frame):
    global positions_large, vel_large

    positions_large += vel_large

    for i in range(3):
        mask_large = (positions_large[:, i] <= 0) | (positions_large[:, i] >= box_size)
        vel_large[mask_large, i] *= -1
        positions_large = np.clip(positions_large, 0, box_size)

    scat_large._offsets3d = tuple(positions_large.T)
    return scat_large,

ani = animation.FuncAnimation(fig, update, frames=steps, interval=30, blit=False)

# Sauvegarde de la vidéo
ani.save("mouvement_brownien_3d_bleu.mp4", writer='ffmpeg', fps=30)
# ani.save("mouvement_brownien_3d_bleu.gif", writer=animation.PillowWriter(fps=30))

plt.close()
print("✅ Vidéo générée : mouvement_brownien_3d_bleu.mp4")

