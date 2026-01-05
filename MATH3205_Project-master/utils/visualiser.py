"""
Animated visualization of traffic intersections:
- Each intersection is a square
- For each direction (N,E,S,W), draw *two* black arrows side-by-side (in/out)
- Inflow/outflow numbers displayed clearly near the arrows
- Green/red circles show light status for incoming arrows
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation


def visualise_traffic(flows_in, flows_out, greens, positions,
                      directions=('N', 'E', 'S', 'W'),
                      intersection_size=0.06, pause_per_frame=300):
    R, D, T = flows_in.shape
    assert D == 4, "Expected 4 directions (N,E,S,W)"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    title = ax.set_title("Traffic Flow Over Time")

    dir_vecs = {'N': (0, 1), 'E': (1, 0), 'S': (0, -1), 'W': (-1, 0)}

    squares = []
    arrows_in, arrows_out = [], []
    texts_in, texts_out = [], []
    lights = []

    for i, (x, y) in enumerate(positions):
        # Draw the intersection square
        sq = patches.Rectangle(
            (x - intersection_size / 2, y - intersection_size / 2),
            intersection_size, intersection_size,
            ec='black', fc='lightgray', lw=1.2, zorder=1)
        ax.add_patch(sq)
        squares.append(sq)

        ai, ao, ti, to, li = [], [], [], [], []

        for d, dir_label in enumerate(directions):
            dx, dy = dir_vecs[dir_label]
            sep = intersection_size * 0.3  # side-by-side spacing
            offset = intersection_size * 0.9
            arrow_len = intersection_size * 0.4

            # Perpendicular offset (rotate direction 90 degrees)
            perp_dx, perp_dy = -dy, dx

            # --------------------
            # Incoming arrow
            # --------------------
            in_start = (x + dx * offset + perp_dx * sep,
                        y + dy * offset + perp_dy * sep)
            arr_in = patches.FancyArrow(
                in_start[0], in_start[1],
                -dx * arrow_len, -dy * arrow_len,
                width=0.008, head_width=0.03,
                length_includes_head=True, color='black', zorder=2)
            ax.add_patch(arr_in)
            ai.append(arr_in)

            # Inflow label
            text_in = ax.text(in_start[0] + dx * arrow_len * 1.2,
                              in_start[1] + dy * arrow_len * 1.2,
                              '', ha='center', va='center', fontsize=15,
                              color='blue', weight='bold', zorder=3)
            ti.append(text_in)

            # Light indicator beside incoming arrow
            light_pos = (in_start[0] + dx * 0.06, in_start[1] + dy * 0.06)
            circ = patches.Circle(light_pos, 0.015, ec='black', fc='red', lw=0.8, zorder=3)
            ax.add_patch(circ)
            li.append(circ)

            # --------------------
            # Outgoing arrow
            # --------------------
            out_start = (x + dx * offset - perp_dx * sep,
                         y + dy * offset - perp_dy * sep)
            arr_out = patches.FancyArrow(
                out_start[0], out_start[1],
                dx * arrow_len, dy * arrow_len,
                width=0.008, head_width=0.03,
                length_includes_head=True, color='black', zorder=2)
            ax.add_patch(arr_out)
            ao.append(arr_out)

            # Outflow label
            text_out = ax.text(out_start[0] + dx * arrow_len * 1.2,
                               out_start[1] + dy * arrow_len * 1.2,
                               '', ha='center', va='center', fontsize=15,
                               color='darkorange', weight='bold', zorder=3)
            to.append(text_out)

        arrows_in.append(ai)
        arrows_out.append(ao)
        texts_in.append(ti)
        texts_out.append(to)
        lights.append(li)

    # --------------------
    # Update function
    # --------------------
    def update(frame):
        title.set_text(f"Timestep {frame + 1} / {T}")
        for i in range(R):
            for d in range(4):
                inflow = flows_in[i, d, frame]
                outflow = flows_out[i, d, frame]
                texts_in[i][d].set_text(f"{int(inflow)}")
                texts_out[i][d].set_text(f"{int(outflow)}")
                lights[i][d].set_facecolor('green' if greens[i, d, frame] else 'red')
        return []

    anim = FuncAnimation(fig, update, frames=T, interval=pause_per_frame, repeat=False)
    anim.save('traffic_animation.gif', writer='pillow', fps=3)
    #anim.save('traffic_animation.mp4', writer='ffmpeg', fps=3)
    plt.show()
    return anim


# ----------------------------------------------------------------------
# Example usage
if __name__ == "__main__":
    R, D, T = 4, 4, 30
    np.random.seed(1)
    flows_in = np.random.randint(0, 30, (R, D, T))
    flows_out = np.random.randint(0, 30, (R, D, T))
    greens = np.zeros((R, D, T), dtype=bool)
    for i in range(R):
        for t in range(T):
            greens[i, (t // 6 + i) % 4, t] = True  # simple cyclic green

    positions = [(0.3, 0.7), (0.7, 0.7), (0.3, 0.3), (0.7, 0.3)]
    visualise_traffic(flows_in, flows_out, greens, positions)
