import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

np.random.seed(10)

Path("figures").mkdir(exist_ok=True)
xmin, xmax = -10, 10
ymin, ymax = -10, 10
n_paths = 50
n_sensors = 4
rho = 0.95
sigma = 4.0

x = np.linspace(xmin, xmax, 200)

def make_paths(n):
    paths = []
    for _ in range(n):
        c0 = np.random.normal(0, 3)
        c1 = np.random.normal(0, 0.25)
        c2 = np.random.normal(0, 0.03)
        y = c0 + c1 * x + c2 * x**2
        y = np.clip(y, ymin, ymax)
        paths.append((x, y, c0, c1, c2))
    return paths

def distance_to_path(sensor, path):
    sx, sy = sensor
    px, py = path[0], path[1]
    distances = np.sqrt((px - sx) ** 2 + (py - sy) ** 2)
    return np.min(distances)

def sensor_detection(sensor, path):
    d = distance_to_path(sensor, path)
    return rho * np.exp(-(d**2) / sigma)

def path_missed_probability(sensors, path):
    miss = 1.0
    for sensor in sensors:
        p_detect = sensor_detection(sensor, path)
        miss *= 1 - p_detect
    return miss

def average_missed_probability(sensors, paths):
    missed_values = []
    for path in paths:
        missed_values.append(path_missed_probability(sensors, path))
    return np.mean(missed_values)

def greedy_sensors(paths):
    grid_x = np.linspace(xmin, xmax, 11)
    grid_y = np.linspace(ymin, ymax, 11)
    candidates = [(i, j) for i in grid_x for j in grid_y]
    selected = []
    for _ in range(n_sensors):
        best_sensor = None
        best_score = 1.0
        for candidate in candidates:
            if candidate in selected:
                continue
            test_sensors = selected + [candidate]
            score = average_missed_probability(test_sensors, paths)
            if score < best_score:
                best_score = score
                best_sensor = candidate
        selected.append(best_sensor)
    return selected

paths = make_paths(n_paths)

uniform_sensors = [(-6, 0), (-2, 0), (2, 0), (6, 0)]
greedy_selected_sensors = greedy_sensors(paths)

uniform_missed = average_missed_probability(uniform_sensors, paths)
greedy_missed = average_missed_probability(greedy_selected_sensors, paths)

uniform_detection = 1 - uniform_missed
greedy_detection = 1 - greedy_missed

results = pd.DataFrame({
    "Method": ["Uniform baseline", "Greedy curved-path"],
    "Average detection probability": [uniform_detection, greedy_detection],
    "Average missed probability": [uniform_missed, greedy_missed]
})

print("\nUniform sensors:")
print(uniform_sensors)
print("\nGreedy sensors:")
print(greedy_selected_sensors)
print("\nResults:")
print(results)

results.to_csv("simulation_results.csv", index=False)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 11

plt.figure(figsize=(8, 6))

for path in paths:
    plt.plot(path[0], path[1], color="gray", linewidth=1, alpha=0.45)

ux, uy = zip(*uniform_sensors)
gx, gy = zip(*greedy_selected_sensors)

plt.scatter(ux, uy, marker="s", s=90, color="black", label="Uniform sensors")
plt.scatter(gx, gy, marker="o", s=90, color="red", label="Greedy curved-path sensors")

plt.xlabel("x position")
plt.ylabel("y position")
plt.title("Curved Ship Paths and Sensor Locations")
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.grid(True, alpha=0.3)
plt.legend(frameon=True)
plt.savefig("figures/figure1_curved_paths_sensors.png", dpi=300, bbox_inches="tight")
plt.show()
slopes = [path[3] for path in paths]
curvatures = [path[4] for path in paths]
plt.figure(figsize=(7, 5))
plt.scatter(slopes, curvatures, color="black", s=35, alpha=0.75)
plt.axhline(0, color="gray", linewidth=1)
plt.axvline(0, color="gray", linewidth=1)

plt.xlabel("slope parameter $c_1$")
plt.ylabel("curvature parameter $c_2$")
plt.title("Curved Trajectory Representation Space")
plt.grid(True, alpha=0.3)
plt.savefig("figures/figure2_curve_representation_space.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(6, 5))

methods = ["Uniform\nbaseline", "Greedy\ncurved-path"]
detection_values = [uniform_detection, greedy_detection]

bars = plt.bar(methods, detection_values, color=["lightgray", "white"], edgecolor="black", linewidth=1.2)

plt.ylabel("Average detection probability")
plt.title("Detection Probability Comparison")
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)

for bar in bars:
    value = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
        fontsize=11
    )
plt.savefig("figures/figure3_detection_probability_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
plt.figure(figsize=(6, 5))
missed_values = [uniform_missed, greedy_missed]
bars = plt.bar(methods, missed_values, color=["lightgray", "white"], edgecolor="black", linewidth=1.2)
plt.ylabel("Average missed detection probability")
plt.title("Missed Detection Probability Comparison")
plt.ylim(0, max(missed_values) + 0.15)
plt.grid(axis="y", alpha=0.3)

for bar in bars:
    value = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
        fontsize=11
    )
plt.savefig("figures/figure4_missed_probability_comparison.png", dpi=300, bbox_inches="tight")
plt.show()