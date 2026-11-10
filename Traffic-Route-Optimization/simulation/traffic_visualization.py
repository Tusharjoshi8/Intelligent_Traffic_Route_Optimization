import matplotlib.pyplot as plt
import numpy as np

# Simulation results
directions = ["North", "South", "East", "West"]

vehicles_arrived = [888, 692, 1219, 978]
vehicles_served = [750, 570, 990, 810]
vehicles_remaining = [138, 122, 229, 168]

average_queue = [60.51, 42.42, 109.95, 102.60]


# 1. Vehicles Arrived vs Vehicles Served
x = np.arange(len(directions))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    vehicles_arrived,
    width,
    label="Vehicles Arrived"
)

plt.bar(
    x + width / 2,
    vehicles_served,
    width,
    label="Vehicles Served"
)

plt.xlabel("Traffic Direction")
plt.ylabel("Number of Vehicles")
plt.title("Vehicles Arrived vs Vehicles Served")

plt.xticks(x, directions)
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("simulation/vehicles_comparison.png")
plt.show()


# 2. Remaining Queue Length
plt.figure(figsize=(10, 6))

plt.bar(directions, vehicles_remaining)

plt.xlabel("Traffic Direction")
plt.ylabel("Remaining Vehicles")
plt.title("Remaining Queue Length by Direction")

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("simulation/remaining_queue.png")
plt.show()


# 3. Average Queue Length
plt.figure(figsize=(10, 6))

plt.bar(directions, average_queue)

plt.xlabel("Traffic Direction")
plt.ylabel("Average Queue Length (Vehicles)")
plt.title("Average Queue Length by Direction")

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("simulation/average_queue.png")
plt.show()

print("\nTRAFFIC VISUALIZATION COMPLETED!")
print("====================================")
print("1. vehicles_comparison.png")
print("2. remaining_queue.png")
print("3. average_queue.png")