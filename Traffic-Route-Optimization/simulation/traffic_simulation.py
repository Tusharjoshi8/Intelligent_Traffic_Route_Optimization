"""Traffic flow simulation."""
import json
import numpy as np
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_FILE = (
    PROJECT_ROOT
    / "signal_optimization"
    / "results"
    / "optimized_signal_timings.json"
)


# Load optimized PSO timings
with open(RESULTS_FILE, "r") as file:
    pso_results = json.load(file)

green_timings = pso_results["optimized_signal_timings"]


# Synthetic traffic volume (vehicles per hour)
traffic_volume = {
    "North": 900,
    "South": 700,
    "East": 1200,
    "West": 1000
}

lanes = {
    "North": 2,
    "South": 2,
    "East": 2,
    "West": 2
}

cycle_time = 120
clearance_time = 4
saturation_flow = 1800

simulation_time = 3600  # 1 hour

directions = ["North", "South", "East", "West"]

# Initialize queues and metrics
queues = {direction: 0 for direction in directions}
service_accumulator = {direction: 0.0 for direction in directions}

total_arrived = {direction: 0 for direction in directions}
total_served = {direction: 0 for direction in directions}
queue_history = {direction: [] for direction in directions}

np.random.seed(42)


# Create signal phase schedule
phase_schedule = []
current_time = 0

for direction in directions:
    green_time = green_timings[direction]

    phase_schedule.append({
        "direction": direction,
        "start": current_time,
        "end": current_time + green_time
    })

    current_time += green_time + clearance_time


# Run simulation
for second in range(simulation_time):

    cycle_position = second % cycle_time

    active_direction = None

    for phase in phase_schedule:
        if phase["start"] <= cycle_position < phase["end"]:
            active_direction = phase["direction"]
            break

    for direction in directions:

        # Generate synthetic vehicle arrivals
        arrival_rate = traffic_volume[direction] / 3600

        arrivals = np.random.poisson(arrival_rate)

        queues[direction] += arrivals
        total_arrived[direction] += arrivals

        # Serve vehicles only during green signal
        if direction == active_direction:

            service_rate = (
                saturation_flow * lanes[direction] / 3600
            )

            service_accumulator[direction] += service_rate

            vehicles_to_serve = int(
                service_accumulator[direction]
            )

            served = min(
                queues[direction],
                vehicles_to_serve
            )

            queues[direction] -= served
            total_served[direction] += served

            service_accumulator[direction] -= served

        queue_history[direction].append(
            queues[direction]
        )


# Display simulation results
print("\nPYTHON TRAFFIC SIMULATION RESULTS")
print("=" * 45)

print("\nOptimized Signal Timings:")
for direction in directions:
    print(
        f"{direction}: {green_timings[direction]} seconds"
    )

print("\nTraffic Performance:")

for direction in directions:

    average_queue = np.mean(queue_history[direction])

    print(f"\n{direction}:")
    print(f"Vehicles arrived: {total_arrived[direction]}")
    print(f"Vehicles served: {total_served[direction]}")
    print(f"Vehicles remaining: {queues[direction]}")
    print(f"Average queue length: {average_queue:.2f}")


print("\nOVERALL RESULTS")
print("=" * 45)

print(
    "Total vehicles arrived:",
    sum(total_arrived.values())
)

print(
    "Total vehicles served:",
    sum(total_served.values())
)

print(
    "Total vehicles remaining:",
    sum(queues.values())
)

print("\nSIMULATION COMPLETED!")