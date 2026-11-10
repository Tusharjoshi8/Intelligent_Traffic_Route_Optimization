import json
from pathlib import Path

from pso_algorithm import ParticleSwarmOptimization

# Run PSO optimization
pso = ParticleSwarmOptimization()
best_position, best_score = pso.optimize()

# Prepare optimized signal timings
directions = ["North", "South", "East", "West"]

optimized_timings = {
    direction: round(float(green_time), 2)
    for direction, green_time in zip(directions, best_position)
}

# Prepare results
results = {
    "algorithm": "Particle Swarm Optimization",
    "optimized_signal_timings": optimized_timings,
    "total_green_time": round(float(sum(best_position)), 2),
    "best_objective_score": round(float(best_score), 2)
}

# Create results folder
results_dir = Path(__file__).resolve().parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# Save JSON file
output_file = results_dir / "optimized_signal_timings.json"

with open(output_file, "w") as file:
    json.dump(results, file, indent=4)

print("\nPSO RESULTS SAVED SUCCESSFULLY!")
print("====================================")
print(json.dumps(results, indent=4))
print(f"\nFile location: {output_file}")