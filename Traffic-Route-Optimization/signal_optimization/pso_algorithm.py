import numpy as np

from signal_data import SIGNAL_CONFIG
from signal_objective import calculate_total_delay
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class ParticleSwarmOptimization:

    def __init__(
        self,
        num_particles=30,
        max_iterations=100,
        w=0.7,
        c1=1.5,
        c2=1.5
    ):

        self.num_particles = num_particles
        self.max_iterations = max_iterations

        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.num_directions = 4

        self.min_green = SIGNAL_CONFIG["minimum_green_time"]
        self.max_green = SIGNAL_CONFIG["maximum_green_time"]

        self.available_green = (
            SIGNAL_CONFIG["cycle_time"]
            - SIGNAL_CONFIG["clearance_time"]
        )

        self.particles = []
        self.velocities = []

        self.personal_best_positions = []
        self.personal_best_scores = []

        self.global_best_position = None
        self.global_best_score = float("inf")

        self.history = []

    # Keep green times within limits and make their sum exact
    def project_position(self, position):

        position = np.asarray(position, dtype=float)

        # Find a value that makes the clipped values sum
        # to the available green time.
        low = np.min(position - self.max_green) - 1
        high = np.max(position - self.min_green) + 1

        for _ in range(100):

            middle = (low + high) / 2

            projected = np.clip(
                position - middle,
                self.min_green,
                self.max_green
            )

            if np.sum(projected) > self.available_green:
                low = middle
            else:
                high = middle

        return np.clip(
            position - (low + high) / 2,
            self.min_green,
            self.max_green
        )

    # Initialize particles
    def initialize_particles(self):

        self.particles = []
        self.velocities = []

        for _ in range(self.num_particles):

            random_position = np.random.uniform(
                self.min_green,
                self.max_green,
                self.num_directions
            )

            position = self.project_position(random_position)

            velocity = np.random.uniform(
                -2,
                2,
                self.num_directions
            )

            self.particles.append(position)
            self.velocities.append(velocity)

        self.particles = np.array(self.particles)
        self.velocities = np.array(self.velocities)

        self.personal_best_positions = self.particles.copy()

        self.personal_best_scores = np.array([
            calculate_total_delay(particle)
            for particle in self.particles
        ])

        best_index = np.argmin(self.personal_best_scores)

        self.global_best_position = (
            self.personal_best_positions[best_index].copy()
        )

        self.global_best_score = self.personal_best_scores[best_index]

    # Run PSO
    def optimize(self):

        np.random.seed(42)

        print("\nPARTICLE SWARM OPTIMIZATION")
        print("====================================")

        self.initialize_particles()

        for iteration in range(self.max_iterations):

            for i in range(self.num_particles):

                r1 = np.random.random(self.num_directions)
                r2 = np.random.random(self.num_directions)

                # Update velocity
                cognitive = (
                    self.c1
                    * r1
                    * (
                        self.personal_best_positions[i]
                        - self.particles[i]
                    )
                )

                social = (
                    self.c2
                    * r2
                    * (
                        self.global_best_position
                        - self.particles[i]
                    )
                )

                self.velocities[i] = (
                    self.w * self.velocities[i]
                    + cognitive
                    + social
                )

                # Update position and enforce constraints
                new_position = (
                    self.particles[i] + self.velocities[i]
                )

                self.particles[i] = self.project_position(
                    new_position
                )

                # Evaluate new solution
                score = calculate_total_delay(
                    self.particles[i]
                )

                # Update personal best
                if score < self.personal_best_scores[i]:

                    self.personal_best_scores[i] = score

                    self.personal_best_positions[i] = (
                        self.particles[i].copy()
                    )

                # Update global best
                if score < self.global_best_score:

                    self.global_best_score = score

                    self.global_best_position = (
                        self.particles[i].copy()
                    )

            self.history.append(self.global_best_score)

            if (iteration + 1) % 10 == 0:

                print(
                    "Iteration:",
                    iteration + 1,
                    "| Best objective score:",
                    round(self.global_best_score, 2)
                )

        return self.global_best_position, self.global_best_score


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    pso = ParticleSwarmOptimization()

    best_timings, best_score = pso.optimize()

    directions = ["North", "South", "East", "West"]

    print("\nOPTIMIZED TRAFFIC SIGNAL TIMINGS")
    print("====================================")

    for direction, green_time in zip(
        directions,
        best_timings
    ):

        print(
            direction,
            ":",
            round(green_time, 2),
            "seconds"
        )

    print("\nTotal green time:", round(np.sum(best_timings), 2))
    print("Best objective score:", round(best_score, 2))

    # Plot convergence graph
    plt.figure(figsize=(10, 5))

    plt.plot(
        range(1, len(pso.history) + 1),
        pso.history
    )

    plt.xlabel("Iterations")
    plt.ylabel("Best Objective Score")
    plt.title("PSO Convergence Graph")

    plt.grid(True)
    plt.tight_layout()

    output_path = (
        PROJECT_ROOT
        / "signal_optimization"
        / "pso_convergence.png"
    )

    plt.savefig(output_path, dpi=300)

    print("\nConvergence graph saved at:")
    print(output_path)

    plt.show()

    print("\nPSO OPTIMIZATION COMPLETED!")