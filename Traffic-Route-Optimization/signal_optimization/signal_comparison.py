from pso_algorithm import ParticleSwarmOptimization
from signal_objective import (
    calculate_total_delay,
    display_delay
)

import numpy as np


def compare_signal_timings():

    print("\nSIGNAL TIMING COMPARISON")
    print("====================================")

    # Original signal timings
    original_timings = np.array([26, 22, 30, 26])

    # Calculate original objective score
    original_score = calculate_total_delay(
        original_timings
    )

    print("\n1. ORIGINAL SIGNAL TIMINGS")

    display_delay(original_timings)

    # Run PSO optimization
    pso = ParticleSwarmOptimization()

    optimized_timings, optimized_score = pso.optimize()

    print("\n2. OPTIMIZED SIGNAL TIMINGS")

    display_delay(optimized_timings)

    # Calculate improvement
    improvement = (
        (original_score - optimized_score)
        / original_score
    ) * 100

    print("\n3. PERFORMANCE COMPARISON")
    print("------------------------------------")

    print(
        "Original objective score:",
        round(original_score, 2)
    )

    print(
        "Optimized objective score:",
        round(optimized_score, 2)
    )

    print(
        "Objective score reduction:",
        round(improvement, 2),
        "%"
    )

    print("\n4. GREEN TIME COMPARISON")
    print("------------------------------------")

    directions = ["North", "South", "East", "West"]

    for i, direction in enumerate(directions):

        print(
            direction,
            "| Original:",
            round(original_timings[i], 2),
            "sec | Optimized:",
            round(optimized_timings[i], 2),
            "sec"
        )

    print("\nSIGNAL OPTIMIZATION COMPARISON COMPLETED!")


if __name__ == "__main__":

    compare_signal_timings()