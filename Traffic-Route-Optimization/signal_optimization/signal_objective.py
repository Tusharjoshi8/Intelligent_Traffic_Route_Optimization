"""Objective functions for signal optimization."""
import numpy as np

from signal_data import INTERSECTION_DATA, SIGNAL_CONFIG


def calculate_direction_delay(
    traffic_volume,
    lanes,
    green_time
):
    """
    Estimate average vehicle delay for one direction.

    This is a simplified signal-delay model.
    """

    cycle_time = SIGNAL_CONFIG["cycle_time"]
    saturation_flow = SIGNAL_CONFIG["saturation_flow"]

    # Calculate traffic capacity in vehicles per hour
    capacity = (
        saturation_flow
        * lanes
        * green_time
        / cycle_time
    )

    # Calculate degree of saturation
    if capacity <= 0:
        return float("inf")

    saturation_ratio = traffic_volume / capacity

    # Prevent division by zero or a negative denominator
    adjusted_ratio = min(saturation_ratio, 0.99)

    denominator = (
        1 - adjusted_ratio * green_time / cycle_time
    )

    # Simplified uniform signal delay formula
    delay = (
        0.5
        * cycle_time
        * (1 - green_time / cycle_time) ** 2
        / denominator
    )

    # Additional penalty if traffic demand exceeds capacity
    if saturation_ratio > 1:
        overflow_penalty = 3600 * (saturation_ratio - 1) ** 2
        delay += overflow_penalty

    return delay


def calculate_total_delay(green_times):

    directions = list(INTERSECTION_DATA.keys())

    # Check number of green-time values
    if len(green_times) != len(directions):
        return float("inf")

    cycle_time = SIGNAL_CONFIG["cycle_time"]
    clearance_time = SIGNAL_CONFIG["clearance_time"]

    minimum_green = SIGNAL_CONFIG["minimum_green_time"]
    maximum_green = SIGNAL_CONFIG["maximum_green_time"]

    available_green = cycle_time - clearance_time

    green_times = np.asarray(green_times, dtype=float)

    # Check signal timing constraints
    if np.any(green_times < minimum_green):
        return float("inf")

    if np.any(green_times > maximum_green):
        return float("inf")

    if not np.isclose(np.sum(green_times), available_green):
        return float("inf")

    total_weighted_delay = 0
    total_traffic = 0

    # Calculate delay for every direction
    for index, direction in enumerate(directions):

        traffic = INTERSECTION_DATA[direction]["traffic_volume"]
        lanes = INTERSECTION_DATA[direction]["lanes"]

        green_time = green_times[index]

        delay = calculate_direction_delay(
            traffic,
            lanes,
            green_time
        )

        total_weighted_delay += traffic * delay
        total_traffic += traffic

    # Calculate traffic-weighted average delay
    average_delay = total_weighted_delay / total_traffic

    return average_delay


def display_delay(green_times):

    directions = list(INTERSECTION_DATA.keys())

    print("\nTRAFFIC SIGNAL DELAY ANALYSIS")
    print("====================================")

    for index, direction in enumerate(directions):

        traffic = INTERSECTION_DATA[direction]["traffic_volume"]
        lanes = INTERSECTION_DATA[direction]["lanes"]

        delay = calculate_direction_delay(
            traffic,
            lanes,
            green_times[index]
        )

        print(
            direction,
            "| Green Time:",
            green_times[index],
            "seconds | Estimated Delay:",
            round(delay, 2),
            "seconds"
        )

    total_delay = calculate_total_delay(green_times)

    print("\nWeighted Average Delay:", round(total_delay, 2), "seconds")


if __name__ == "__main__":

    # Example valid signal timing
    sample_green_times = [26, 22, 30, 26]

    print("SAMPLE GREEN SIGNAL TIMINGS")
    print("------------------------------------")

    print("North:", sample_green_times[0], "seconds")
    print("South:", sample_green_times[1], "seconds")
    print("East:", sample_green_times[2], "seconds")
    print("West:", sample_green_times[3], "seconds")

    display_delay(sample_green_times)