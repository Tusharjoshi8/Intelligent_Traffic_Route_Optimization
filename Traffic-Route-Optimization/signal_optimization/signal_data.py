"""Traffic signal input data structures."""
# ==========================================
# TRAFFIC SIGNAL OPTIMIZATION - SIGNAL DATA
# ==========================================

# Four-way intersection traffic information

INTERSECTION_DATA = {

    "North": {
        "traffic_volume": 900,
        "lanes": 2
    },

    "South": {
        "traffic_volume": 700,
        "lanes": 2
    },

    "East": {
        "traffic_volume": 1200,
        "lanes": 2
    },

    "West": {
        "traffic_volume": 1000,
        "lanes": 2
    }

}


# Signal configuration

SIGNAL_CONFIG = {

    "cycle_time": 120,

    "clearance_time": 16,

    "minimum_green_time": 10,

    "maximum_green_time": 40,

    "saturation_flow": 1800

}


def display_intersection_data():

    print("\nFOUR-WAY TRAFFIC INTERSECTION")
    print("====================================")

    for direction, details in INTERSECTION_DATA.items():

        print("\nDirection:", direction)

        print(
            "Traffic Volume:",
            details["traffic_volume"],
            "vehicles/hour"
        )

        print("Number of Lanes:", details["lanes"])

    print("\nSIGNAL CONFIGURATION")
    print("------------------------------------")

    for parameter, value in SIGNAL_CONFIG.items():

        print(parameter, ":", value)


if __name__ == "__main__":

    display_intersection_data()