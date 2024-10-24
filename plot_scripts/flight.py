import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Assuming flight_data is your DataFrame after preprocessing
# Function to visualize delayed flights
def visualize_delays(flight_data):
    # Filter out flights with delays (only include those with non-zero delay)
    delayed_flights = flight_data[(flight_data['DepDelayMin'] > 0) | (flight_data['ArrDelayMin'] > 0)]

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot Departure Delays
    plt.scatter(delayed_flights['FlightDepDateTime'], delayed_flights['DepDelayMin'],
                color='red', label='Departure Delay (min)', alpha=0.6)

    # Formatting the date on the x-axis
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gcf().autofmt_xdate()  # Rotation for better readability

    # Adding labels and title
    plt.title('Delayed Flights: Departure and Arrival Delays Over Time')
    plt.xlabel('Flight Departure Date and Time')
    plt.ylabel('Delay Time (min)')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    count_delays_and_on_time(flight_data)
    plt.show()

def count_delays_and_on_time(flight_data):
    # Count delayed flights (DepDelayMin > 0)
    delayed_count = flight_data[flight_data['DepDelayMin'] > 0].shape[0]

    # Count on-time or faster flights (DepDelayMin <= 0)
    on_time_count = flight_data[flight_data['DepDelayMin'] <= 0].shape[0]

    print(f"Number of delayed flights: {delayed_count}")
    print(f"Number of on-time or faster flights: {on_time_count}")