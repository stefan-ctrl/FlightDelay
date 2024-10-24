from matplotlib import pyplot as plt
import seaborn as sns

def plot_weather_obscuration_flight_delay(data):
    # Create a new column to classify whether the flight was delayed (1) or not (0)
    data['Delayed'] = data['DepDelayMin'] > 0  # True if delay > 0, False otherwise

    # Group by 'Weather_Obscuration' and count the delayed and not delayed flights
    delay_counts = data.groupby('Weather_Obscuration')['Delayed'].value_counts().unstack(fill_value=0)

    # Rename the columns for clarity
    delay_counts.columns = ['Not_Delayed', 'Delayed']

    # Reset index for easier plotting
    delay_counts = delay_counts.reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    # Create a stacked bar plot for Delayed and Not Delayed counts
    delay_counts.set_index('Weather_Obscuration').plot(kind='bar', stacked=True, color=['#1f77b4', '#ff7f0e'],
                                                       figsize=(12, 6))

    # Adding labels and title
    plt.title('Flight Delays by Weather Obscuration', fontsize=16)
    plt.xlabel('Weather Obscuration', fontsize=12)
    plt.ylabel('Number of Flights', fontsize=12)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_weather_precipitation_flight_delay(data):
    # Create a new column to classify whether the flight was delayed (1) or not (0)
    data['Delayed'] = data['DepDelayMin'] > 0  # True if delay > 0, False otherwise

    # Group by 'Weather_Obscuration' and count the delayed and not delayed flights
    delay_counts = data.groupby('Weather_Precipitation')['Delayed'].value_counts().unstack(fill_value=0)

    # Rename the columns for clarity
    delay_counts.columns = ['Not_Delayed', 'Delayed']

    # Reset index for easier plotting
    delay_counts = delay_counts.reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    # Create a stacked bar plot for Delayed and Not Delayed counts
    delay_counts.set_index('Weather_Precipitation').plot(kind='bar', stacked=True, color=['#1f77b4', '#ff7f0e'],
                                                       figsize=(12, 6))

    # Adding labels and title
    plt.title('Flight Delays by Weather Precipitation', fontsize=16)
    plt.xlabel('Weather Precipitation', fontsize=12)
    plt.ylabel('Number of Flights', fontsize=12)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

    # Show the plot
    plt.tight_layout()
    plt.show()

