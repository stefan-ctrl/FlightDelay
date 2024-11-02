from matplotlib import pyplot as plt
import seaborn as sns

def plot_weather_obscuration_flight_delay(data, airport, weather_delay_only=False, save_base_name='', title_suffix=''):
    # Create a new column to classify whether the flight was delayed (1) or not (0)
    data['Delayed'] = data['DepDelayMin'] > 0  # True if delay > 0, False otherwise

    if weather_delay_only:
        data = data[data['WeatherDelay'] > 0]
    print(f'Weather delay only: {weather_delay_only}; Data shape: {data.shape}')
    # Group by 'Weather_Obscuration' and count the delayed and not delayed flights
    delay_counts = data.groupby('Weather_Obscuration')['Delayed'].value_counts().unstack(fill_value=0)
    delay_counts = delay_counts.reindex(columns=[False, True], fill_value=0)
    delay_counts.columns = ['Not_Delayed', 'Delayed']

    # Reset index for easier plotting
    delay_counts = delay_counts.reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    # Create a stacked bar plot for Delayed and Not Delayed counts
    delay_counts.set_index('Weather_Obscuration').plot(kind='bar', stacked=True, color=['#1f77b4', '#ff7f0e'])



    title = '[' + airport + '] Flight Delays by Weather Obscuration'
    if  weather_delay_only:
        title = '[' + airport + '] Weather Delayed Flights by Weather Obscuration'
    plt.title(title, fontsize=16)
    plt.xlabel('Weather Obscuration', fontsize=12)
    plt.ylabel('Number of Flights', fontsize=12)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

    # Show the plot
    plt.tight_layout()
    #plt.show()
    if save_base_name:
        plt.savefig(f'{airport}_{save_base_name}_{title_suffix}.png', dpi=1000)


def plot_weather_precipitation_flight_delay(data, airport, weather_delay_only=False, save_base_name='', title_suffix=''):
    # Create a new column to classify whether the flight was delayed (1) or not (0)
    data['Delayed'] = data['DepDelayMin'] > 0  # True if delay > 0, False otherwise

    if weather_delay_only:
        data = data[data['WeatherDelay'] > 0]
    # Group by 'Weather_Obscuration' and count the delayed and not delayed flights
    delay_counts = data.groupby('Weather_Precipitation')['Delayed'].value_counts().unstack(fill_value=0)
    delay_counts = delay_counts.reindex(columns=[False, True], fill_value=0)
    delay_counts.columns = ['Not_Delayed', 'Delayed']

    # Reset index for easier plotting
    delay_counts = delay_counts.reset_index()
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    # Create a stacked bar plot for Delayed and Not Delayed counts
    delay_counts.set_index('Weather_Precipitation').plot(kind='bar', stacked=True, color=['#1f77b4', '#ff7f0e'])

    print(f'delay_counts: {delay_counts}')

    # Adding labels and title
    title = '[' + airport + '] Flight Delays by Weather Precipitation'
    if  weather_delay_only:
        title = '[' + airport + '] Weather Delayed Flights by Weather Precipitation'
    plt.title(title, fontsize=16)
    plt.xlabel('Weather Precipitation', fontsize=12)
    plt.ylabel('Number of Flights', fontsize=12)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

    # Show the plot
    plt.tight_layout()
    #plt.show()
    if save_base_name:
        plt.savefig(f'{airport}_{save_base_name}_{title_suffix}.png', dpi=1000)

