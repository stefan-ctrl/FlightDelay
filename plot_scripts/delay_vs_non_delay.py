from matplotlib import pyplot as plt
import seaborn as sns

def plot_delay_vs_non_delay(df, airport, title_suffix='', save_base_name=''):
    # Calculate the required metrics
    total_flights = df.shape[0]
    total_delayed = df[df['DepDelayMin'] > 0].shape[0]
    delayed_over_15min = df[df['DepDel15'] == 1].shape[0]
    weather_delayed = df[df['WeatherDelay'] > 0].shape[0]
    weather_delayed_over_15min = df[df['WeatherDelay'] > 15].shape[0]

    # Calculate percentages
    values = [
        100,  # Total Flights as 100%
        (total_delayed / total_flights) * 100,
        (delayed_over_15min / total_flights) * 100,
        (weather_delayed / total_flights) * 100,
        (weather_delayed_over_15min / total_flights) * 100
    ]

    # Raw counts for labels
    counts = [total_flights, total_delayed, delayed_over_15min, weather_delayed, weather_delayed_over_15min]

    # Prepare the labels
    categories = [
        "Total Flights",
        "Total Delayed",
        "Delayed Over 15 Min",
        "Weather-Related \nDelay",
        "Weather-Related \nDelay Over 15 Min"
    ]

    # Plot the data
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, values, color=['blue', 'orange', 'green', 'red', 'purple'])
    plt.xlabel("Flight Categories")
    plt.ylabel("Percentage of Total Flights (%)")
    title = f"[{airport}] Total Flight Delay Analysis ({title_suffix})"
    plt.title(title)
    plt.xticks(rotation=0, ha='center', fontsize=9, fontweight='bold', color='black')
    plt.tight_layout()
    plt.ylim(0, 100)  # Ensure the y-axis is scaled to 100%

    # Add percentage and counts to each bar
    for i, (bar, value, count) in enumerate(zip(bars, values, counts)):
        if i == 0:  # For "Total Flights" (100%)
            plt.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,  # Inside the bar, vertically centered
                f'{value:.1f}% ({count})', ha='center', va='center', color='white', fontsize=9, fontweight='bold'
            )
        else:  # For other categories, place above the bar
            plt.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,  # Slightly above the bar
                f'{value:.1f}% ({count})', ha='center', va='bottom'
            )

    #plt.show()
    if save_base_name:
        plt.savefig(f'{airport}_{save_base_name}_{title_suffix}.png', dpi=1000)