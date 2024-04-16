import seaborn
import pandas as pd
import click
import matplotlib.pyplot as plt
import os

from ..tools import simulator


@click.command()
@click.option('--path_to_pickle', help='Benchmark directory')
@click.option('--workers')
@click.option('--nodes')
def visualize(path_to_pickle, workers, nodes):
    plot_types = ['n_active', 'duration', 'violin_dur_per_steps', 'hist_duration_steps']
    path, filename = os.path.split(path_to_pickle)
    fig = plt.figure(figsize=(15, 15))
    simulation = simulator.Simulation.load(path_to_pickle)

    # Get DataFrame
    df = pd.DataFrame(simulation.steps_info)
    df_parts = pd.DataFrame.from_records(df['parts'])
    # Remove follow up route selection
    # print(df_parts[['selected_routes', 'selected_routes_followup']])
    # df_parts = df_parts.loc[:, df_parts.columns != 'selected_routes_followup']
    df = pd.concat([df, df_parts], axis=1)

    for i, plot_type in enumerate(plot_types):
        ax = fig.add_subplot(2, 2, i + 1)
        if plot_type == 'n_active':
            seaborn.lineplot(df[['n_active', 'need_new_route']], ax=ax).set(title=f'{plot_type}')
        elif plot_type == 'duration':
            seaborn.violinplot(x=df['duration'], ax=ax).set(title=f'{plot_type}')
        elif plot_type == 'violin_dur_per_steps':
            seaborn.violinplot(ax=ax, data=df[['alternatives', 'advance_vehicle', 'selected_routes']]).set(title=f'{plot_type}')
        elif plot_type == 'hist_duration_steps':
            seaborn.histplot(ax=ax, data=df[['alternatives', 'advance_vehicle', 'selected_routes']], bins=100).set(title=f'{plot_type}')
        else:
            raise "Bad name for plot type"
    plt.suptitle(f'Several figures for time consuming parts of simulation\n{nodes} nodes with {workers} workers per node')
    plt.savefig(path + 'visual.png')

    # Get times in seconds
    simulation_time = simulation.duration.total_seconds()
    print(f'Simulation time: {simulation_time}')

    df_parts = df_parts.sum() / 1000
    measured_time = df_parts.sum()
    print('Time in seconds')
    print(df_parts)

    print('\nNormalized')
    print(df_parts / measured_time)

    plt.clf()
    (df_parts / measured_time).plot.bar(title=f'Computation distribution \n{nodes} nodes with {workers} workers per node')
    plt.xticks(rotation=30)
    plt.savefig(path + 'distribution.png')


if __name__ == '__main__':
    visualize()
