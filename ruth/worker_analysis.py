import click
import glob
import pandas

from natsort import natsorted


@click.command()
@click.option('--experiment_path', type=str)
@click.option('--output_path', type=str, default='')
def analyze_workers(experiment_path, output_path):
    """
    Assumes:
     all workers have: worker_id
     all nodes have: node_id
     experiment folder contains only experimental files (experiment_path/node_id/worker_id)
    """
    table = {}

    # Find all nodes
    for i, node_path in enumerate(natsorted(glob.glob(f'{experiment_path}/node_*', recursive=True))):
        node_id = i + 1
        node_alternatives = []
        for j, worker_path in enumerate(natsorted(glob.glob(f'{node_path}/worker_*', recursive=True))):
            worker_id = j
            out_file, err_file = f'{worker_path}/worker_{worker_id}.out', f'{worker_path}/worker_{worker_id}.err'

            file = open(out_file, "r")
            content = file.readlines()
            file.close()

            alternatives_count = 0
            for line in content:
                if 'Alternative solutions' in line:
                    alternatives_count += 1

            node_alternatives.append(alternatives_count)
        table[f'node_{node_id}'] = node_alternatives

    worker_index = [f'worker_{id}' for id in range(len(node_alternatives))]
    df = pandas.DataFrame(table, index=worker_index)

    # Write to file
    if output_path == '':
        df.to_csv(f'{experiment_path}/worker_analysis.csv')
    else:
        df.to_csv(output_path)


if __name__ == '__main__':
    analyze_workers()