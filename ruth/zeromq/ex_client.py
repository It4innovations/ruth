from src.client import Client
import json
import time
import click


@click.command()
@click.option('--msg_count', default=20, help='Amount of messages to send')
def main(msg_count):
    client = Client()

    # Create dummy array
    array = [[x, x + 1] for x in range(msg_count)]

    # Serialize array
    array = [json.dumps(x).encode() for x in array]

    start = time.time()
    results = client.compute(array)
    end = time.time()

    print(f'Computation time: {end - start} for messages: {msg_count}')
    print(f'Cars per second: {msg_count/(end-start)}')
    # print(results)


if __name__ == '__main__':
    main()
