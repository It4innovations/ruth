import logging

from src import worker
import click


@click.command()
@click.option('--address', default='localhost', help='Client address')
@click.option('--port', default=5560, type=int, help='Client port')
@click.option('--map', help='Graph filepath')
def main(address: str, port: int, map: str):
    logging.info(f"Starting alternatives worker. Connecting to {address}:{port} with map {map}")
    w = worker.Worker(map=map, address=address, port=port)
    w.run()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(levelname)-4s %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )
    main()
