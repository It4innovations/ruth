import os
import click
from tqdm import tqdm
import pandas as pd
import numpy as np
import shapely.wkt
import random as rnd
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DT_FMT = "%Y-%m-%d %H:%M:%S%z"


@dataclass
class Record:
    lat_from: float
    lon_from: float
    lat_to: float
    lon_to: float
    datetime: datetime


class ODMatrix:
    LAT_FROM = "lat_from"
    LON_FROM = "lon_from"
    LAT_TO = "lat_to"
    LON_TO = "lon_to"
    TIME_OFFSET = "start_offset_s"
    FREQUENCY = "route_step_s"
    FCD_SAMPLING_PERIOD = "log_step_s"

    COLUMNS = [
        LAT_FROM,
        LON_FROM,
        LAT_TO,
        LON_TO,
        TIME_OFFSET,
        FREQUENCY,
        FCD_SAMPLING_PERIOD,
    ]

    def __init__(self):
        # TODO: change after the refactoring of the preprocessing will be done
        # ODMatrix.TIME_OFFSET: pd.Series(dtype="timedelta64[s]"),
        # ODMatrix.FREQUENCY: pd.Series(dtype="timedelta64[s]"),
        # ODMatrix.FCD_SAMPLING_PERIOD: pd.Series(dtype="timedelta64[s]"),

        self.data = pd.DataFrame(
            {
                ODMatrix.LAT_FROM: pd.Series(dtype="float64"),
                ODMatrix.LON_FROM: pd.Series(dtype="float64"),
                ODMatrix.LAT_TO: pd.Series(dtype="float64"),
                ODMatrix.LON_TO: pd.Series(dtype="float64"),
                ODMatrix.TIME_OFFSET: pd.Series(dtype="int64"),
                ODMatrix.FREQUENCY: pd.Series(dtype="int64"),
                ODMatrix.FCD_SAMPLING_PERIOD: pd.Series(dtype="int64"),
            }
        )

    def add(
            self,
            records: List[Record],
            frequency: timedelta,
            fcd_sampling_period: timedelta,
    ):
        # compute time offsets
        records = sorted(records, key=lambda record: record.datetime)
        assert len(records) > 0
        min_dt = records[0].datetime

        def to_nptd(td):
            return np.timedelta64(int(td / timedelta(seconds=1)), "s")

        def to_secs(td):
            return int(td / timedelta(seconds=1))

        records = map(
            lambda r: (
                r.lat_from,
                r.lon_from,
                r.lat_to,
                r.lon_to,
                to_secs(r.datetime - min_dt),
            ),
            records,
        )
        new_data = pd.DataFrame(records, columns=ODMatrix.COLUMNS[:-2])
        new_data[
            [ODMatrix.FREQUENCY, ODMatrix.FCD_SAMPLING_PERIOD]
        ] = (to_secs(frequency), to_secs(fcd_sampling_period))

        self.data = pd.concat([self.data, new_data])
        self.data.index.name = "id"  # name the id column as excepted in the following tools

    def store(self, path):
        self.data.to_csv(path, sep=";")


def random_point(minx, miny, maxx, maxy):
    return rnd.uniform(minx, maxx), rnd.uniform(miny, maxy)


def n_random_points(n, minx, miny, maxx, maxy):
    return [random_point(minx, miny, maxx, maxy) for _ in range(n)]


def box_center(minx, miny, maxx, maxy):
    return (minx + maxx) / 2, (miny + maxy) / 2


def rnd_time_in_range(dt1, dt2):
    assert dt1 < dt2
    return datetime.fromtimestamp(rnd.uniform(dt1.timestamp(), dt2.timestamp()))


@click.command()
@click.argument("traffic_flow_file_path", type=click.Path(exists=True), metavar="<input_path>")
@click.option("--frequency_s", type=int, default=20,
              help="A number of seconds between routing queries. Default value is 20s")
@click.option("--fcd_sampling_period_s", type=int, default=5,
              help="A number of seconds that divides the frequency."
                   " It is a sub-sampling period in which the FCD info is stored between queries."
                   " Default value is 5s")
@click.option("--out", type=click.Path(), default="out.csv", help="An output file name (CSV).")
def convert(traffic_flow_file_path, frequency_s, fcd_sampling_period_s, out):
    """The conversion tool that takes the csv file containing the traffic flow description in <input_path> and generates
     an Origin/Destination matrix. The traffic flow describes a number of devices (mobile phones)
     that started travelling from the origin area (rectangle) to the destination area (rectangle) at a particular time
     window.

     \b
     Expected columns of the input file are:
     =======================================
     \b
     NOTE: the following columns are obligatory; the order of columns can be arbitrary
     \b
       - start_time: a window start time
       - end_time: a window end time
       - count_devices: number of devices going from the origin rectangle to destination one.
       - geom_rectangle_from: geometry of origin rectangle
       - geom_rectangle_to: geometry of destination rectangle

     \b
     Output description:
     ===================

       For each device two random GPS points are generated, one in origin rectangle (LAT_FROM, LON_FROM) and the other
     in destination rectangle (LAT_TO, LON_TO). A departure time offset is also randomly picked from
     the time window (TIME_OFFSET).
    """
    logger.info(f"Converting the traffic flow to O/D matrix.")
    logger.info(f" * input: {traffic_flow_file_path}")
    logger.info(f" * frequency: {frequency_s}s")
    logger.info(f" * fcd sampling periods: {fcd_sampling_period_s}s")

    df = pd.read_csv(traffic_flow_file_path)

    records = []
    for i, record in tqdm(df.iterrows(), total=len(df)):
        start_time = datetime.strptime(
            f"{record.start_time}00", DT_FMT
        )  # add 00 at the end to change timezone format +02 or +01 to +0200, ..
        end_time = datetime.strptime(f"{record.end_time}00", DT_FMT)
        poly_from = shapely.wkt.loads(record.geom_rectangle_from)

        n = record.count_devices
        points_from = n_random_points(n, *poly_from.bounds)

        poly_to = shapely.wkt.loads(record.geom_rectangle_to)
        points_to = n_random_points(n, *poly_to.bounds)

        for arrow in zip(points_from, points_to):
            (x_from, y_from), (x_to, y_to) = arrow
            dt = rnd_time_in_range(start_time, end_time)
            records.append(Record(y_from, x_from, y_to, x_to, dt))

    od_matrix = ODMatrix()
    od_matrix.add(
        records,
        timedelta(seconds=frequency_s),
        timedelta(seconds=fcd_sampling_period_s),
    )

    out_path = os.path.abspath(out)
    od_matrix.store(out_path)
    logger.info(f"O/D matrix generated into: {out_path}.")


if __name__ == "__main__":
    convert()
