
from shapely.geometry import Point
from dataclasses import dataclass


@dataclass
class GeoPoint:
    lat: float
    lon: float

    def point(self):
        # NOTE: shapely.geometry.Point takes the arguments in the opposite order
        return Point(self.lon, self.lat)
