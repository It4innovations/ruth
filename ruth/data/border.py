import os
from enum import Enum
from pathlib import Path
from functools import reduce
from typing import Optional, List

import geopandas as gpd
from osmnx import geocode_to_gdf
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from matplotlib import cm

from .geopoint import GeoPoint
from ..log import console_logger as cl
from ..metaclasses import Singleton


class BorderType(Enum):
    COUNTRY = 2
    COUNTY = 6
    DISTRICT = 7
    TOWN = 8

    @property
    def admin_level(self):
        return self.value


class Border(metaclass=Singleton):
    def __init__(
        self,
        name: str,
        geocode: dict,
        kind: BorderType,
        data_dir,
        load_from_cache,
    ):

        self.name = name
        self.geocode = geocode
        self.kind = kind
        self.file_path = Path(os.path.join(data_dir, f"{name}.geojson"))

        self._super_area: Optional[Border] = None
        self._sub_areas: List[Border] = []

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        self.data, fresh_data = self._load(load_from_cache)
        if fresh_data:
            self._store()

    def __eq__(self, other):
        return self.name == other.name and self.geocode == other.geocode and self.kind == other.kind

    def __hash__(self):
        return hash((self.name,
                     tuple(sorted(self.geocode.items(), key=lambda kv: kv[0])),
                     self.kind))

    @property
    def super_area(self):
        return self._super_area

    @super_area.setter
    def super_area(self, area):
        area.add(self)

    @property
    def admin_level(self):
        return self.kind.admin_level

    def add(self, areas: List["Border"]):
        for area in areas:
            assert area.admin_level > self.admin_level, (
                "The added area has to be of lower administration level"
                " than the one to which is added."
            )

            if area.super_area is not None:
                raise Exception("Attempt to add an already assigned sub-area.")

            self._sub_areas.append(area)
            area._super_area = self

    def polygon(self):
        polygons = self.data.geometry.to_list()
        if all(map(lambda p: p.is_valid, polygons)):
            return reduce(lambda p1, p2: p1.union(p2), polygons, Polygon([]))
        raise Exception("Cannot create a polygon as not all polygons are valid.")

    def contains(self, geo_point: GeoPoint):
        polygon = self.polygon()
        return polygon.contains(geo_point.point())

    def bottom_up_borders(self, list=None):
        list_ = [] if list is None else list
        list_.append(self)
        if self.super_area is not None:
            return self.super_area.bottom_up_borders(list_)
        return list_

    def closest_border(self, geo_point: GeoPoint):
        if self.contains(geo_point):
            if len(self) > 0:
                for sb in self:
                    b = sb.closest_border(geo_point)
                    if b is not None:
                        return b
            return self
        return None

    def enclosed_border(self, start_geo_point, end_geo_point):
        if self.contains(start_geo_point) and self.contains(end_geo_point):
            return self

        if self.super_area is not None:
            return self.super_area.enclosed_border(start_geo_point, end_geo_point)

        return None

    def plot(self, *args, **kwargs):
        return self.data.plot(*args, **kwargs)

    def plot_with_context(self, *args, **kwargs):
        color_palete = cm.get_cmap("tab20c")

        top_down = reversed(self.bottom_up_borders())
        ax = plt.gca()
        for i, border in enumerate(top_down):
            new_kwargs = kwargs.copy()
            new_kwargs.update({"ax": ax, "color": color_palete(i)})
            ax = border.plot(*args, **new_kwargs)
        return ax

    def _load(self, load_from_cache):  # TODO: load asynchrnously
        if os.path.exists(self.file_path) and load_from_cache:
            cl.info(f"Loading data for '{self.name}' from localy stored data.")
            return (
                gpd.read_file(self.file_path, driver="GeoJSON"),
                False,
            )  # TODO: store driver info in a config
        else:
            cl.info(f"Loading data for '{self.name}' via the OSM API.")
            return (geocode_to_gdf(self.geocode), True)

    def _store(
        self,
    ):
        if self.data is not None:
            self.data.to_file(self.file_path, driver="GeoJSON")
            cl.info(f"Border of '{self.name}' saved into: '{self.file_path}'.")

    def __len__(self):
        return len(self._sub_areas)

    def __iter__(self):
        return iter(self._sub_areas)

    def __repr__(self):
        return f"Border({self.name})"
