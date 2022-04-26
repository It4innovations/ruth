import os
from enum import Enum
from pathlib import Path
from functools import reduce
from typing import Optional, List
from lazy_object_proxy import Proxy as LazyProxy
from hashlib import md5

import geopandas as gpd
from osmnx import geocode_to_gdf
from shapely.geometry import Polygon
import shapely.wkt
from shapely.errors import WKTReadingError

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

    @classmethod
    def parse(cls, kind):
        kind_ = kind.lower()
        if kind_ == "country":
            return BorderType.COUNTRY
        elif kind_ == "county":
            return BorderType.COUNTY
        elif kind_ == "district":
            return BorderType.DISTRICT
        elif kind_ == "town":
            return BorderType.TOWN
        else:
            raise ValueError(f"Invalid kind: '{kind}'."
                             " Allowed values are: 'country', 'county', 'district', and 'town'.")


class BorderDefinition:

    def __hash__(self):
        return hash(f"{self.md5()}")

    def __eq__(self, other):
        raise NotImplementedError("Subclasses should implement this!")

    def md5(self):
        raise NotImplementedError("Subclasses should implement this!")

    def load(self):
        raise NotImplementedError("Subclasses should implement the data loading method!")


class GeocodeBorderDef(BorderDefinition):

    def __init__(self, geocode: dict):
        self.geocode = geocode

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        return self._get_sorted_tuple() == other._get_sorted_tuple()

    def md5(self):
        return md5(repr(self._get_sorted_tuple()).encode('UTF-8')).hexdigest()

    def _get_sorted_tuple(self):
        # get unique representation of the dictionary
        return tuple(sorted(self.geocode.items(), key=lambda kv: kv[0]))

    def load(self):
        cl.info(f"Loading data for '{self.name}' via the OSM API.")
        return geocode_to_gdf(self.geocode)


class PolygonBorderDef(BorderDefinition):

    def __init__(self, polygon: str, on_disk=False):
        self.on_disk = on_disk

        try:
            self.polygon = None if polygon is None else shapely.wkt.loads(polygon)
        except WKTReadingError as e:
            cl.error(f"Invalid polygon format: '{polygon}'")
            raise e

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        return self.polygon == other.polygon

    def md5(self):
        if self.polygon is None:
            return -1
        return md5(self.polygon.wkt.encode('UTF-8')).hexdigest()

    def load(self):
        if self.on_disk and polygon is None:
            # fails if not present on the disk
            raise Error("The data should be loaded from the disk. "
                        "If not provide exisiting name of the border.")

        return gpd.GeoSeries(self.polygon)


class Border(metaclass=Singleton):
    def __init__(
        self,
        name: str,
        border_def: BorderDefinition,
        kind: BorderType,
        data_dir,
        load_from_cache,
    ):

        self.name = name
        self.border_def = border_def
        self.kind = kind
        self.file_path = Path(os.path.join(data_dir, f"{name}.geojson"))

        self._super_area: Optional[Border] = None
        self._sub_areas: List[Border] = []

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        self.data = self._load(load_from_cache)

    def __eq__(self, other):
        return self.name == other.name and self.border_def == other.border_def and self.kind == other.kind

    def __hash__(self):
        return hash((self.name, self.border_def, self.kind))

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
                f"The added area ({area.name}/{area.admin_level}) has to be of lower"
                " administration level"
                f" than the one to which is added ({self.name}/{self.admin_level})."
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

        def load_from_file():
            cl.info(f"Loading data for '{self.name}' from localy stored data.")
            return gpd.read_file(self.file_path, driver="GeoJSON"),

        def download_based_on_border_def():
            data = self.border_def.load()
            self._store(data)
            return data

        if os.path.exists(self.file_path) and load_from_cache:
            return LazyProxy(load_from_file)  # TODO: store driver info in a config
        return LazyProxy(download_based_on_border_def)

    def _store(self, data):
        if data is not None:
            data.to_file(self.file_path, driver="GeoJSON")
            cl.info(f"Border of '{self.name}' saved into: '{self.file_path}'.")

    def __len__(self):
        return len(self._sub_areas)

    def __iter__(self):
        return iter(self._sub_areas)

    def __repr__(self):
        return f"Border({self.name})"
