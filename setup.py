#!/usr/bin/env/ python

import sys
from setuptools import setup, find_packages

if sys.version_info.major < 3 or (
        sys.version_info.major == 3 and sys.version_info.minor < 6
):
    sys.exit("Python 3.6 or new is required")

VERSION = "0.1"

with open("requirements.txt") as reqs:
    requirements = [line.strip() for line in reqs.readlines()]

setup(
    name="ruth",
    version=VERSION,
    description="Framework for making a distrubuted deterministic simulator.",
    author="Martin Šurkovský",
    author_email="martin.surkovsky@vsb.cz",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ruth-simulator = ruth.tools.simulator:main",
            "ruth-simulator-conf = ruth.tools.simulator_conf:main",
            "ruth-data-preprocessing = ruth.tools.preprocessbenchmarkdata:main",
            "ruth-traffic-flow-to-od-matrix = ruth.tools.trafficflow2odmatrix:convert",
            "ruth-aggregate-fcd = ruth.tools.globalview2aggregatedfcd:aggregate_cmd",
            "ruth-speed-profiles = ruth.tools.speed_profiles:aggregate_speed_profiles",
            "ruth-od-matrix-to-simulator-input = ruth.tools.odmatrix2simulatorinput:convert",
            "ruth-combine-od-matrix = ruth.tools.combineodmatrix:combine_odmatrix",
            "ruth-distributed = ruth.zeromq.distributed:distributed",
            "traffic-flow-map-2 = ruth.flowmap.app:main",
        ]
    }
)
