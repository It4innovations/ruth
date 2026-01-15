#!/usr/bin/env/ python

import sys
from setuptools import setup, find_packages

if sys.version_info.major < 3 or (
        sys.version_info.major == 3 and sys.version_info.minor < 6
):
    sys.exit("Python 3.6 or new is required")

VERSION = "2.1"

with open("requirements.txt") as reqs:
    requirements = [line.strip() for line in reqs.readlines()]

setup(
    name="ruth",
    version=VERSION,
    description="Framework for making a distrubuted deterministic simulator.",
    author="IT4Innovation",
    author_email="paulo.silva@vsb.cz",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ruth-simulator = ruth.tools.simulator:main",
            "ruth-simulator-conf = ruth.tools.simulator_conf:main",
            "ruth-traffic-flow-to-od-matrix = ruth.tools.trafficflow2odmatrix:convert",
            "ruth-od-matrix-to-simulator-input = ruth.tools.odmatrix2simulatorinput:convert",
            "ruth-combine-od-matrix = ruth.tools.combineodmatrix:combine_odmatrix",
            "ruth-distributed = ruth.zeromq.distributed:distributed",
            "traffic-flow-map = ruth.flowmap.app:main",
        ]
    }
)
