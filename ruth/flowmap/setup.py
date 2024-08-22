from setuptools import setup, find_packages

with open("requirements.txt") as reqs:
    requirements = [line.strip() for line in reqs.readlines()]

setup(
    name='flowmap',
    version='0.2.1',
    description='Package generating video from traffic history records.',
    author='Sofia Michailidu, Pavlína Smolková',
    author_email='mic0427@vsb.cz, smo0117@vsb.cz',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "traffic-flow-map = flowmap.app:main"
        ]
    }
)
