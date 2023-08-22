# FlowMapVideo

FlowMapVideo is a tool for visualizing the evolution of traffic flow over time. The output is a video with a map showing the traffic flow intensities at each time interval. The animation is generated using the [FlowMapFrame](https://github.com/It4innovations/FlowMapFrame) library to render individual frames based on the output of the [Ruth](https://github.com/It4innovations/ruth) traffic simulator. It is designed for linux systems.

## Installation

### Prerequisites

To run, you need to install `FFmpeg` and [Ruth](https://github.com/It4innovations/ruth).

```
sudo apt install ffmpeg
```
### Install

1. Create and activate a virtual environment:
```
virtualenv <VENV>
source <VENV>/bin/activate
```


2. Install via pip
```
python3 -m pip install git+https://code.it4i.cz/intern/trafficflowmap/flowmapopt.git@bench/no-dataframe
```

## Run
```
traffic-flow-map --help
```

### Check the animation length 
use `get-info` to check the animation length for given data and speed
```
traffic-flow-map get-info --help
```
with parameters:
* `--time-unit` - time unit of the information about the animation
* `--speed` - speed of the animation

### Generate the animation 
use `generate-animation`:
```
traffic-flow-map generate-animation --help
```
Don't forget to specify the `--speed` parameter that was tested with `get-info` command.

For fixed number of vehicles that will be depicted with maximum line width, use the `--max-width-density` parameter.

#### Example
```
traffic-flow-map generate-animation <PATH_TO_DATA> --speed 350 --title "Traffic flow" -c
```

