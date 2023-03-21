# Ruth

_Ruth_ is a deterministic traffic simulator designed to simulate traffic flow within the cities. With the simulator we aim to optimize the entire traffic flow within a city instead of optimizing routes for particular drivers. For this purpose the simulator contains several routing algoritms including our in-house _Probabilistic Time Dependent Routing_ (PTDR). Its main component is an estimator of of delay on a route at departure time, called _probable delay_. It can be seen as kind of forecast what happens on route in future.

Computation of _probable delay_ is based on Monte Carlo simulation. A crucial input for this step is having access to _probabality profiles of level of service on route segments_. These can be computed based on either historical data or the result of simulation as it produces the same output, i.e. _Floating Car Data_ (FCD).

## Use cases

- urban planers, i.e., what-if analysis

- data booster for AI training data


## References
TODO:

