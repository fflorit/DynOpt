# DynOpt
DynOpt is a toolbox for chemical reaction optimization using dynamic experiments in a flow-chemistry setup, leveraging Bayesian optimization to suggest new dynamic experiments to perform in an experimental setup.

Data provided to the algorithm can come from both steady or dynamic experiments under different conditions (e.g., composition, temperature, residence time) in a continuous/Euclidean chemical design space. The algorithm will provide a trajectory (optimization parameters as a function of time) to explore such design space. Such trajectory can be run experimentally using a single dynamic experiment or (less efficiently) with a series of steady experiments in discrete location of the trajectory. After providing the new data to the algorithm (re-training), the procedure is repeated until the algorithm stopping criteria are met.

DynO is compatible with Python 3 (>= 3.6) and has been tested on Windows. For details about theory see the [paper](url) on dynamic experiments and the one on optimization [paper](url).

## Installation
You can install DynO via pip:
```
$ pip install dynopt
```

## Simplest use
DynO is meant to be used in a Python code with few simple APIs and then using the results in an experimental setup.

Simple example..........

**Advanced use**


&nbsp;

### Contributors
Federico Florit: [github](https://github.com/fflorit)

### Citation
If you use any part of this code in your work, please cite the [paper](url).
```
@article{placeholder,
  author  = {...},
  title   = {...},
  journal = {...},
  year    = {...},
  volume  = {...},
  number  = {...},
  pages   = {...},
  url     = {...}
}
```

### License
This software is released under a BSD 3-Clause license. For more details, please refer to
[LICENSE](https://github.com/fflorit/DynOpt/blob/main/LICENSE).

"Copyright 2023 Federico Florit"
