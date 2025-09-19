---
title: Is there a way to display all intermediate steps when a pipeline is executed ?
---
Yes, you can run a pipeline enabling parameter `with_intermediate_steps` with function 
[`pyxel.run_mode`](https://esa.gitlab.io/pyxel/doc/stable/references/api/run.html#pyxel.run_mode).

In this example, the result of all models executed in the pipelines are displayed (`load_image`, `photoelectrons`, ...).

```python
>>> import pyxel
>>> config = pyxel.load("configuration.yaml")
>>> data_tree = pyxel.run_mode(
...     mode=config.exposure,
...     detector=config.detector,
...     pipeline=config.pipeline,
...     with_intermediate_steps=True,  # <== Enable this
... )
>>> data_tree["/data/intermediate"]  # <== Display all intermediate results
    DataTree('intermediate', parent="data")
    │   Attributes:
    │       long_name:  Store all intermediate results modified along a pipeline
    └── DataTree('idx_0')
        │   Attributes:
        │       long_name:       Pipeline for one unique time
        │       pipeline_count:  0
        │       time:            1.0 s
        ├── DataTree('photon_collection')
        │   │   Attributes:
        │   │       long_name:  Model group: 'photon_collection'
        │   └── DataTree('load_image')
        │           Dimensions:  (y: 100, x: 100)
        │           Coordinates:
        │             * y        (y) int64 0 1 2 3 4 5 6 7 8 9 10 ... 90 91 92 93 94 95 96 97 98 99
        │             * x        (x) int64 0 1 2 3 4 5 6 7 8 9 10 ... 90 91 92 93 94 95 96 97 98 99
        │           Data variables:
        │               photon   (y, x) float64 1.515e+04 1.592e+04 ... 1.621e+04 1.621e+04
        │           Attributes:
        │               long_name:  Group: 'load_image'
        ├── DataTree('charge_generation')
        │   │   Attributes:
        │   │       long_name:  Model group: 'charge_generation'
        │   └── DataTree('photoelectrons')
        │           Dimensions:  (y: 100, x: 100)
        │           Coordinates:
        │             * y        (y) int64 0 1 2 3 4 5 6 7 8 9 10 ... 90 91 92 93 94 95 96 97 98 99
        │             * x        (x) int64 0 1 2 3 4 5 6 7 8 9 10 ... 90 91 92 93 94 95 96 97 98 99
        │           Data variables:
        │               charge   (y, x) float64 1.515e+04 1.592e+04 ... 1.621e+04 1.621e+04
        │           Attributes:
        │               long_name:  Group: 'photoelectrons'
        ├── DataTree('charge_collection')
        │   │   Attributes:
        │   │       long_name:  Model group: 'charge_collection'
        │   └── DataTree('simple_collection')
        │           Dimensions:  (y: 100, x: 100)
        │           Coordinates:
        │             * y        (y) int64 0 1 2 3 4 5 6 7 8 9 10 ... 90 91 92 93 94 95 96 97 98 99
        │             * x        (x) int64 0 1 2 3 4 5 6 7 8 9 10 ... 90 91 92 93 94 95 96 97 98 99
        │           Data variables:
        │               pixel    (y, x) float64 1.515e+04 1.592e+04 ... 1.621e+04 1.621e+04
        │           Attributes:
        │               long_name:  Group: 'simple_collection'
        ├── DataTree('charge_measurement')
        │   │   Attributes:
        │   │       long_name:  Model group: 'charge_measurement'
        │   └── DataTree('simple_measurement')
        │           Dimensions:  (y: 100, x: 100)
        │           Coordinates:
        │             * y        (y) int64 0 1 2 3 4 5 6 7 8 9 10 ... 90 91 92 93 94 95 96 97 98 99
        │             * x        (x) int64 0 1 2 3 4 5 6 7 8 9 10 ... 90 91 92 93 94 95 96 97 98 99
        │           Data variables:
        │               signal   (y, x) float64 0.04545 0.04776 0.04634 ... 0.05004 0.04862 0.04862
        │           Attributes:
        │               long_name:  Group: 'simple_measurement'
        └── DataTree('readout_electronics')
            │   Attributes:
            │       long_name:  Model group: 'readout_electronics'
            └── DataTree('simple_adc')
                    Dimensions:  (y: 100, x: 100)
                    Coordinates:
                      * y        (y) int64 0 1 2 3 4 5 6 7 8 9 10 ... 90 91 92 93 94 95 96 97 98 99
                      * x        (x) int64 0 1 2 3 4 5 6 7 8 9 10 ... 90 91 92 93 94 95 96 97 98 99
                    Data variables:
                        image    (y, x) uint32 298 314 304 304 304 314 ... 325 339 339 328 319 319
                    Attributes:
                        long_name:  Group: 'simple_adc'

Display changes from the first model executed (here `photon_collection.load_image`)

>>> data_tree["/data/intermediate/idx_0/photon_collection/load_image"]
DataTree('load_image')
    Dimensions:  (y: 100, x: 100)
    Coordinates:
      * y        (y) int64 0 1 2 3 4 5 6 7 8 9 10 ... 90 91 92 93 94 95 96 97 98 99
      * x        (x) int64 0 1 2 3 4 5 6 7 8 9 10 ... 90 91 92 93 94 95 96 97 98 99
    Data variables:
        photon   (y, x) float64 1.515e+04 1.592e+04 ... 1.621e+04 1.621e+04
    Attributes:
        long_name:  Group: 'load_image'

Display changes from the second model executed (here `charge_generation/photoelectrons`)

>>> data_tree["/data/intermediate/idx_0/charge_generation/photoelectrons"]
DataTree('charge_generation')
│   Attributes:
│       long_name:  Model group: 'charge_generation'
└── DataTree('photoelectrons')
        Dimensions:  (y: 100, x: 100)
        Coordinates:
          * y        (y) int64 0 1 2 3 4 5 6 7 8 9 10 ... 90 91 92 93 94 95 96 97 98 99
          * x        (x) int64 0 1 2 3 4 5 6 7 8 9 10 ... 90 91 92 93 94 95 96 97 98 99
        Data variables:
            charge   (y, x) float64 1.515e+04 1.592e+04 ... 1.621e+04 1.621e+04
        Attributes:
            long_name:  Group: 'photoelectrons'     
```
