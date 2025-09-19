
# Frequently Asked Questions
- [How to contribute to Pyxel?](#how-to-contribute-to-pyxel)
- [How to install Pyxel?](#how-to-install-pyxel)
- [How to run Pyxel?](#how-to-run-pyxel)
- [Is there a simple example of a configuration file?](#is-there-a-simple-example-of-a-configuration-file)
- [Is there a way to add photons to a Detector directly from a file containing photons or pixel instead of converting from image in ADU to photons via the PTF?](#is-there-a-way-to-add-photons-to-a-detector-directly-from-a-file-containing-photons-or-pixel-instead-of-converting-from-image-in-adu-to-photons-via-the-ptf)
- [Is there a way to display all intermediate steps when a pipeline is executed ?](#is-there-a-way-to-display-all-intermediate-steps-when-a-pipeline-is-executed-)
- [Is there any way to use Pyxel to produce a bias or dark image without including any image file?](#is-there-any-way-to-use-pyxel-to-produce-a-bias-or-dark-image-without-including-any-image-file)
- [What are the different running modes in Pyxel?](#what-are-the-different-running-modes-in-pyxel)
- [What detectors types are implemented in Pyxel?](#what-detectors-types-are-implemented-in-pyxel)
- [What happens to my code when I contribute?](#what-happens-to-my-code-when-i-contribute)
- [What is the easiest way to get the signal to noise ratio from the detector data buckets?](#what-is-the-easiest-way-to-get-the-signal-to-noise-ratio-from-the-detector-data-buckets)
- [Where can I see the latest changes in Pyxel?](#where-can-i-see-the-latest-changes-in-pyxel)
- [Why do I retrieve a blank image but no error?](#why-do-i-retrieve-a-blank-image-but-no-error)

<a name="how-to-contribute-to-pyxel"></a>
## How to contribute to Pyxel?

If you found a bug or want to suggest a new feature, you can create an [issue on Gitlab](https://gitlab.com/esa/pyxel/-/issues).
If you have a question, you can use the Chat on [Gitter](https://gitter.im/pyxel-framework/community) to get help from the Pyxel community.
[Read more](https://esa.gitlab.io/pyxel/doc/stable/tutorials/get_help.html).

If you are using Pyxel on a regular basis and want to contribute, have a look into the 
[Contributing guide](https://esa.gitlab.io/pyxel/doc/latest/references/contributing.html). 
Contact us if you have further questions or want to participate in the monthly Pyxel Developer meeting.
You can always reach us via email: [pyxel@esa.int](mailto:pyxel@esa.int).

<a name="how-to-install-pyxel"></a>
## How to install Pyxel?

You can install Pyxel using Anaconda or Miniconda, Mamba, Pip or install it from source, cloning the 
[Pyxel Giltab repository](https://gitlab.com/esa/pyxel).
You can run also tutorials and examples **without prior installation** of Pyxel in a live session with 
[binder](https://mybinder.org/v2/gl/esa%2Fpyxel-data/HEAD?urlpath=lab).
Look at the [Installation Guide](https://esa.gitlab.io/pyxel/doc/stable/tutorials/install.html) to select your installation method.

<a name="how-to-run-pyxel"></a>
## How to run Pyxel?

You can run Pyxel from the command line, run it in a jupyter notebook or from a Docker container.
Look in the documentation to know [how to run Pyxel](https://esa.gitlab.io/pyxel/doc/stable/tutorials/running.html).

<a name="is-there-a-simple-example-of-a-configuration-file"></a>
## Is there a simple example of a configuration file?

There are a couple of models that are required in the pipeline to get an image in the end.  
One has to make use of simple models in the pipeline that the conversion **photon->charge->pixel->signal->image** is happening.

A simple pipeline example of a configuration yaml file in exposure mode can be found here: 
[simple_exposure.yaml](https://gitlab.com/esa/pyxel-data/-/blob/master/examples/exposure/simple_exposure.yaml).

<a name="is-there-a-way-to-add-photons-to-a-detector-directly-from-a-file-containing-photons-or-pixel-instead-of-converting-from-image-in-adu-to-photons-via-the-ptf"></a>
## Is there a way to add photons to a Detector directly from a file containing photons or pixel instead of converting from image in ADU to photons via the PTF?

Yes, with the model `load_image` in the photon generation model group it is possible to load photons directly from a file.
You can set the argument `convert_to_photons` to `false`, and it will use your input array without converting it via PTF.
See [here](https://esa.gitlab.io/pyxel/doc/stable/references/model_groups/photon_collection_models.html#load-image) for more details.

<a name="is-there-a-way-to-display-all-intermediate-steps-when-a-pipeline-is-executed-"></a>
## Is there a way to display all intermediate steps when a pipeline is executed ?

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

<a name="is-there-any-way-to-use-pyxel-to-produce-a-bias-or-dark-image-without-including-any-image-file"></a>
## Is there any way to use Pyxel to produce a bias or dark image without including any image file?

Without the models in the pipeline, Pyxel will still run, but will generate nothing, so just zero arrays. 
Dark image for example would be generated using a dark current model in charge_generation. 
The detector object stores the data and some properties that are needed by more than one model, 
but it doesn't directly influence how the stored data is edited, this information is in the pipeline.

<a name="what-are-the-different-running-modes-in-pyxel"></a>
## What are the different running modes in Pyxel?

There are three [running modes](https://esa.gitlab.io/pyxel/doc/stable/background/running_modes.html) in Pyxel:

[Exposure mode:](https://esa.gitlab.io/pyxel/doc/stable/background/running_modes/exposure_mode.html#exposure-mode) is used for a simulation of a single exposure, at a single or with incrementing readout times 
(quick look/ health check, simulation of non-destructive readout mode and time-dependent effects).
[Observation mode:](https://esa.gitlab.io/pyxel/doc/stable/background/running_modes/observation_mode.html) consists of multiple exposure pipelines looping over a range of model or detector parameters (sensitivity analysis).
[Calibration mode:](https://esa.gitlab.io/pyxel/doc/stable/background/running_modes/calibration_mode.html) is used to optimize model or detector parameters to fit target data sets using a user-defined fitness function/figure of merit 
(model fitting, instrument optimization).

<a name="what-detectors-types-are-implemented-in-pyxel"></a>
## What detectors types are implemented in Pyxel?

The following [detector types](https://esa.gitlab.io/pyxel/doc/stable/background/detectors.html#implemented-detector-types) 
are implemented in Pyxel:
- [CCD](https://esa.gitlab.io/pyxel/doc/stable/background/detectors/ccd.html)
- [CMOS](https://esa.gitlab.io/pyxel/doc/stable/background/detectors/cmos.html)
- [MKID](https://esa.gitlab.io/pyxel/doc/stable/background/detectors/mkid.html)
- [APD](https://esa.gitlab.io/pyxel/doc/stable/background/detectors/apd.html)

<a name="what-happens-to-my-code-when-i-contribute"></a>
## What happens to my code when I contribute?

If you want to contribute to Pyxel, have a look into the 
[Contributing guide](https://esa.gitlab.io/pyxel/doc/latest/references/contributing.html).
If you want your code to be published in the open-source Pyxel framework, you have to add a Licence to your code.
We are using a [MIT Licence](https://gitlab.com/esa/pyxel/-/blob/master/LICENSE.txt) 
and your code will have to have a Licence, which is compatible with the [MIT Licence](https://en.wikipedia.org/wiki/MIT_License).

Example: 
```python
# Copyright (c) <year> <name of model developer(s)>, <name of institution>
#
# <email address(es)>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
```

<a name="what-is-the-easiest-way-to-get-the-signal-to-noise-ratio-from-the-detector-data-buckets"></a>
## What is the easiest way to get the signal to noise ratio from the detector data buckets?

The easiest way is like this:

```python 
signal = result.signal.mean()
noise = result.signal.var()
snr = signal / noise
snr
```

The snr is an array with each exposure time in exposure mode 
(ndarray when using observation mode) with the result of the simulation, e.g. in exposure mode:
```python
result = pyxel.run_mode(
    mode=exposure,
    detector=detector,
    pipeline=pipeline,
)
```

<a name="where-can-i-see-the-latest-changes-in-pyxel"></a>
## Where can I see the latest changes in Pyxel?

The latest changes can be found in the [Changelog file](https://esa.gitlab.io/pyxel/doc/stable/references/changelog.html).

<a name="why-do-i-retrieve-a-blank-image-but-no-error"></a>
## Why do I retrieve a blank image but no error?

There are a couple of models that are required in the pipeline to get an image in the end.  
One have to make use of simple models in the pipeline that the conversion **photon->charge->pixel->signal->image** is happening.

Example: 
If you have only the model `load_image` in your pipeline and make use of the function `pyxel.display_detector(detector)`
you will retrieve the plot with photon, but the plots showing pixel or image are blank, because no conversion is taking place in the pipeline.

A simple pipeline example of a configuration yaml file in exposure mode can be found here: 
[simple_exposure.yaml](https://gitlab.com/esa/pyxel-data/-/blob/master/examples/exposure/simple_exposure.yaml).

<hr>

Generated by [FAQtory](https://github.com/willmcgugan/faqtory)