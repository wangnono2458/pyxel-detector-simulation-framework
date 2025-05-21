# Metadata

Each model is registered in the `metadata.yaml` file located at `pyxel/models/<model_group>/metadata.yaml`.

The `metadata.yaml` file contains structured information about each model, including:
- The model's short and fullname (e.g. 'simple_conversion' and 'Simple photoconversion')
- The detector types the model is valid for (e.g. all, CCD, CMOS, ...)
- The current status of the model: `draft`, `validated`, `in review`, `deprecated`, ...
- A detailed description of the model
- An example configuration snippet in YAML format
- Information about the author(s)
- A history or changelog for the model
- ...

## Example

Here is an example metadata entry for the {ref}`Simple photoconversion` model, which belongs 
to the {ref}`charge_generation` mode group.

This entry is defined in `pyxel/models/charge_generation/metadata.yaml`.

```yaml
models:
  - name: simple_conversion
    full_name: Simple photoconversion
    detector: all
    status: null
    description: >
      With this model you can create and add charge to :py:class:`~pyxel.detectors.Detector` via
      photoelectric effect by converting photons to charge.
      This model supports both monochromatic and multiwavelength photons, converting either a 2D photon array or
      3D photon array to the 2D charge array.
      If the previous model group :ref:`photon collection <photon_collection>` returns a 3D photon array, the
      photon array will be integrated along the wavelength dimension before applying the quantum efficiency (:term:`QE`).
      
      Binomial sampling of incoming Poisson distributed photons is used in the conversion by default,
      with probability :term:`QE`. It can be turned off by setting the argument ``binomial_sampling`` to ``False``.
      User can provide an optional quantum efficiency (``quantum_efficiency``) parameter.
      If not provided, quantum efficiency from detector :py:class:`~pyxel.detectors.Characteristics` is used.
      It is also possible to set the seed of the random generator with the argument ``seed``.
    warnings: Model assumes shot noise model was applied to photon array when using binomial sampling.
    config: >
      - name: simple_conversion
        func: pyxel.models.charge_generation.simple_conversion
        enabled: true
        arguments:
          quantum_efficiency: 0.8
```

## Usage

The `metadata.yaml` files will be used to generate documentation in various formats (e.g. for a GUI) and 
styles (text, html, pdf, ...), to filter models for specific detector types, ... .
