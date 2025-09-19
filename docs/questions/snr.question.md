---
title: What is the easiest way to get the signal to noise ratio from the detector data buckets?
---

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