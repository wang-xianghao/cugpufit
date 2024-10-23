# cuGpufit
cuGpufit supports Levenberg-Marquardt fitting on multiple-GPUs via cuNumeric.

This codes are adapted from [fabiodimarco/tf-levenberg-marquardt](https://github.com/fabiodimarco/tf-levenberg-marquardt), which implements Levenberg-Marquardt optimizer by Tensorflow.

# Quick start
```python
from cugpufit import Model
from cugpufit.fits import LevenbergMarquardtFit
from cugpufit.losses import MeanSquaredError

# Prepare train data
...

# Create model
model = YourModel()

# Train
lm_fit = LevenbergMarquardtFit(model, MeanSquaredError())
lm_fit.fit(X_train, Y_train, epoches=10, batch_size=batch_size)

# Predict 
Y_test = model.predict(X_test)
```
Please see [examples/curve_fitting.py](./examples/curve_fitting.py) for details of define your own model.

# Limitations
* Auto-differentiation is not supported, users should manually implement Jacobian computation in `model.compute_jacobian_with_outputs()`.