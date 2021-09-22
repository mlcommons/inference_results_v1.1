# FuriosaAI MLPerf Quantization

FuriosaAI uses the following quantization scheme for 8-bit linear post-training
quantization.

## Activations

We use a per-tensor asymmetric post-training quantization scheme for
activations. To determine quantization parameters, scale and zero-point, for
activations, we collect the min-max range [MIN, MAX] of each activation in a
model over a calibration data set. The collected range gets extended, if
necessary, to include zero so that a floating-point zero can be precisely
represented in the quantized range. We linearly map the resulting (asymmetric)
range to the int8 range [-128, 127].

## Weights

We use a per-tensor symmetric post-training quantization scheme for weights,
except for convolutions' weights. For each weight in a model, we find its
min-max range [MIN, MAX] and calculate the maximum of the absolute values of
MIN and MAX. Instead of mapping the min/max range as we do with activations, we
map a symmetric range [-max(|MIN|, |MAX|), max(|MIN|, |MAX|)] into the int8
range [-127, 127], guaranteeing that zero-point is always 0. Convolutions'
weights are quantized in a per-channel symmetric fashion and have a different
scale parameter for each channel.
