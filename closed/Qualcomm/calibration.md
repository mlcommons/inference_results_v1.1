# Qualcomm - Calibration Details

We use regular profile-guided post-training quantization.
We pass a set of calibration samples through the neural network to obtain a
profile of tensor values for the network operations.  We then use the profile
to calculate the scales and offsets for quantization to have a negligible
impact on accuracy.
