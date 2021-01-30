# Interpolation_CUPY
The cupy verson of cubic spline interpolation which is transferd from scipy.interpolation

You need an Nvidia GPU to run this. The main function is CuCubicSpline.CubicSpline(), which same as [scipy.interpolate.CubicSpline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html) but much faster. 

Details of why I started this project and some troubles I met is on stackoverflow: [link](https://stackoverflow.com/questions/60663086/how-to-do-a-cubic-spline-interpolation-on-python-with-cupy/63625626#63625626)
