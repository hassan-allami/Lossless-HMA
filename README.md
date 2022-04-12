# Lossless-HMAs
codes for calculating the lossless contours of doped highly mismatched alloys (HMAs)

## File Description
### LosslessFunctions.py
> is a collection of all functions that calculate plasma frequency of a doped HMA and determines if it can become lossless at some doping level.
### region_interband_collecting.py
> sweeps HMA parameter space for 3 different `\ell` and records the results in 3 different `.txt` files.
### region_interband_plotting.py
> generates plots like Fig. 3 reading from collected data.
> uses analytic form for `\ell = 0`.
### data files
> the already collected data for the 3 `\ell` used in the paper are included in the `data/regions` folder.
> running `region_interband_plotting.py` as it is would use them to generate Fig. 3.

**NOTE:** to generate the lossless contour for a different `\ell`, `region_interband_collecting.py` should be modified accordingly.
then `region_interband_plotting.py` should be modified to read from the collected data.

## Dependencies
numpy
scipy
matplotlib
functools

## How to Use
 
