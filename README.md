# Lossless-HMAs
codes for calculating the lossless contours of doped highly mismatched alloys (HMAs)

## File Description

### `LosslessFunctions.py`
> is a collection of all functions that calculate plasma frequency of a doped HMA and determine if it can become lossless at some doping level.  
> has separate functions for the `\ell = 0` case, which use analytic forms.
 
### `region_interband_collecting.py`
> sweeps HMA parameter space for 3 `\ell`s and records the results in 3 `.txt` files.

### `region_interband_plotting.py`
> generates plots like Fig. 3 reading from collected data.  
> uses analytic form for `\ell = 0`.

### `data files`
> The already collected data for the 3 cases used in the paper are included in the `data/regions` folder.  
> Running `region_interband_plotting.py` as it is would use them to generate Fig. 3.

**NOTE:** To generate the lossless contour for a different `\ell`, `region_interband_collecting.py` should be modified accordingly.
Then `region_interband_plotting.py` should be modified to read from the collected data.  
_On a typical laptop it takes several hours to to collect data for a single_ `\ell` _with the default resolution._
_The resolution is adjustable._


## Dependencies
```
numpy  
scipy  
matplotlib  
functools
```
