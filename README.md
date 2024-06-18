# LSCTSplitter

## Overview
The LSCTSplitter is a Python package designed to decompose the yield curve into its fundamental components: level, slope, curvature, and twist. This package includes functionalities to fit the LSCT model to the provided data, calculate errors, perform inverse transformations, and plot the results.
## Example
Here is a complete example of how to use the LSCTSplitter package.

```python
from lsctsplitter import LSCTSplitter, loadYC
data = yc.load('./data/yc_data.csv')
maturities=[1.0/12, 2.0/12, 3.0/12, 6.0/12, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]

# LSCT transform
lscsplitter = LSCTSplitter(maturities, n_factors=4, lambdas=0.49, verbose=0)
factors = lscsplitter.fit_transform(data.values)

# Plot factors along the time axis
lscsplitter.plot_factors(data.values, t = data.index)

# Inverse LSCT factors into yield curve data
Xs = lscsplitter.inverse_transform(factors)
```
## Reference
- [Nelson, Charles R., and Andrew F. Siegel. "Parsimonious modeling of yield curves." Journal of business (1987): 473-489.](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.jstor.org/stable/pdf/2352957.pdf?casa_token=us1W8496haEAAAAA:qkjbLPi2BOklfh6Zv3ypmg-Ya0Yy_7TkdLwuC8Nc1k9aEqyiaGj9DlufKO4U0V9eRWGWbwGvc3N43LNYa1VABLM3i5tCP998VhHvIEB6-zoFv92fvcI)
- [Ma, Xuyang. "The twist factor of yields." Advances in Economics and Business 5 (2017): 411-422.](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.hrpub.org/download/20170730/AEB4-11808719.pdf)
- [Ayliffe, Kelly and Rubin, Tomas. "A Quantitative Comparison of Yield Curve Models in the MINT Economies". EPFL Infoscience.](http://infoscience.epfl.ch/record/279314)
- [Tomas Rubin, Yield Curve Forecasting](https://github.com/tomasrubin/yield-curve-forecasting?tab=readme-ov-file)
