# LSCTSplitter

## Overview
The LSCTSplitter is a Python package designed to decompose the yield curve into its fundamental components: level, slope, curvature, and twist. This package includes functionalities to fit the LSCT model to the provided data, calculate errors, perform inverse transformations, and plot the results.
## Example
Here is a complete example of how to use the LSCTSplitter package.

```python
import pandas as pd
import numpy as np
import datetime
from helper.utils import Timer
from your_module import LSCTSplitter, loadYC

if __name__ == "__main__":
    # Load data
    data = loadYC()
    data = data.tail(252)
    
    # Initialize LSCTSplitter
    lscsplitter = LSCTSplitter()
    
    # Fit model
    lscsplitter.fit(data=data)
    
    # Perform inverse transformation
    lscsplitter.inverse()
    
    # Plot factors
    lscsplitter.plotfactors()
    
    # Plot data and residuals
    lscsplitter.plot()
    lscsplitter.plot(resid=True)
```
