# Partially Coherent Sources

This Python package provides functionality for generating masks for the production of partially coherent light beams. It includes two methods for mask generation: phase randomized and appearance probability.

**Docs**: (https://marcsgil.github.io/partially_coherent_sources/)[https://marcsgil.github.io/partially_coherent_sources/]
**GitHub**: (https://github.com/marcsgil/partially_coherent_sources.git)[https://github.com/marcsgil/partially_coherent_sources.git]

## Installation

To install this package, clone the repository and install using pip:

```bash
pip install git+https://github.com/marcsgil/partially_coherent_sources.git
```

## Usage
The main function in this package is `generate_masks`. Here is a basic example of how to use it:

```py
import numpy as np
from partially_coherent_sources import generate_masks

# Define the number of masks, weights, and fields
n_masks = 10
weights = np.array([0.1, 0.2, 0.3, 0.4])
fields = np.random.rand(4, 5, 5)

# Generate masks using the phase randomized method
masks = generate_masks(n_masks, weights, fields, method='phase_randomized')

# Generate masks using the appearance probability method
masks = generate_masks(n_masks, weights, fields, method='appearance_probability')
```
