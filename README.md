# Mixed Crystal Signature
Python 3 software to locally classify crystalline structures in 3D pointcloud data as retrieved from moleculardynamical simulations, colloids or complex plasmas.

# Features
- Calculation of a feature vector for local classification of crystalline structures as described in [Dietz et al.](https://doi.org/10.1103/PhysRevE.96.011301)
- Training of a neural network with artificial crystal lattices of fcc, bcc, and hcp

# Tutorials
- [Crystal analysis using MCS](analyzecrystal_example.ipynb)
- [Training](training_example.ipynb)

# Dependencies
- Numpy
- Scipy
- Scikit-learn
- Pandas
- Sympy
- Numba
- Multiprocessing (optional)

# Installation
- Download the repository.
- Install the [anaconda](https://anaconda.org/) Python 3 distribution
- The package should work out of the box in a standard anaconda installation

If requested, I might create a complete Anaconda package for this repository.

# Citation
If you use this package for a publication, we would be very happy to be cited:

Dietz, C., T. Kretz, and M. H. Thoma. "Machine-learning approach for local classification of crystalline structures in multiphase systems." *Physical Review E* 96.1 (2017): 011301.
