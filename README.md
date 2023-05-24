# NegInvVector3D
## Negation Invariant Representations of 3D Vectors
This repository accompanies the following paper published in IEEE TGRS (Transactions on Geoscience and Remote Sensing):

D. Kluvanec, K. J. W. McCaffrey, T. B. Phillips and N. A. Moubayed,
"Negation Invariant Representations of 3D Vectors for Deep Learning Models applied to Fault Geometry Mapping in 3D Seismic Reflection Data,"
in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2023.3273329.

https://doi.org/10.1109/TGRS.2023.3273329

## Description
This repository contains an implementation of the representations in python using pytorch.

The functions in the repository transform 3D vectors ***v*** to representations ***r*** for a batches of vectors of shape [batch, vector, ...], where the operation is broadcast along the batch dimension and any other following dimensions.

The following are the implemented representations:
- `doubleangle`
  - This is a representation of 2D vectors
  - Requires 4 dimensions
- `dip90_strike360`
  - This is an imperfect representation and has discontinuities
  - Requires 3 dimensions
- `dip180_strike180`
  - This is an imprefect representaiton and has discontinuities
  - Requires 3 dimensions
- `projection_doubleangle`
  - This is the best representation in our experiments
  - Requires 6 dimensions
- `piecewise_align`
  - Requires 9 dimensions
  - (Could be expanded to n-dimensional vectors easily. The representation would require n<sup>2</sup> dimensions.)
- `classification_dip_strike`
  - Requires 10 dimensions
- `classification_icosahedron`
  - Requires 6 dimensions

Functions are named `vector_2_[representation]` and `[representation]_2_vector`, where `[representation]` is replaced with the name of the representaiton.

## Installation
`pip install git+https://github.com/KluvaDa/NegInvVector3D.git`
