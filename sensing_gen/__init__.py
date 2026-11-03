"""sensing_gen: simulation + generative modeling for real-world physiological time series.

Sub-packages
------------
data       PhysioNet readers (WFDB format 16 / 212), PTB-XL and MIT-BIH NSTDB loaders.
physics    Differentiable ECG dynamical model (McSharry et al., 2003) and a lumped-element
           electrode-skin sensor / artifact model used for simulation and augmentation.
pinn       Inverse physics-informed neural network for ECG denoising + parameter recovery.
diffusion  Conditional 1-D denoising diffusion model (DDPM training, DDIM sampling, CFG).
eval       Fidelity metrics (Frechet distance, MMD, spectral distance, discriminative score)
           and downstream train/evaluate utilities.
experiments  End-to-end scripts that produce every number reported in sensing_gen/README.md.
"""

__version__ = "0.1.0"
