Theory
======

A system's Raman spectrum is related to the frequencies at which the components of the dielectric tensor fluctuate with time. 




Therefore, to calculate a Raman spectrum from a molecular dynamics trajectory, one needs to derive a timeseries of the dielectric tensor. This could be done by calculating the dielectric tensor at every molecular dynamics timestep. However, this is computationally *extremely* expensive. Instead, ramannoodle approximates this timeseries by assuming each atom has an independent effect on the dielectric constant dependent on that atom's displacement from it's equilibrium position. 



