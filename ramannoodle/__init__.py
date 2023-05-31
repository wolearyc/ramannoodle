"""Contains classes and functions to aid in the calculation of Raman specta
from VASP molecular dynamics calculations."""

import gc

import numpy as np

import scipy
from scipy.fftpack import fft, ifft, fftfreq, fftshift
import scipy.signal

import matplotlib

import ase
from ase import Atoms
import ase.io

import phonopy

import spglib

print_status  = True
print_warning = True
def set_print_verbosity(print_status = True, print_warning = True):
    """Sets the printing verbosity.

    Parameters
    ----------
    print_status : bool
        If true, useful status information will be printed.
    print_warning: bool
        If true, warning information will be printed. 
    """
    print_status = print_status
    print_warning = print_warning

def print_status(msg, end = '\n'):
    """Prints status information (if allowed)

    Parameters
    ----------
    msg : str
        The message to print 
    end : str
        Terminating character (default is a new line)
    """
    if print_status:
        print(msg, end = end)
def print_warning(msg):
    """Prints warning information (if allowed)

    Parameters
    ----------
    msg : str
        The message to print 
    """
    if print_warning:
        print('WARNING |', msg)
def print_error(msg):
    print('ERROR |', msg)
    assert(False)


# Boltzmann Constant in eV/K
kb = 8.617332478E-5
# electron volt in Joule
ev = 1.60217733E-19
# Avogadro's Constant
Navogadro = 6.0221412927E23


def get_signal_correlation(signal):
    """Gets the positive autocorrelation of signal

    Parameters
    ----------
    signal : numpy.ndarray
        The signal
    
    Returns
    -------
    numpy.ndarray
        The positive autocorrelation of signal
    """
    new_signal = signal
    AF = scipy.signal.correlate(new_signal, new_signal, 'full')
    AF = AF[(len(AF)-1)//2:] # no normalization
    return AF

def get_signal_spectrum(signal, potim):
    """Returns the (positive frequency) Fourier transform of the autocorrelation 
    of a signal.

    Parameters
    ----------
    signal : numpy.ndarray
        The signal
    potim : float
        The signal's sampling rate in fs (typically the molecular dynamics 
        timestep)

    Returns
    -------
    tuple
        The first and second components are wavenumbers (in inverse cm)
        and intensities (arbitrary units) respectively. 
    """
    timesteps = range(len(signal)-1)
    N = len(timesteps)
    AF = get_signal_correlation(signal)
    omega = fftfreq(AF.size, potim) * 33.35640951981521 * 1E3 # convert to cm-1
    intensities = np.real(fft(AF))
    wavenumbers = omega[omega>=0]
    intensities = intensities[omega>=0]
    return(wavenumbers, intensities)


class Trajectory:
    """Stores a molecular dynamics trajectory output by VASP.

    Parameters
    ----------
    vasprun : str
        Path to a vasprun.xml of the molecular dynamics run.
    potim : float
        Timestep used for molecular dynamics in fs.
    load_forces : bool
        Specifies whether to load forces from the vasprun.xml file. Forces 
        are not required for Raman calculations, and loading them will 
        increase the memory footprint substantially. However, one may wish
        to load forces for the purposes of convergence testing, etc.
    """
    def __init__(self, vasprun, potim = 1, load_forces = False):
        """Constructor method.
        """

        # Use a class within phonopy to load the vasprun.xml file. 
        # This is not memory efficient, so at some point I'd like to 
        # replace this with my own code.
        file = open(f'{vasprun}','rb')
        run = phonopy.interface.vasp.VasprunxmlExpat(file)
        print_status('Parsing vasprun xml...', end = '')
        run.parse()
        print_status('done.') 
        file.close()

        self.ase_atoms = Atoms(symbols = run.get_symbols(), 
                               scaled_positions = run.points[0], 
                               cell = run.lattice[0], pbc = [1,1,1])
        self.symbols = run.get_symbols()
        self.cell =run.lattice[0]
        self.potim = potim
        
        self.direct_positions = run.points
        self.Niter = self.direct_positions.shape[0]
        self.Nions = self.direct_positions.shape[1]
        self.forces = False
        if load_forces:
            self.forces = run.forces

        run = None
        gc.collect(generation=2)

        print_status(f'    {len(self.direct_positions)} {self.potim} fs steps')
        print_status(f'    {len(self.symbols)} atoms')
    
    
    def get_power_spectrum(self, timesteps = 'all', atom_indexes = 'all', 
                           direction_indexes = 'all', smearing = 0, normalize = True):
        
        """
        Returns the power spectrum of the trajectory. 

        Parameters
        ----------
        timesteps : str | numpy.ndarray, optional
            Timesteps to use in power spectrum generation. Should be continuous. 
            'all' specifies the use of all timesteps.
        atom_indexes : str | numpy.ndarray, optional
            Atom indexes to take into account. 'all' will use all atoms
        direction_indexes : str | ndarray, optional
            Cartesian directional indexes for each atom. 'all' will use all
            directions
        smearing : float, optional
            Gaussian smearing width. 0 turns off smearing
        normalize : bool, optional
            Whether or not to normalize
        
        Returns
        -------
        tuple
            The first and second components are wavenumbers (in inverse cm)
            and intensities (arbitrary units) respectively. 
        """

        velocity = np.diff(self.direct_positions, axis=0)
        # apply periodic boundary condition
        velocity[velocity > 0.5] -= 1.0
        velocity[velocity <-0.5] += 1.0

        # Velocity in Angstrom per femtosecond
        for i in range(self.Niter-1):
            velocity[i,:,:] = np.dot(velocity[i,:,:], self.cell) / self.potim
        
        
        if timesteps == 'all':
            timesteps = range(len(velocity))
        if atom_indexes == 'all':
            atom_indexes = range(self.Nions)
        if direction_indexes == 'all':
            direction_indexes = [[0,1,2] for _ in range(self.Nions)]
        
        N = len(timesteps)
        omega = fftfreq(N, self.potim) * 33.35640951981521 * 1E3
        
        VAF2 = np.zeros(N)
        for i,directions in zip(atom_indexes,direction_indexes):
            for j in directions:
                VAF2 += self.ase_atoms.get_masses()[i] * get_signal_correlation(velocity[timesteps,i,j])
        
        intensities = np.real(fft(VAF2))
        pdos = intensities[omega>=0]
        omega = omega[omega>=0]
        
        if smearing == 0:
            if normalize:
                return omega, pdos/np.max(pdos)
            else: 
                return omega, pdos
        else:
            x = omega
            omega = smearing / (np.max(x) - np.min(x)) * len(x)
            smeared_spectrum = scipy.ndimage.gaussian_filter1d(pdos, omega)
            if normalize:
                return x, smeared_spectrum / np.max(smeared_spectrum)
            else:
                return x, smeared_spectrum      
            
    def write_average_structure(self, file_path):
        """
        Calculates the writes the average strucure to a file in the VASP POSCAR
        format.

        Parameters
        ----------
        file_path : str
            destination file name
        """



        # Calculate mean displacements w.r.t. initial position
        displacements = np.array(self.direct_positions)
        displacements -= self.direct_positions[0]
        displacements[displacements > 0.5] -= 1.0
        displacements[displacements <-0.5] += 1.0
        displacements = np.mean(displacements,axis=0)
        
        mean_direct_positions = self.direct_positions[0] + displacements
        
        atoms = Atoms(symbols = self.ase_atoms.get_chemical_symbols(), 
                      scaled_positions = mean_direct_positions, 
                      cell = self.cell, pbc = [1,1,1])
        
        ase.io.write(file_path, atoms, format = 'vasp')
        
    def append(self, vasprun, potim = 1):
        """
        Appends the trajectory from another vasprun.xml file

        Parameters
        ----------
        vasprun : str
            Path to a vasprun.xml of the molecular dynamics run.
        potim : float
            Timestep used for molecular dynamics in fs.
        """

        # Use a class within phonopy to load the vasprun.xml file. 
        # I suspect this is not memory efficient...
        file = open(f'{vasprun}','rb')
        run = phonopy.interface.vasp.VasprunxmlExpat(file)
        print_status('Parsing vasprun xml...', end = '')
        run.parse()
        print_status('done.') 
        file.close()

        if self.potim != potim:
            print_error('Inconsistent timestep')

        print_status(f'    {len(self.direct_positions)} -> {len(self.direct_positions) + len(run.points)} steps')
        
        self.direct_positions = np.append(self.direct_positions, run.points, axis = 0)
        self.Niter = self.direct_positions.shape[0]
        self.forces = False
        
        run = None ; gc.collect(generation=2)


        
        
        
            
class DielectricModel:
    """Dielectric tensor model for a single atom displaced in a single direction
    
    Parameters
    ----------
    atom_number : int
        Number of the atom (1-indexed)
    direction : numpy.ndarray
        3D vector giving the displacement direction
    displacements : numpy.ndarray
        List of floats giving displacement magnitudes
    dielectric_tensors : numpy.ndarray
        Array of dielectric tensors corresponding to displacements
    derivative_only : bool, optional
        If true, model will be built using the 2 point interpolation method.
        If false, model will be constructed using a spline

    """


    def __init__(self, atom_number, direction, displacements, dielectric_tensors, 
                 derivative_only = False):
        """Constructor.
        """
        self.atom_index = atom_number-1
        self.atom_number = atom_number
        self.direction = np.array(direction)
        interpolation_mode = 'quadratic'
        
        # in this mode, only derivatives are calculated
        if derivative_only:
            max_dielectric = dielectric_tensors[np.argmax(displacements)]
            min_dielectric = dielectric_tensors[np.argmin(displacements)]
            derivative = (np.array(max_dielectric) - np.array(min_dielectric))/(np.max(displacements) - np.min(displacements))
            displacements = [np.min(displacements),0,np.max(displacements)]
            dielectric_tensors = [derivative * displacements[0], derivative*displacements[1], derivative * displacements[2]]
            interpolation_mode = 'linear'
        
        
        
        self.displacements = np.array(displacements)
        self.dielectric_tensors = np.array(dielectric_tensors)
        self.ref_dielectric_tensor = self.dielectric_tensors[np.argmin(np.abs(self.displacements))]
                        
        self.dielectric_tensor_interpolations = [[None,None,None],[None,None,None],[None,None,None]]
        for i in range(3):
            for j in range(3):
                self.dielectric_tensor_interpolations[i][j] = scipy.interpolate.interp1d(self.displacements, 
                                                                                         self.dielectric_tensors[:,i,j],
                                                                                         fill_value = 'extrapolate',
                                                                                         kind=interpolation_mode)
    
    def get_dielectric_tensor_time_series(self, traj):
        """Returns the dielectric tensor time series for a given trajectory. Note to avoid possible
        numerical problems when adding up contributions from a collection of sites, 
        just the change in dielectric tensor from zero displacement is returned. 

        Parameters
        ----------
        traj : Trajectory
            Molecular dynamics trajectory
        """


        # CALCULATE DISPLACEMENTS
        
        # Initially arbitrarily reference to the initial position
        displacements_3d = np.array(traj.direct_positions[:,self.atom_index,:])
        displacements_3d -= displacements_3d[0]
        displacements_3d[displacements_3d > 0.5] -= 1.0
        displacements_3d[displacements_3d <-0.5] += 1.0
        displacements_3d = displacements_3d@traj.cell
        # Re-reference to average position
        displacements_3d -= np.mean(displacements_3d,axis=0)
        
        displacements = displacements_3d.dot(self.direction)        
        
        A = np.zeros((traj.Niter, 3, 3))
        displacement_range = np.max(self.displacements) - np.min(self.displacements)
        outside_displacements = (displacements < np.min(self.displacements)) + (displacements > np.max(self.displacements))
        percentage_outside = np.sum(outside_displacements) / len(outside_displacements) * 100
        if percentage_outside > 10:
            print_warning(f'{percentage_outside}% outside interpolation, atom # {self.atom_index+1} in direction {np.round(self.direction,2)}')
        
        for x in range(3):
            for y in range(3):
                A[:,x,y] += self.dielectric_tensor_interpolations[x][y](displacements) - self.dielectric_tensor_interpolations[x][y](0)
        return A
    
    def get_rotated(self, new_atom_number, axis_name, angle_in_degrees):
        """Generates a new dielectric model by rotating around a specified
        cartesian axis

        Parameters
        ----------
        new_atom_number : int
            Atom number of new model
        axis_name : str
            cartesian axis name ('x', 'y' or 'z')
        angle_in_degrees : float
            Angle to rotate in degrees 

        Returns
        -------
        DielectricModel
            A rotated dielectric model assigned to a new atom number
        """

        rotation_matrix = scipy.spatial.transform.Rotation.from_euler(axis_name, angle_in_degrees, degrees=True).as_matrix()
        new_direction = rotation_matrix.dot(self.direction.T)
        
        dielectric_displacements = self.dielectric_tensors - self.ref_dielectric_tensor
        new_dielectric_displacements = rotation_matrix@(dielectric_displacements)@np.linalg.inv(rotation_matrix)
        new_dielectric_tensors = self.ref_dielectric_tensor + new_dielectric_displacements
        
        return(DielectricModel(new_atom_number, 
                               new_direction,
                               self.displacements, 
                               new_dielectric_tensors))
    
    def get_scaled(self, factor):
        """Generates a new dielectric model with the changes in the dielectric
        tensor scaled by a given factor

        Parameters
        ----------
        factor : float
            scaling factor (factor=1 will give no amplification)

        Returns
        -------
        DielectricModel
            A scaled dielectric model (no change in atom number) 
        """


        dielectric_displacements = self.dielectric_tensors - self.ref_dielectric_tensor
        new_dielectric_tensors = self.ref_dielectric_tensor + factor*dielectric_displacements
        
        return(DielectricModel(self.atom_number, 
                               self.direction,
                               self.displacements, 
                               new_dielectric_tensors))
    
    def get_transformed(self, new_atom_number, rotation_matrix):
        """Returns a new dielectric model for an arbitrary transformation

        Parameters
        ----------
        new_atom_number : int
            Atom number of new model
        rotation_matrix : numpy.ndarray
            3x3 transformation matrix

        Returns
        -------
        DielectricModel
            A transformed dielectric model 
        """

        new_direction = rotation_matrix.dot(self.direction.T)
        
        # Here we have sort sort of problem...
        dielectric_displacements = self.dielectric_tensors - self.ref_dielectric_tensor
        #new_dielectric_displacements = np.linalg.inv(rotation_matrix)@(dielectric_displacements)@(rotation_matrix)
        new_dielectric_displacements = rotation_matrix@(dielectric_displacements)@np.linalg.inv(rotation_matrix)
        new_dielectric_tensors = self.ref_dielectric_tensor + new_dielectric_displacements
        
        return(DielectricModel(new_atom_number, 
                               new_direction,
                               self.displacements, 
                               new_dielectric_tensors))
        
    def get_copy(self, new_atom_number):
        """Returns a copy of the dielectric model, now with a new assigned
        atom number

        Parameters
        ----------
        new_atom_number : int
            Atom number of new model

        Returns
        -------
        DielectricModel
            An identical dielectric model, this time with a newly assigned
            atom number
        """
        return(DielectricModel(new_atom_number, 
                               self.direction,
                               self.displacements, 
                               self.dielectric_tensors))
    
    def test_plot(self, axis, fixed_color = None):
        """Plots the dielectric information for testing purposes

        Parameters
        ----------
        axis : Axis
            a matplotlib axis object on which to plot
        fixed_color : None | str
            Color which to plot. None means to use matplotlib's default color
            cycling behavior

        """
        for x in range(3):
            for y in range(3):
                directions = ['x','y','z']
                label = "$a_{" + directions[x] + directions[y] + "}$"
                sc = None
                
                if fixed_color is None:
                    sc = axis.scatter(self.displacements, self.dielectric_tensors[:,x,y]-self.ref_dielectric_tensor[x,y], label = label)
                else:
                    sc = axis.scatter(self.displacements, self.dielectric_tensors[:,x,y]-self.ref_dielectric_tensor[x,y], label = label, color = fixed_color, 
                                     alpha = 0.2)
                
                color = sc.get_facecolors()[0]
                
                sample_displacements = np.linspace(np.min(self.displacements)-0.05, np.max(self.displacements) + 0.05,1000)
                axis.plot(sample_displacements, 
                          self.dielectric_tensor_interpolations[x][y](sample_displacements)-self.ref_dielectric_tensor[x][y], color = color, zorder = -10)


class SymmetricDielectricModel:
    """Dielectric model for a crystalline system that takes advantage of site 
    symmetries. 
    
    Parameters
    ----------
    ref_structure_path : str
        Path to reference structure. Any format supported by ASE should be okay, but 
        VASP's POSCAR format is recommended.
    symprec : float, optional
        Symmetry precision.
    angle_tolerance : float, optional
        Angle Tolerance

    """

    def __init__(self, ref_structure_path, symprec = 1e-5, angle_tolerance = -1.0):
        """Constructor

        """
        self.ref_structure = ase.io.read(ref_structure_path)
        
        self.lattice = self.ref_structure.cell
        self.positions = self.ref_structure.get_scaled_positions()
        self.numbers = self.ref_structure.get_atomic_numbers() 
        
        cell = (self.lattice, self.positions, self.numbers)
        
        self.symmetry = spglib.get_symmetry(cell, symprec=symprec, angle_tolerance=angle_tolerance)
        num_equivalent_atoms = len(set(self.symmetry['equivalent_atoms']))
        print(f'Symmetry analysis complete. {num_equivalent_atoms} unique atoms.')
        
        self.dielectric_models = []
            
    def add_dielectric_model(self, model):
        """Adds a dielectric model, automatically mapping the model onto all 
        symmetrically equivalent atoms.
        
        Parameters
        ----------
        model : DielectricModel
            Dielectric model to add. 
    
        """
        
        models_to_add = []
        
        basis_coord = self.ref_structure.get_scaled_positions()[model.atom_index]
        
        # Go through all transformations and fill in for all equivalent atoms
        for rot, trans in zip(self.symmetry['rotations'], self.symmetry['translations']):
            coord = np.mod(np.dot(rot, basis_coord) + trans, 1.)
            
            # Assign an atom index to coord
            displacements = coord - np.array(self.ref_structure.get_scaled_positions())
            displacements[displacements > 0.5] -= 1.0
            displacements[displacements <-0.5] += 1.0
            distances = np.sqrt(np.sum(displacements**2, axis=1))
            equivalent_index = np.argmin(distances)
            
            # Check so see if this model (and it's corresponding DOF) has been accounted for already.
            trial_model = model.get_transformed(equivalent_index+1, rot)
            
            add_model = True
            for added_model in models_to_add:
                if added_model.atom_index == trial_model.atom_index:
                    if np.dot(added_model.direction, trial_model.direction)**2 > 0.001:
                        add_model = False
                        break
            for added_model in self.dielectric_models:
                if added_model.atom_index == trial_model.atom_index:
                    if np.dot(added_model.direction, trial_model.direction)**2 > 0.001:
                        add_model = False
                        print('ERROR: Something is wrong!!!')       
            
            if add_model:
                models_to_add.append(trial_model)
                
        self.dielectric_models += models_to_add        
    
    
    # Helper function orginally used to test implementation of symmetry.
    #def test_symops(self):
    #    
    #    # We need to construct valid transformations from each atom to it's equivalent atoms
    #            
    #    elements = []
    #    scaled_positions = [] 
    #    
    #    for basis_index in set(self.symmetry['equivalent_atoms']):
    #        basis_coord = self.ref_structure.get_scaled_positions()[basis_index]
    #        
    #        for rot, trans in zip(self.symmetry['rotations'], self.symmetry['translations']):
    #            coord = np.mod(np.dot(rot, basis_coord) + trans, 1.)
    #            
    #            if len(scaled_positions) == 0:
    #                elements.append(self.ref_structure.symbols[basis_index])
    #                scaled_positions.append(coord)
    #            else:                    
    #                displacements = coord - np.array(scaled_positions)
    #                displacements[displacements > 0.5] -= 1.0
    #                displacements[displacements <-0.5] += 1.0
    #                distances = np.sqrt(np.sum(displacements**2, axis=1))
    #                
    #                closest_distance = np.min(distances)
    #                if closest_distance > 0.1:
    #                    elements.append(self.ref_structure.symbols[basis_index])
    #                    scaled_positions.append(coord)   
    #    print(elements)
    #    return(Atoms(cell = self.ref_structure.cell, 
    #                 scaled_positions = scaled_positions, 
    #                 symbols = elements, pbc = True))
    
    def print_missing_models(self):
        """Prints degrees of freedom that have no associated dielectric model"""


        def is_ortho(v1, v2):
            return np.abs(np.dot(v1,v2)) < 0.001
        
        accounted_for = []
        
        for basis_index in set(self.symmetry['equivalent_atoms']):
            element = self.ref_structure.get_chemical_symbols()[basis_index]
            directions = []
            for model in self.dielectric_models:
                if model.atom_index == basis_index:
                    directions.append(model.direction)
            
            if len(directions) == 3 and \
               is_ortho(directions[0], directions[1]) and \
               is_ortho(directions[1], directions[2]) and \
               is_ortho(directions[0], directions[2]):
                accounted_for.append(basis_index)
            else:
                print(f'Atom # {basis_index+1} ({element}) underspecified. Directions so far {np.round(directions,2)}')
            
def betf(wavenumber, T):
    """Returns bose-Einstein correction factors for Raman intensities.

    Parameters
    ----------
    wavenumber : numpy.ndarray
        Wavenumber(s) in inverse cm
    T : float
        Temperature in K

    """ 
    energy_eV = wavenumber * 29979245800 * 4.1357E-15
    kB = 8.617333262e-5
    return(1/(1-np.exp(-energy_eV/(kB*T))))


def get_total_time_series(traj,models):
    """Returns the dielectric tensor time series given a trajectory and a list
    of dielectric models.
    
    Parameters
    ----------
    traj : Trajectory
        Molecular dynamics trajectory.
    models : list
        List of DielectricModels. Of course to obtain sensible results, the atom 
        number's in these models should correspond to the appropriate atoms 
        contained in the trajectory.

    Returns
    -------
    numpy.ndarray
        An array of 3x3 dielectric matrices for each step in the trajectory.

    """

    A = np.zeros((traj.Niter, 3, 3))
    for model in models:
        A += model.get_dielectric_tensor_time_series(traj)
    return A
    
def get_raman(traj, models, smearing = 0, 
              laser_wavelength_nm = 532, intensity_correction_factors = True, 
              components = [1,1,1,1,1,1,1], 
              zero_padding = 0):
    """Calculates and returns polycrystalline Raman spectrum.  

    Parameters
    ----------
    traj : Trajectory
        Molecular dynamics trajectory
    models : list
        List of dielectric models. Of course to obtain sensible results, the atom 
        number's in these models should correspond to the appropriate atoms 
        contained in the trajectory.
    smearing : float
        Gaussian smearing width in inverse cm. Normally values between 5-10 cm-1
        are sensitive. A value of 0 turns off smearing.
    laser_wavelength : float, optional
        Wavelength of the incident light in nm. Is set to 532 nm by default.
    intensity_correction_factors : bool, optional
        If true, applies intensity correction factors.
    components : list, optional
        Turns on and off the various components in the sum of intensities. 
        Examination of individual components (or using them to generate a 
        stacked line plot) allows assignment of symmetries to the various peaks.
        All components are included by default, giving the full polycrystalline
        spectrum.
    zero_padding : int, optional
        Amount of zero padding, which may or may not improve the signal. I'm
        not so sure... A value of 0 turns off zero padding (recommended). Zero 
        padding is off by default. 
    """


    laser_wavelength_invcm = 10000000/laser_wavelength_nm
    
    A = get_total_time_series(traj, models)
                
    Aprime = np.diff(A, axis=0)
    Aprime = np.append(Aprime,np.zeros((zero_padding,3,3)), axis=0)
                
    wavenumbers, _ = get_signal_spectrum(Aprime[:,0,0], traj.potim)        
    
    alpha2 = components[0] * 1/9* get_signal_spectrum(Aprime[:,0,0] + Aprime[:,1,1] + Aprime[:,2,2], traj.potim)[1]
    gamma2 = components[1] * 1/2*get_signal_spectrum(Aprime[:,0,0] - Aprime[:,1,1], traj.potim)[1] + \
             components[2] * 1/2*get_signal_spectrum(Aprime[:,1,1] - Aprime[:,2,2], traj.potim)[1] + \
             components[3] * 1/2*get_signal_spectrum(Aprime[:,2,2] - Aprime[:,0,0], traj.potim)[1] + \
             components[4] * 3 * get_signal_spectrum(Aprime[:,0,1], traj.potim)[1] + \
             components[5] * 3 * get_signal_spectrum(Aprime[:,1,2], traj.potim)[1] + \
             components[6] * 3 * get_signal_spectrum(Aprime[:,0,2], traj.potim)[1]
        
    intensities = (45.0*alpha2 + 7.0*gamma2)
    if intensity_correction_factors:
        intensities *= betf(wavenumbers,300)*((wavenumbers-laser_wavelength_invcm)/10000)**4/wavenumbers
    
    # Remove 0 cm-1, as it will be infinite
    intensities = intensities[1:]
    wavenumbers = wavenumbers[1:]
    
    if smearing == 0:
        return wavenumbers, intensities
    else:
        x = np.linspace(0, 2000, 2000)
        smeared_spectrum = np.zeros(len(x))
        for wavenumber, intensity in zip(wavenumbers, intensities):
            smeared_spectrum += intensity * np.exp( - (wavenumber - x)**2 / (2*smearing**2) )
        return x, smeared_spectrum 

import os
import shutil

def setup_dielectric_displacement(root_dir, atom_number, direction, displacements):
    """Helps set up dielectric displacement calculations for a specific atom
    in a specific direction. This method sets up
    directories and build the necessary structures. However, it expects a very 
    specific directory structure, so use carefully. 
    

    Parameters
    ----------
    root_dir : str
        Root directory for the dielectric calculations.
    atom_number : int
        Atom number (1-indexed) of the atom 
    direction : str
        Direction of displacement. 'x', 'y', or 'z'. One can also use
        a vector, but this is advanced usage. 
    displacements : list
        List of floats specifying the displacements to probe. Zero displacements
        can be included, but these will be ignored. 
    """

    atom_index = atom_number-1
    ref_structure = ase.io.read(f'{root_dir}/ref/POSCAR')
    element = ref_structure.get_chemical_symbols()[atom_index]
    
    vector = None
    directory_name = None
    old_style = False
    if isinstance(direction, str):
        vector = {'x' : (1,0,0), 'y' : (0,1,0), 'z' : (0,0,1)}[direction]
        directory_name = direction
        old_style = True
    else:
        vector = direction/ np.sqrt(np.dot(direction,direction))
        directory_name = np.array2string(np.round(vector,5), separator=',')[1:-1]
        directory_name = directory_name.replace('-','m').replace(' ', '')

    for displacement in displacements:
        if np.abs(displacement - 0) < 0.0001:
            continue
        print('Creating directories')
        if old_style:
            leaf_name = f'{displacement}{directory_name}'.replace('-', 'm')
        else:
            leaf_name = f'{displacement}'.replace('-', 'm')

        disp_dir = f'{root_dir}/{element}{atom_number}/{directory_name}/{leaf_name}'
        os.makedirs(disp_dir)
        new_structure = ref_structure.copy()
        new_structure.positions[atom_index] += np.array(vector)*displacement
        ase.io.write(f'{disp_dir}/POSCAR', new_structure)

def read_dielectric_tensor(directory):
    """Reads the dielectric tensor from a 'dielectric_tensor' file.
    
    Parameters
    ----------
    directory : str
        Directory containing the 'dielectric_tensor' file.

    """
    dielectric = []
    with open(f'{directory}/dielectric_tensor') as file:
        file.readline()
        file.readline()
        for _ in range(3):
            line = [float(num) for num in file.readline().strip().split()]
            dielectric.append(line)
    
    return np.array(dielectric)

def get_dielectric_model(root_dir, atom_number, direction, derivative_only = False):
    """Reads a dielectric tensor model. This function expects a specific
    directory structure, which can be generated either manually or with the
    setup_dielectric_displacement function.
    
    Parameters
    ----------
    root_dir : str
        Root directory for the dielectric calculations.
    atom_number : int
        Atom number (1-indexed) of the atom 
    direction : str
        Direction of displacement. 'x', 'y', or 'z'. One can also use
        a vector, but this is advanced usage. 
    derivative_only : bool, optional
        If true, the resulting model will be fully first-order. 

    """


    atom_index = atom_number-1
    ref_dielectric = read_dielectric_tensor(f'{root_dir}/ref')
    ref_structure = ase.io.read(f'{root_dir}/ref/POSCAR')
    element = ref_structure.get_chemical_symbols()[atom_index]
    
    vector = None
    old_style = False
    if direction in ['x', 'y', 'z']:
        vector = {'x' : (1,0,0), 'y' : (0,1,0), 'z' : (0,0,1)}[direction]
        old_style = True
    else:
        vector = np.fromstring(direction.replace('m','-'), sep=',')
    
    disp_dirs = next(os.walk(f'{root_dir}/{element}{atom_number}/{direction}'))[1]
    
    displacements = [0]
    dielectrics = [ref_dielectric]
    for disp_dir in disp_dirs:
        if old_style:
            displacements.append(float(disp_dir[:-1].replace('m','-')))
        else:
            displacements.append(float(disp_dir.replace('m','-')))
        dielectrics.append(read_dielectric_tensor(f'{root_dir}/{element}{atom_number}/{direction}/{disp_dir}'))
    
    dielectrics = np.array(dielectrics)
    dielectrics = dielectrics[np.argsort(displacements)]
    displacements = np.sort(displacements)
    
    return(DielectricModel(atom_number, vector, displacements, dielectrics, derivative_only))

                        