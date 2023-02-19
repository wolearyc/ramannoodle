import numpy as np
import scipy
import matplotlib
import ase.io
import phonopy
import ase
import matplotlib
from ase import Atoms

from scipy.fftpack import fft, ifft, fftfreq, fftshift
import scipy.signal




def get_signal_correlation(signal):
    '''
    Returns in the positive autocorrelation of signal
    :type signal: ndarray
    :rtype: ndarray
    '''
    new_signal = signal - np.mean(signal)
    AF = scipy.signal.correlate(new_signal, new_signal, 'full')
    AF = AF[(len(AF)-1)//2:] # no normalization
    return AF

def get_signal_spectra(signal, potim = 1):
    '''
    Returns in the (positive frequency) Fourier transform of the autocorrelation 
    of a signal.
    :type signal: ndarray
    :param potim: Signals sampling rate in fs (typically the molecular dynamics
    timestep).
    :return: A tuple of wavenumbers (in inverse cm) and intensities
    '''
    timesteps = range(len(signal)-1)
    N = len(timesteps)
    AF = get_signal_correlation(signal)
    omega = fftfreq(AF.size, potim) * 33.35640951981521 * 1E3 # for cm-1
    intensities = np.real(fft(AF))
    wavenumbers = omega[omega>=0]
    intensities = intensities[omega>=0]
    return(wavenumbers, intensities)


# Boltzmann Constant in [eV/K]
kb = 8.617332478E-5
# electron volt in [Joule]
ev = 1.60217733E-19
# Avogadro's Constant
Navogadro = 6.0221412927E23


class Trajectory:
    '''
    The Trajectory object stores a molecular dynamics trajectory output by VASP. 
    :param vasprun: Path to a vasprun.xml of the molecular dynamics run.
    :type vasprun: str
    :param potim: Timestep used for molecular dynamics in fs.
    :type potim: float
    :param load_forces: Specifies whether to load forces from the vasprun.xml file. Forces are NOT required for Raman calculations, and loading them will increase the memory footprint substantially.
    :type load_forces: bool
    '''

    def __init__(self, vasprun, potim = 1, load_forces = False):
        
        # Use a class within phonopy to load the vasprun.xml file. 
        # I suspect this is not memory efficient...
        file = open(f'{vasprun}','rb')
        run = phonopy.interface.vasp.VasprunxmlExpat(file)
        print(f'Parse...', end=''); run.parse() ; print('done. ', end='') 
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
        
    
    
    # WARNING: this is not correctly implemented
    def getPhononDos(self, unit='cm-1', timesteps = 'all', atom_indexes = 'all', 
                     direction_indexes = 'all', smearing = 0, normalize = True):
        
        velocity = np.diff(self.direct_positions, axis=0)
        # apply periodic boundary condition
        velocity[velocity > 0.5] -= 1.0
        velocity[velocity <-0.5] += 1.0
        # Velocity in Angstrom per femtosecond
        for i in range(self.Niter-1):
            velocity[i,:,:] = np.dot(velocity[i,:,:], self.cell) / self.potim
        
        
        """ Phonon DOS from VAF """
        if timesteps == 'all':
            timesteps = range(len(velocity))
        if atom_indexes == 'all':
            atom_indexes = range(self.Nions)
        if direction_indexes == 'all':
            direction_indexes = [[0,1,2] for _ in range(self.Nions)]
        
        N = len(timesteps)
        omega = fftfreq(N, self.potim)
        # Frequency in THz
        if unit.lower() == 'thz':
            omega *= 1E3
        # Frequency in cm^-1
        if unit.lower() == 'cm-1':
            omega *= 33.35640951981521 * 1E3
        # Frequency in mev
        if unit.lower() == 'mev':
            omega *= 4.13567 * 1E3
        
        VAF2 = np.zeros(N)
        for i,directions in zip(atom_indexes,direction_indexes):
            for j in directions:
                VAF2 += get_signal_correlation(velocity[timesteps,i,j])
        
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
        
        
            
class DielectricModel:
    
    def __init__(self, atom_index, direction, displacements, dielectric_tensors, 
                 derivative_only = False):
        self.atom_index = atom_index
        self.direction = np.array(direction)
        interpolation_mode = 'quadratic'
        
        # in this mode, only derivatives are calculated
        if derivative_only:
            max_dielectric = dielectric_tensors[np.argmax(displacements)]
            min_dielectric = dielectric_tensors[np.argmin(displacements)]
            derivative = (np.array(max_dielectric) - np.array(min_dielectric))/(np.max(displacements) - np.min(displacements))
            displacements = [-0.2,0,0.2]
            dielectric_tensors = [derivative * -0.2, derivative*0, derivative * 0.2]
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
    
    # Returns the time series of the dielectric tensor according to the interpolated model
    def get_dielectric_tensor_time_series(self, md_data):
        
        # CALCULATE DISPLACEMENTS
        
        # Initially arbitrarily reference to the initial position
        displacements_3d = np.array(md_data.direct_positions[:,self.atom_index,:])
        displacements_3d -= displacements_3d[0]
        displacements_3d[displacements_3d > 0.5] -= 1.0
        displacements_3d[displacements_3d <-0.5] += 1.0
        displacements_3d = displacements_3d@md_data.cell
        # Re-reference to average position
        displacements_3d -= np.mean(displacements_3d,axis=0)
        
        displacements = displacements_3d.dot(self.direction)        
        
        A = np.zeros((md_data.Niter, 3, 3))
        displacement_range = np.max(self.displacements) - np.min(self.displacements)
        outside_displacements = (displacements < np.min(self.displacements)) + (displacements > np.max(self.displacements))
        percentage_outside = np.sum(outside_displacements) / len(outside_displacements) * 100
        if percentage_outside > 10:
            print(f'Warning: {percentage_outside}% outside interpolation, atom # {self.atom_index+1} in direction {np.round(self.direction,2)}')
        
        for x in range(3):
            for y in range(3):
                A[:,x,y] += self.dielectric_tensor_interpolations[x][y](displacements)
        return A
    
    def get_rotated(self, new_atom_index, axis_name, angle_in_degrees):
        '''Angle is in degrees. Will hopefully rotate the displacements and produce a new DielectricDisplacement object'''
        rotation_matrix = scipy.spatial.transform.Rotation.from_euler(axis_name, angle_in_degrees, degrees=True).as_matrix()
        new_direction = rotation_matrix.dot(self.direction.T)
        
        dielectric_displacements = self.dielectric_tensors - self.ref_dielectric_tensor
        new_dielectric_displacements = rotation_matrix@(dielectric_displacements)@np.linalg.inv(rotation_matrix)
        new_dielectric_tensors = self.ref_dielectric_tensor + new_dielectric_displacements
        
        return(DielectricModel(new_atom_index, 
                               new_direction,
                               self.displacements, 
                               new_dielectric_tensors))
    
    def get_amplified(self, factor):
        dielectric_displacements = self.dielectric_tensors - self.ref_dielectric_tensor
        new_dielectric_tensors = self.ref_dielectric_tensor + factor*dielectric_displacements
        
        return(DielectricModel(self.atom_index, 
                               self.direction,
                               self.displacements, 
                               new_dielectric_tensors))
    
    def get_transformed(self, new_atom_index, rotation_matrix):
        new_direction = rotation_matrix.dot(self.direction.T)
        
        # Here we have sort sort of problem...
        dielectric_displacements = self.dielectric_tensors - self.ref_dielectric_tensor
        #new_dielectric_displacements = np.linalg.inv(rotation_matrix)@(dielectric_displacements)@(rotation_matrix)
        new_dielectric_displacements = rotation_matrix@(dielectric_displacements)@np.linalg.inv(rotation_matrix)
        new_dielectric_tensors = self.ref_dielectric_tensor + new_dielectric_displacements
        
        return(DielectricModel(new_atom_index, 
                               new_direction,
                               self.displacements, 
                               new_dielectric_tensors))
        
    def get_copy(self, new_atom_index):
        return(DielectricModel(new_atom_index, 
                               self.direction,
                               self.displacements, 
                               self.dielectric_tensors))
    
    def test_plot(self, axes, fixed_color = None):
        for x in range(3):
            for y in range(3):
                directions = ['x','y','z']
                label = "$a_{" + directions[x] + directions[y] + "}$"
                sc = None
                
                if fixed_color is None:
                    sc = axes.scatter(self.displacements, self.dielectric_tensors[:,x,y]-self.ref_dielectric_tensor[x,y], label = label)
                else:
                    sc = axes.scatter(self.displacements, self.dielectric_tensors[:,x,y]-self.ref_dielectric_tensor[x,y], label = label, color = fixed_color, 
                                     alpha = 0.2)
                
                color = sc.get_facecolors()[0]
                
                sample_displacements = np.linspace(np.min(self.displacements)-0.05, np.max(self.displacements) + 0.05,1000)
                axes.plot(sample_displacements, 
                          self.dielectric_tensor_interpolations[x][y](sample_displacements)-self.ref_dielectric_tensor[x][y], color = color, zorder = -10)
                
                
# implemented directly using the spglib library, since the ase symmetry routines are 
# seem to be bogus

import spglib

class SymmetricDielectricModel:
    
    def __init__(self, ref_structure, symprec = 1e-5, angle_tolerance = -1.0):
        self.ref_structure = ref_structure
        
        self.lattice = ref_structure.cell
        self.positions = ref_structure.get_scaled_positions()
        self.numbers = ref_structure.get_atomic_numbers() 
        
        cell = (self.lattice, self.positions, self.numbers)
        
        self.symmetry = spglib.get_symmetry(cell, symprec=symprec, angle_tolerance=angle_tolerance)
        num_equivalent_atoms = len(set(self.symmetry['equivalent_atoms']))
        print(f'Symmetry analysis complete. {num_equivalent_atoms} unique atoms.')
        
        self.dielectric_models = []
            
    def add_dielectric_model(self, model):
        # For a given atom index, this function should add the model and 
        # figure out which additional models to fill in based on crystal symmetry. 
        # An error should be thrown if something fishy is going on! Keep in mind that
        # the identify matrix is a possible rotation, so model will be added
        # properly
        
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
            trial_model = model.get_transformed(equivalent_index, rot)
            
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
    
    
    
    def test_symops(self):
        
        # We need to construct valid transformations from each atom to it's equivalent atoms
                
        elements = []
        scaled_positions = [] 
        
        for basis_index in set(self.symmetry['equivalent_atoms']):
            basis_coord = self.ref_structure.get_scaled_positions()[basis_index]
            
            for rot, trans in zip(self.symmetry['rotations'], self.symmetry['translations']):
                coord = np.mod(np.dot(rot, basis_coord) + trans, 1.)
                
                if len(scaled_positions) == 0:
                    elements.append(self.ref_structure.symbols[basis_index])
                    scaled_positions.append(coord)
                else:                    
                    displacements = coord - np.array(scaled_positions)
                    displacements[displacements > 0.5] -= 1.0
                    displacements[displacements <-0.5] += 1.0
                    distances = np.sqrt(np.sum(displacements**2, axis=1))
                    
                    closest_distance = np.min(distances)
                    if closest_distance > 0.1:
                        elements.append(self.ref_structure.symbols[basis_index])
                        scaled_positions.append(coord)   
        print(elements)
        return(Atoms(cell = self.ref_structure.cell, 
                     scaled_positions = scaled_positions, 
                     symbols = elements, pbc = True))
    
    def print_missing_models(self):
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
    energy_eV = wavenumber * 29979245800 * 4.1357E-15
    kB = 8.617333262e-5
    return(1/(1-np.exp(-energy_eV/(kB*T))))

betf(10,300)

def get_total_time_series(md_data,models):
    A = np.zeros((md_data.Niter, 3, 3))
    for model in models:
        A += model.get_dielectric_tensor_time_series(md_data)
    return A
    
def get_raman(md_data, models, smearing = 0, component = 'all', laser_wavelength_nm = 532, 
             intensity_correction_factors = True, components = [1,1,1,1,1,1,1]):
    
    laser_wavelength_invcm = 10000000/laser_wavelength_nm
    
    A = get_total_time_series(md_data, models)
                
    Aprime = np.diff(A, axis=0)
    print(np.mean(Aprime,axis=0))
                
    wavenumbers, _ = get_signal_spectra(Aprime[:,0,0])        
    
    alpha2 = components[0] * 1/9* get_signal_spectra(Aprime[:,0,0] + Aprime[:,1,1] + Aprime[:,2,2])[1]
    gamma2 = components[1] * 1/2*get_signal_spectra(Aprime[:,0,0] - Aprime[:,1,1])[1] + \
             components[2] * 1/2*get_signal_spectra(Aprime[:,1,1] - Aprime[:,2,2])[1] + \
             components[3] * 1/2*get_signal_spectra(Aprime[:,2,2] - Aprime[:,0,0])[1] + \
             components[4] * 3 * get_signal_spectra(Aprime[:,0,1])[1] + \
             components[5] * 3 * get_signal_spectra(Aprime[:,1,2])[1] + \
             components[6] * 3 * get_signal_spectra(Aprime[:,0,2])[1]
        
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
    dielectric = []
    with open(f'{directory}/dielectric_tensor') as file:
        file.readline()
        file.readline()
        for _ in range(3):
            line = [float(num) for num in file.readline().strip().split()]
            dielectric.append(line)
    
    return np.array(dielectric)

def get_dielectric_model(root_dir, atom_number, directory_name, derivative_only = False):

    atom_index = atom_number-1
    ref_dielectric = read_dielectric_tensor(f'{root_dir}/ref')
    ref_structure = ase.io.read(f'{root_dir}/ref/POSCAR')
    element = ref_structure.get_chemical_symbols()[atom_index]
    
    vector = None
    old_style = False
    if directory_name in ['x', 'y', 'z']:
        vector = {'x' : (1,0,0), 'y' : (0,1,0), 'z' : (0,0,1)}[directory_name]
        old_style = True
    else:
        vector = np.fromstring(directory_name.replace('m','-'), sep=',')
    
    disp_dirs = next(os.walk(f'{root_dir}/{element}{atom_number}/{directory_name}'))[1]
    
    displacements = [0]
    dielectrics = [ref_dielectric]
    for disp_dir in disp_dirs:
        if old_style:
            displacements.append(float(disp_dir[:-1].replace('m','-')))
        else:
            displacements.append(float(disp_dir.replace('m','-')))
        dielectrics.append(read_dielectric_tensor(f'{root_dir}/{element}{atom_number}/{directory_name}/{disp_dir}'))
    
    dielectrics = np.array(dielectrics)
    dielectrics = dielectrics[np.argsort(displacements)]
    displacements = np.sort(displacements)
    
    return(DielectricModel(atom_index, vector, displacements, dielectrics, derivative_only))
                       

                
            




                        