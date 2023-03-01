from scipy.optimize import minimize
from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import ElectronicStructureDriverType, ElectronicStructureMoleculeDriver
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper, BravyiKitaevMapper
from qiskit.opflow.converters import TwoQubitReduction
from qiskit_nature.algorithms.ground_state_solvers.minimum_eigensolver_factories import NumPyMinimumEigensolverFactory
from qiskit_nature.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit_nature.circuit.library.initial_states.hartree_fock import *
from qiskit.opflow import *
from qiskit import Aer
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, SPSA, SLSQP
from qiskit.algorithms import VQE
from qiskit_nature.circuit.library.ansatzes import UCC
from qiskit.utils import *
from qiskit_nature.circuit.library.ansatzes.utils import generate_fermionic_excitations
import time
import pickle
import copy 
import numpy as np 






#In the fillowing, the convension is as follows: 
#Letters i,j,.. correspond to occupied alpha spin orbitals.
#Letters I,J,.. correspond to occupied beta spin orbitals.
#Letters a,b,.. correspond to unoccupied alpha spin orbitals.
#Letters A,B,.. correspond to unoccupied alpha beta orbitals.

#Defining a funciton that performs spin adaptation with input as a list of excitations to be included in the cluster operator.
def perform_spin_adaptation(relevant_excitations):
  global t_i_a, t_I_A, t_ij_ab, t_IJ_AB, t_iJ_aB, t_iJ_Ab, t_iJ_aA, t_Ij_Ab, t_Ij_aB, t_Ij_aA, t_iI_aB, t_iI_Ab, t_iI_aA
  global spin_nonadapted_exct
  global nonadapted_t_i_a_idx, nonadapted_t_I_A_idx, nonadapted_t_ij_ab_idx, nonadapted_t_IJ_AB_idx, nonadapted_t_iJ_aB_idx, nonadapted_t_Ij_Ab_idx, nonadapted_t_iJ_Ab_idx, nonadapted_t_Ij_aB_idx, nonadapted_t_iJ_aA_idx, nonadapted_t_Ij_aA_idx, nonadapted_t_iI_aB_idx, nonadapted_t_iI_Ab_idx, nonadapted_t_iI_aA_idx
  global spin_adapted_exct
  global adapted_t_i_a_idx, adapted_t_ij_ab_idx, adapted_t_iJ_aB_idx, adapted_t_iJ_aA_idx, adapted_t_iI_aB_idx, adapted_t_iI_aA_idx
  global nonadapted_exct_index_extractor
  global adapted_exct_index_extractor
  global adapted_to_nonadapted_idx


  singles = [i for i in relevant_excitations if len(i[0])==1] #extracting the single excitations
  doubles = [i for i in relevant_excitations if len(i[0])==2] #extracting the double excitations


  t_i_a = []
  for i in singles:
      if i[0][0]<nso:
          t_i_a.append(i)

  t_I_A = []
  for i in t_i_a:
      t_I_A.append(((i[0][0]+nso,), (i[1][0]+nso,)))




  t_ij_ab = []
  for i in doubles:
      if i[0][0]<nso and i[0][1]<nso:
          t_ij_ab.append(i)
  t_IJ_AB = []
  for i in t_ij_ab:
      t_IJ_AB.append(((i[0][0]+nso, i[0][1]+nso), (i[1][0]+nso, i[1][1]+nso)))
      
      

  t_iJ_aB = []
  for i in t_ij_ab:
      x = ((i[0][0], i[0][1]+nso), (i[1][0], i[1][1]+nso))
      if x in doubles:
          t_iJ_aB.append(x)
  for i in doubles:
      if i[0][0]<i[0][1]-nso and i[1][0]<i[1][1]-nso and i not in t_iJ_aB:
          t_iJ_aB.append(i)
        
      
  t_iJ_Ab = []
  for i in t_ij_ab:
      x = ((i[0][0], i[0][1]+nso), (i[1][1], i[1][0]+nso))
      if x in doubles:
          t_iJ_Ab.append(x)
  for i in t_iJ_aB:
      x = (i[0], (i[1][1]-nso, i[1][0]+nso))
      if x in doubles and x not in t_iJ_Ab:
          t_iJ_Ab.append(x)

      
      
  t_iJ_aA = []
  for i in doubles:
      if i[0][0]<i[0][1]-nso and i[1][0]==i[1][1]-nso:
          t_iJ_aA.append(i)




  t_Ij_Ab = []
  for i in t_iJ_aB:
      t_Ij_Ab.append( ((i[0][1]-nso, i[0][0]+nso), (i[1][1]-nso, i[1][0]+nso)) )
          
  t_Ij_aB = []
  for i in t_iJ_Ab:
      t_Ij_aB.append(((i[0][1]-nso, i[0][0]+nso), (i[1][1]-nso, i[1][0]+nso)))

  t_Ij_aA = []
  for i in t_iJ_aA:
      t_Ij_aA.append(((i[0][1]-nso, i[0][0]+nso), i[1]))




  t_iI_aB = []
  for i in doubles:
      if i[0][0]==i[0][1]-nso and i[1][0]<i[1][1]-nso:
          t_iI_aB.append(i)

  t_iI_Ab = []
  for i in t_iI_aB:
      t_iI_Ab.append((i[0], (i[1][1]-nso, i[1][0]+nso)))
          
  t_iI_aA = []
  for i in doubles:
      if i[0][0]==i[0][1]-nso and i[1][0]==i[1][1]-nso:
          t_iI_aA.append(i)


  #List of excitations without applying spin adaptation, arranged in a specific order.
  spin_nonadapted_exct = t_i_a + t_I_A + t_ij_ab + t_IJ_AB + t_iJ_aB + t_Ij_Ab + t_iJ_Ab + t_Ij_aB + t_iJ_aA + t_Ij_aA + t_iI_aB + t_iI_Ab + t_iI_aA

  #Given an excitation, the following function returns all the indices of the excitations in the above list that are equivalent to it according to spin adaptation.
  def nonadapted_exct_index_extractor (t_pq_rs):
      return [i for i in range(len(spin_nonadapted_exct)) if spin_nonadapted_exct[i] in t_pq_rs]
  nonadapted_t_i_a_idx = nonadapted_exct_index_extractor(t_i_a)
  nonadapted_t_I_A_idx = nonadapted_exct_index_extractor(t_I_A)
  nonadapted_t_ij_ab_idx = nonadapted_exct_index_extractor(t_ij_ab)
  nonadapted_t_IJ_AB_idx = nonadapted_exct_index_extractor(t_IJ_AB)
  nonadapted_t_iJ_aB_idx = nonadapted_exct_index_extractor(t_iJ_aB)
  nonadapted_t_Ij_Ab_idx = nonadapted_exct_index_extractor(t_Ij_Ab)
  nonadapted_t_iJ_Ab_idx = nonadapted_exct_index_extractor(t_iJ_Ab)
  nonadapted_t_Ij_aB_idx = nonadapted_exct_index_extractor(t_Ij_aB)
  nonadapted_t_iJ_aA_idx = nonadapted_exct_index_extractor(t_iJ_aA)
  nonadapted_t_Ij_aA_idx = nonadapted_exct_index_extractor(t_Ij_aA)
  nonadapted_t_iI_aB_idx = nonadapted_exct_index_extractor(t_iI_aB)
  nonadapted_t_iI_Ab_idx = nonadapted_exct_index_extractor(t_iI_Ab)
  nonadapted_t_iI_aA_idx = nonadapted_exct_index_extractor(t_iI_aA)


  #List of excitations after applying spin adaptation, arranged in a specific order.
  spin_adapted_exct = t_i_a + t_ij_ab + t_iJ_aB + t_iJ_aA + t_iI_aB + t_iI_aA

  #Given an excitation, the following function returns all the indices of the excitations in the above list that are equivalent to it according to spin adaptation.
  def adapted_exct_index_extractor (t_pq_rs):
      return [i for i in range(len(spin_adapted_exct)) if spin_adapted_exct[i] in t_pq_rs]
  adapted_t_i_a_idx = adapted_exct_index_extractor(t_i_a)
  adapted_t_ij_ab_idx = adapted_exct_index_extractor(t_ij_ab)
  adapted_t_iJ_aB_idx = adapted_exct_index_extractor(t_iJ_aB)
  adapted_t_iJ_aA_idx = adapted_exct_index_extractor(t_iJ_aA)
  adapted_t_iI_aB_idx = adapted_exct_index_extractor(t_iI_aB)
  adapted_t_iI_aA_idx = adapted_exct_index_extractor(t_iI_aA)

  #Given an excitation from the list "spin_adapted_exct", the following function returns all the excitations that are equivalent to it present in the the list "spin_nonadapted_exct".
  def adapted_to_nonadapted_idx (T_pq_rs):
      try:
          idx = adapted_t_i_a_idx.index(T_pq_rs)
          return [nonadapted_t_i_a_idx[idx], nonadapted_t_I_A_idx[idx]]
      except ValueError:
          pass
      
      try:
          idx = adapted_t_ij_ab_idx.index(T_pq_rs)
          return [nonadapted_t_ij_ab_idx[idx], nonadapted_t_IJ_AB_idx[idx]]
      except ValueError:
          pass
      
      try:
          idx = adapted_t_iJ_aB_idx.index(T_pq_rs)
          return [nonadapted_t_iJ_aB_idx[idx], nonadapted_t_Ij_Ab_idx[idx]]
      except ValueError:
          pass
      
      try:
          idx = adapted_t_iJ_aA_idx.index(T_pq_rs)
          return [nonadapted_t_iJ_aA_idx[idx], nonadapted_t_Ij_aA_idx[idx]]
      except ValueError:
          pass

      try:
          idx = adapted_t_iI_aB_idx.index(T_pq_rs)
          return [nonadapted_t_iI_aB_idx[idx], nonadapted_t_iI_Ab_idx[idx]]
      except ValueError:
          pass
      
      try:
          idx = adapted_t_iI_aA_idx.index(T_pq_rs)
          return [nonadapted_t_iI_aA_idx[idx]]
      except ValueError:
          pass

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#Given a set of spin-adapted parameters (in the order of the list "spin_adapted_exct"), the following function gives the corresponding expectation of the Hamiltonian.
def adapted_energy_eval_func(adapted_params):
    global nonadapted_params
    nonadapted_params = np.zeros_like(t)
    for i,j in zip([adapted_t_i_a_idx, adapted_t_ij_ab_idx, adapted_t_iJ_aB_idx, adapted_t_iJ_aA_idx, adapted_t_iI_aB_idx, adapted_t_iI_aA_idx], [nonadapted_t_i_a_idx, nonadapted_t_ij_ab_idx, nonadapted_t_iJ_aB_idx, nonadapted_t_iJ_aA_idx, nonadapted_t_iI_aB_idx, nonadapted_t_iI_aA_idx]):
        nonadapted_params[j] = adapted_params[i]
    nonadapted_params[nonadapted_t_I_A_idx] = adapted_params[adapted_t_i_a_idx]
    nonadapted_params[nonadapted_t_IJ_AB_idx] = adapted_params[adapted_t_ij_ab_idx]
    nonadapted_params[nonadapted_t_Ij_Ab_idx] = adapted_params[adapted_t_iJ_aB_idx]
    if len(nonadapted_t_iJ_Ab_idx) == len(nonadapted_t_ij_ab_idx) and len(nonadapted_t_iJ_Ab_idx) == len(nonadapted_t_iJ_aB_idx):
        nonadapted_params[nonadapted_t_iJ_Ab_idx] = adapted_params[adapted_t_ij_ab_idx] + adapted_params[adapted_t_iJ_aB_idx]    
    else:
        for i in nonadapted_t_iJ_Ab_idx:
            p = t_iJ_Ab[i-min(nonadapted_t_iJ_Ab_idx)]
            t1 = ((p[0][0], p[0][1]-nso), (p[1][1]-nso, p[1][0]))
            try:
                idx1 = t_ij_ab.index(t1)
                nonadapted_params[i] += adapted_params[adapted_t_ij_ab_idx[idx1]]
            except ValueError:
                pass
            t2 = ((p[0][0], p[0][1]), (p[1][1]-nso, p[1][0]+nso))
            try:
                idx2 = t_iJ_aB.index(t2)
                nonadapted_params[i] += adapted_params[adapted_t_iJ_aB_idx[idx2]]
            except ValueError:
                pass
    nonadapted_params[nonadapted_t_Ij_aB_idx] = nonadapted_params[nonadapted_t_iJ_Ab_idx]
    nonadapted_params[nonadapted_t_Ij_aA_idx] = adapted_params[adapted_t_iJ_aA_idx]
    nonadapted_params[nonadapted_t_iI_Ab_idx] = adapted_params[adapted_t_iI_aB_idx]

    a = energy_eval_func(nonadapted_params)
    return a


#Given the number of maximum iterations allowed, the following function performs the Spin Adapted VQE.
def spin_adapted_VQE(maxiter):
  global adapted_ordered_params
  global x_0

  t0 = 0
  nonadapted_params = np.zeros_like(t)       #Will be used later
  print ('Energy at the initial parameter values: ', adapted_energy_eval_func(x_0) + result_exact.nuclear_repulsion_energy)
  
  iter_count = 0
  #Callback function for minimization


  def print_fun(adapted_params):
      global adapted_ordered_params
      adapted_ordered_params = np.concatenate((adapted_ordered_params, adapted_params.reshape(1,len(spin_adapted_exct))))
      global iter_count
      iter_count += 1
      print (iter_count)
      print (adapted_energy_eval_func(adapted_params) + result_exact.nuclear_repulsion_energy, '(time taken = %f)'%(time.perf_counter() - t0))
      print(np.flip(np.argsort(np.abs(adapted_params))))

  t0 = time.perf_counter()
  minimization_result = minimize(fun=adapted_energy_eval_func, x0=x_0, method='L-BFGS-B', tol = 1e-8, options={'maxiter': maxiter}, callback = print_fun)
#   minimization_result = minimize_parallel(fun=adapted_energy_eval_func, x0=x_0, tol = 1e-8, options={'maxiter': maxiter}, callback = print_fun)
  t1 = time.perf_counter()

  adapted_ordered_params = np.delete(adapted_ordered_params, (0), axis=0)

  UCC_E = adapted_energy_eval_func(adapted_ordered_params[-1,:]) + result_exact.nuclear_repulsion_energy
  print ('Final energy after minimization: ', UCC_E)  
  print ('Time ellapsed: ', t1-t0, '\n')

  return adapted_ordered_params

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#Given the number of maximum iterations allowed, the following function performs the Non Spin Adapted VQE.
def nonspin_adapted_VQE(maxiter):
  global x_0
  print ('Energy at the initial parameter values: ', energy_eval_func(x_0) + result_exact.nuclear_repulsion_energy)
  UCC_E = 0

  global ordered_params

  iter_count = 0
  #Callback function for minimization
  def print_fun(x):
      global ordered_params
      global iter_count
      iter_count += 1
      print (iter_count)
      ordered_params = np.concatenate((ordered_params, x.reshape(1,len(t))))
      energy = energy_eval_func(x)
      print (energy + result_exact.nuclear_repulsion_energy, '(time taken = %f)'%(time.perf_counter() - t0))
      print(np.flip(np.argsort(np.abs(x))))

  t0 = time.perf_counter()
  minimization_result = minimize(fun=energy_eval_func, x0=x_0, method='L-BFGS-B', tol = 1e-8, options={'maxiter': maxiter}, callback = print_fun)
#   minimization_result = minimize_parallel(fun=energy_eval_func, x0=x_0, tol = 1e-8, options={'maxiter': maxiter}, callback = print_fun)
  t1 = time.perf_counter()

  ordered_params = np.delete(ordered_params, (0), axis=0)

  UCC_E = energy_eval_func(ordered_params[-1,:]) + result_exact.nuclear_repulsion_energy
  print ('Final energy after minimization: ', UCC_E)  
  print ('Time ellapsed: ', t1-t0, '\n')
  return ordered_params

        
        
        
        
        
        
        
        
        
        
#To start from an intermediate step
try:
    with open('Final data/Pre ML Data/H20/Cases_Completed.pkl', 'rb') as infile:
        Cases_Completed = pickle.load(infile)
except FileNotFoundError:
    Cases_Completed = []

try:
    with open('Final data/Pre ML Data/H20/Bond_Lengths.pkl', 'rb') as infile:
        Bond_Lengths = pickle.load(infile)
    with open('Final data/Pre ML Data/H20/H20_True_Energies.pkl', 'rb') as infile:
        H4_Linear_True_Energies = pickle.load(infile)
    H4_Linear_True_Energies = H4_Linear_True_Energies[:len(Bond_Lengths)]    
except FileNotFoundError:
    Bond_Lengths = []
    H4_Linear_True_Energies = []    
        
        
        

    


try:
    with open('Final data/Pre ML Data/H20/H20_SA_0_ZAF_0_Params.pkl', 'rb') as infile:
        H4_Linear_SA_0_ZAF_0_Params = pickle.load(infile)
    with open('Final data/Pre ML Data/H20/H20_SA_0_ZAF_0_Energies.pkl', 'rb') as infile:
        H4_Linear_SA_0_ZAF_0_Energies = pickle.load(infile)
    with open('Final data/Pre ML Data/H20/H20_SA_0_ZAF_0_Time.pkl', 'rb') as infile:
        H4_Linear_SA_0_ZAF_0_Time = pickle.load(infile) 
    min_len = min([len(H4_Linear_SA_0_ZAF_0_Params), len(H4_Linear_SA_0_ZAF_0_Energies), len(H4_Linear_SA_0_ZAF_0_Time)])
    H4_Linear_SA_0_ZAF_0_Params = H4_Linear_SA_0_ZAF_0_Params[:min_len]
    H4_Linear_SA_0_ZAF_0_Energies = H4_Linear_SA_0_ZAF_0_Energies[:min_len]
    H4_Linear_SA_0_ZAF_0_Time = H4_Linear_SA_0_ZAF_0_Time[:min_len]         
except FileNotFoundError:
    H4_Linear_SA_0_ZAF_0_Params = []
    H4_Linear_SA_0_ZAF_0_Energies = []
    H4_Linear_SA_0_ZAF_0_Time = []


    

try:
    with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_0_Params.pkl', 'rb') as infile:
        H4_Linear_SA_1_ZAF_0_Params = pickle.load(infile)
    with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_0_Energies.pkl', 'rb') as infile:
        H4_Linear_SA_1_ZAF_0_Energies = pickle.load(infile)
    with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_0_Time.pkl', 'rb') as infile:
        H4_Linear_SA_1_ZAF_0_Time = pickle.load(infile)
    min_len = min([len(H4_Linear_SA_1_ZAF_0_Params), len(H4_Linear_SA_1_ZAF_0_Energies), len(H4_Linear_SA_1_ZAF_0_Time)])
    H4_Linear_SA_1_ZAF_0_Params = H4_Linear_SA_1_ZAF_0_Params[:min_len]
    H4_Linear_SA_1_ZAF_0_Energies = H4_Linear_SA_1_ZAF_0_Energies[:min_len]
    H4_Linear_SA_1_ZAF_0_Time = H4_Linear_SA_1_ZAF_0_Time[:min_len]  
except FileNotFoundError:
    H4_Linear_SA_1_ZAF_0_Params = []
    H4_Linear_SA_1_ZAF_0_Energies = []
    H4_Linear_SA_1_ZAF_0_Time = []
    

    
try:
    with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_1_Params.pkl', 'rb') as infile:
        H4_Linear_SA_1_ZAF_1_Params = pickle.load(infile)
    with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_1_Energies.pkl', 'rb') as infile:
        H4_Linear_SA_1_ZAF_1_Energies = pickle.load(infile)
    with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_1_Time.pkl', 'rb') as infile:
        H4_Linear_SA_1_ZAF_1_Time = pickle.load(infile)
    with open('Final data/Pre ML Data/H20/H20_spin_adapted_Removable_Idx.pkl', 'rb') as infile:
        H4_Linear_spin_adapted_Removable_Idx = pickle.load(infile)
    with open('Final data/Pre ML Data/H20/H20_nonspin_adapted_Removable_Idx.pkl', 'rb') as infile:
        H4_Linear_nonspin_adapted_Removable_Idx = pickle.load(infile)       
    min_len = min([len(H4_Linear_SA_1_ZAF_1_Params), len(H4_Linear_SA_1_ZAF_1_Energies), len(H4_Linear_SA_1_ZAF_1_Time), len(H4_Linear_spin_adapted_Removable_Idx), len(H4_Linear_nonspin_adapted_Removable_Idx)])
    H4_Linear_SA_1_ZAF_1_Params = H4_Linear_SA_1_ZAF_1_Params[:min_len]
    H4_Linear_SA_1_ZAF_1_Energies = H4_Linear_SA_1_ZAF_1_Energies[:min_len]
    H4_Linear_SA_1_ZAF_1_Time = H4_Linear_SA_1_ZAF_1_Time[:min_len]
    H4_Linear_spin_adapted_Removable_Idx = H4_Linear_spin_adapted_Removable_Idx[:min_len]
    H4_Linear_nonspin_adapted_Removable_Idx = H4_Linear_nonspin_adapted_Removable_Idx[:min_len]
except FileNotFoundError:
    H4_Linear_SA_1_ZAF_1_Params = []
    H4_Linear_SA_1_ZAF_1_Energies = []
    H4_Linear_SA_1_ZAF_1_Time = []
    H4_Linear_spin_adapted_Removable_Idx = []
    H4_Linear_nonspin_adapted_Removable_Idx = []


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
# Performing VQE for 3 cases namely, General UCCSD, Spin adapted UCCSD, Zero amplitude filtered Spin adapted UCCSD

# Cases_Completed = []
# Bond_Lengths = []

# H4_Linear_True_Energies = []

# H4_Linear_SA_0_ZAF_0_Params = []
# H4_Linear_SA_0_ZAF_0_Energies = []
# H4_Linear_SA_0_ZAF_0_Time = []

# H4_Linear_SA_1_ZAF_0_Params = []
# H4_Linear_SA_1_ZAF_0_Energies = []
# H4_Linear_SA_1_ZAF_0_Time = []

# H4_Linear_SA_1_ZAF_1_Params = []
# H4_Linear_SA_1_ZAF_1_Energies = []
# H4_Linear_SA_1_ZAF_1_Time = []
# H4_Linear_spin_adapted_Removable_Idx = []
# H4_Linear_nonspin_adapted_Removable_Idx = []

# H4_Linear_UCCSD_Removable_Idx = []











#Selecting the backend
backend = Aer.get_backend('aer_simulator_statevector')
optimizer = L_BFGS_B(maxiter=100)

#Thw code performs three different cases: 1)Both Spin adaptation and ZAF, 2) Only Spin adaptation and 3) Direct VQE with UCCSD
All_cases = [('True', 'True'), ('True', 'False'), ('False', 'False')]
for case_idx in range (len(All_cases)):
    if len(Cases_Completed) == 0:
        case = All_cases[case_idx]
    else:
        case_idx = Cases_Completed[-1]+1
        case = All_cases[case_idx]
        
    print ('_______________________________Starting case: %i_______________________________'%(case_idx+1))
    include_spin_adaptation = case[0]
    include_zero_filteration = case[1]

    #To select a molecule from below, bring it outside of the comment. "left" and "right" set the range of bond length to be considered.
    '''
    # H4 linear
    left = 0.5
    right = 3.5
    No_of_points = 30

    #H4_Ring
    radius = 0.5*np.sqrt(2)
    angles = np.linspace(np.pi/12, np.pi/12*5, 30)

    # H6 linear
    left = 0.5
    right = 3.2
    No_of_points = 30

    # LiH
    left = 0.55
    right = 4.0
    No_of_points = 30

    # H20
    left = 0.65
    right = 2.85
    No_of_points = 30

    # BeH2
    left = 0.75
    right = 3.75
    No_of_points = 30
    '''


    # H20
    left = 0.65
    right = 2.85
    No_of_points = 30

    R = np.linspace(left, right, No_of_points)              #The Bond length for the molecule with number of points = "No_of_points"


    #Running the VQE for various bond lengths

    # for r_idx in range (len(angles)):                     #Only for H4 Ring as its parametrization is different from other molecules.
    #   theta = angles[r_idx]                               #Only for H4 Ring
    #   r = radius*np.cos(theta)*2                          #Only for H4 Ring
    for r_idx in range (len(R)):
      if len(Bond_Lengths) == 0:
          r = R[r_idx]
      else:
          r_idx = len(Bond_Lengths)
          r = R[r_idx]


            
      print ('_______________________Bond Length = %f    and    iteration number = %i out of %i_______________________' %(r, r_idx+1, len(R)))
      print ('Building molecule... \n')
      
      #Building the molecule
      molecule = Molecule(
          # coordinates are given in Angstrom
          geometry=[
              # ["Be", [0.0, 0.0, 0.0]],
              # ["H", [0.0, 0.0, -r]],
    #           ["H", [0.0, 0.0, r]],
    #           ["Li", [0.0, 0.0, 0.0]],
              ["O", [0.0, 0.0, 0.0]],
              ["H", [np.cos(0.911498202)*r,  np.sin(0.911498202)*r, 0.0]],
              ["H", [np.cos(0.911498202)*r, -np.sin(0.911498202)*r, 0.0]],
              # ["H", [0.0, radius*np.cos(theta), radius*np.sin(theta)]],
              # ["H", [0.0, radius*np.cos(theta), -radius*np.sin(theta)]],
              # ["H", [0.0, -radius*np.cos(theta), radius*np.sin(theta)]],
              # ["H", [0.0, -radius*np.cos(theta), -radius*np.sin(theta)]],
              # ["H", [0.0, 0.0, 1.0]],
#               ["H", [0.0, 0.0, 0.0]],
#               ["H", [0.0, 0.0, 0.0+r]],
#               ["H", [0.0, 0.0, 0.0+2*r]],
#               ["H", [0.0, 0.0, 0.0+3*r]],
#               ["H", [0.0, 0.0, 0.0+4*r]],
#               ["H", [0.0, 0.0, 0.0+5*r]]
          ],
          multiplicity=1,  # = 2*spin + 1
          charge=0,
      )
      driver = ElectronicStructureMoleculeDriver(
          molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYSCF
      )

      problem = ElectronicStructureProblem(driver)
      second_q_op = problem.second_q_ops()

      qubit_converter = QubitConverter(mapper = JordanWignerMapper(), two_qubit_reduction = True)
      Hamiltonian = qubit_converter.convert(second_q_op[0], num_particles=problem.num_particles)

      nso = int(problem.num_spin_orbitals/2)        #Number of SPATIAL orbitals
      nelec = problem.num_particles                 #Number of alpha and beta electrons in a tuple format

      print ('Building molecule completed! \n')












      print ('Calculating true ground state energy... \n')

      #Calculating the exact energy
      def exact_diagonalizer(problem, qubit_converter):
          solver = NumPyMinimumEigensolverFactory()
          calc = GroundStateEigensolver(qubit_converter, solver)
          result = calc.solve(problem)
          return result

      result_exact = exact_diagonalizer(problem, qubit_converter)
      exact_energy = np.real(result_exact.eigenenergies[0])
      # print('Exact solution: \n', result_exact)

      H4_Linear_True_Energies.append(result_exact.groundenergy + result_exact.nuclear_repulsion_energy)

      print ('Calculating true ground state energy completed! The ground state energy = %f \n'%(result_exact.groundenergy + result_exact.nuclear_repulsion_energy))

      H20_True_Energies = copy.copy(H4_Linear_True_Energies)
      if case_idx == 0:
          with open('Final data/Pre ML Data/H20/H20_True_Energies.pkl', 'wb') as outfile:
              pickle.dump(H20_True_Energies, outfile, pickle.HIGHEST_PROTOCOL)




      HF = HartreeFock(num_spin_orbitals=nso*2, num_particles=nelec, qubit_converter=QubitConverter(mapper=JordanWignerMapper(), two_qubit_reduction=True))







      #Performing the UCC-VQE based on the case chosen above

      iter_count = 0

      temp_UCC_op = UCC(num_spin_orbitals=nso*2, num_particles=nelec, excitations=[1, 2], qubit_converter=QubitConverter(mapper=JordanWignerMapper(), two_qubit_reduction=True), initial_state = HF)
      all_excitations = temp_UCC_op.excitation_list
      t = all_excitations

      # Performing Spin adapted VQE
      if include_spin_adaptation == 'True':
        print ('__________________Spin Adaptation has been applied__________________ \n')

        perform_spin_adaptation(all_excitations)

        def custom_excitation_list(num_spin_orbitals, num_particles):
          return spin_nonadapted_exct
        UCC_operator = UCC(num_spin_orbitals=nso*2, num_particles=nelec, excitations=custom_excitation_list, qubit_converter=QubitConverter(mapper=JordanWignerMapper(), two_qubit_reduction=True), initial_state = HF)
        t = UCC_operator.parameters

        # Spin adapted | zero-amplitude filtered VQE
        if include_zero_filteration == 'True':
          algorithm = VQE(UCC_operator, optimizer=optimizer, quantum_instance=backend)
          energy_eval_func = algorithm.get_energy_evaluation(Hamiltonian, False)
          x_0 = np.zeros(len(spin_adapted_exct))
          adapted_ordered_params = np.zeros((1, len(spin_adapted_exct)))

          adapted_ordered_params = spin_adapted_VQE(2)

          removable_idx_1 = []
          j = np.flip(np.argsort(np.abs(adapted_ordered_params[-1, :])))
          for i in j:
              if np.abs(adapted_ordered_params[1,i] - adapted_ordered_params[0,i]) < 1e-4 and np.abs(adapted_ordered_params[1,i]) < 1e-5:
                  # plt.plot(np.arange(len(adapted_ordered_params[:,0])), adapted_ordered_params[:,i], 'r-o')
                  removable_idx_1.append(i)
          nonadapted_removable_idx = []
          for i in removable_idx_1:
                  nonadapted_removable_idx = nonadapted_removable_idx + adapted_to_nonadapted_idx(i)

          for i in range (len(adapted_t_ij_ab_idx)):
              try:
                  a = removable_idx_1.index(adapted_t_ij_ab_idx[i]) 
                  if adapted_t_iJ_aB_idx[i] in removable_idx_1:
                      nonadapted_removable_idx = nonadapted_removable_idx + [nonadapted_t_iJ_Ab_idx[i], nonadapted_t_Ij_aB_idx[i]]
              except ValueError:
                  pass

          #Storing the removable indices
          H4_Linear_spin_adapted_Removable_Idx.append(removable_idx_1)
          H4_Linear_nonspin_adapted_Removable_Idx.append(nonadapted_removable_idx)

          relevant_excitations = [spin_nonadapted_exct[i] for i in range (len(spin_nonadapted_exct)) if i not in nonadapted_removable_idx]

          perform_spin_adaptation(relevant_excitations)

          def custom_excitation_list(num_spin_orbitals, num_particles):
            return spin_nonadapted_exct
          UCC_operator = UCC(num_spin_orbitals=nso*2, num_particles=nelec, excitations=custom_excitation_list, qubit_converter=QubitConverter(mapper=JordanWignerMapper(), two_qubit_reduction=True), initial_state = HF)
          t = UCC_operator.parameters
          algorithm = VQE(UCC_operator, optimizer=optimizer, quantum_instance=backend)
          energy_eval_func = algorithm.get_energy_evaluation(Hamiltonian, False)

          adapted_ordered_params = np.zeros((1,len(spin_adapted_exct)))

          print ('\n __________________Zero-amplitude filteration has been applied__________________ \n')

          x_0 = np.zeros((len(spin_adapted_exct)))
          t0 = time.perf_counter()
          adapted_ordered_params = spin_adapted_VQE(35)
          t1 = time.perf_counter()

          H4_Linear_SA_1_ZAF_1_Params.append(adapted_ordered_params)
          H4_Linear_SA_1_ZAF_1_Energies.append(adapted_energy_eval_func(adapted_ordered_params[-1, :]) + result_exact.nuclear_repulsion_energy)
          H4_Linear_SA_1_ZAF_1_Time.append(t1-t0)

            
            
          H20_SA_1_ZAF_1_Params = copy.copy(H4_Linear_SA_1_ZAF_1_Params)
          H20_SA_1_ZAF_1_Energies = copy.copy(H4_Linear_SA_1_ZAF_1_Energies)
          H20_SA_1_ZAF_1_Time = copy.copy(H4_Linear_SA_1_ZAF_1_Time)
          H20_spin_adapted_Removable_Idx = copy.copy(H4_Linear_spin_adapted_Removable_Idx)
          H20_nonspin_adapted_Removable_Idx = copy.copy(H4_Linear_nonspin_adapted_Removable_Idx)  
            
            
            
          with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_1_Params.pkl', 'wb') as outfile:
                pickle.dump(H20_SA_1_ZAF_1_Params, outfile, pickle.HIGHEST_PROTOCOL)
          with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_1_Params.pkl', 'rb') as infile:
                H20_SA_1_ZAF_1_Params = pickle.load(infile)

          with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_1_Energies.pkl', 'wb') as outfile:
                pickle.dump(H20_SA_1_ZAF_1_Energies, outfile, pickle.HIGHEST_PROTOCOL)
          with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_1_Energies.pkl', 'rb') as infile:
                H20_SA_1_ZAF_1_Energies = pickle.load(infile)

          with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_1_Time.pkl', 'wb') as outfile:
                pickle.dump(H20_SA_1_ZAF_1_Time, outfile, pickle.HIGHEST_PROTOCOL)
          with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_1_Time.pkl', 'rb') as infile:
                H20_SA_1_ZAF_1_Time = pickle.load(infile)

          with open('Final data/Pre ML Data/H20/H20_spin_adapted_Removable_Idx.pkl', 'wb') as outfile:
                pickle.dump(H20_spin_adapted_Removable_Idx, outfile, pickle.HIGHEST_PROTOCOL)
          with open('Final data/Pre ML Data/H20/H20_spin_adapted_Removable_Idx.pkl', 'rb') as infile:
                H20_spin_adapted_Removable_Idx = pickle.load(infile)

          with open('Final data/Pre ML Data/H20/H20_nonspin_adapted_Removable_Idx.pkl', 'wb') as outfile:
                pickle.dump(H20_nonspin_adapted_Removable_Idx, outfile, pickle.HIGHEST_PROTOCOL)
          with open('Final data/Pre ML Data/H20/H20_nonspin_adapted_Removable_Idx.pkl', 'rb') as infile:
                H20_nonspin_adapted_Removable_Idx = pickle.load(infile)
            
            
            
            
            
            
            
            
            
            
            
            
            
        # Performing Spin adapted | non zero-amplitude filtered VQE
        else:
          print ('\n __________________Zero-amplitude filteration has NOT been applied__________________ \n')
          algorithm = VQE(UCC_operator, optimizer=optimizer, quantum_instance=backend)
          energy_eval_func = algorithm.get_energy_evaluation(Hamiltonian, False)

          adapted_ordered_params = np.zeros((1,len(spin_adapted_exct)))
          x_0 = np.zeros(len(spin_adapted_exct))
          t0 = time.perf_counter()
          adapted_ordered_params = spin_adapted_VQE(35)
          t1 = time.perf_counter()

          H4_Linear_SA_1_ZAF_0_Params.append(adapted_ordered_params)
          H4_Linear_SA_1_ZAF_0_Energies.append(adapted_energy_eval_func(adapted_ordered_params[-1, :]) + result_exact.nuclear_repulsion_energy)
          H4_Linear_SA_1_ZAF_0_Time.append(t1-t0)
        
        
        
          H20_SA_1_ZAF_0_Params = copy.copy(H4_Linear_SA_1_ZAF_0_Params)
          H20_SA_1_ZAF_0_Energies = copy.copy(H4_Linear_SA_1_ZAF_0_Energies)
          H20_SA_1_ZAF_0_Time = copy.copy(H4_Linear_SA_1_ZAF_0_Time)


        
          with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_0_Params.pkl', 'wb') as outfile:
              pickle.dump(H20_SA_1_ZAF_0_Params, outfile, pickle.HIGHEST_PROTOCOL)
          with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_0_Params.pkl', 'rb') as infile:
              H20_SA_1_ZAF_0_Params = pickle.load(infile)

          with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_0_Energies.pkl', 'wb') as outfile:
              pickle.dump(H20_SA_1_ZAF_0_Energies, outfile, pickle.HIGHEST_PROTOCOL)
          with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_0_Energies.pkl', 'rb') as infile:
              H20_SA_1_ZAF_0_Energies = pickle.load(infile)

          with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_0_Time.pkl', 'wb') as outfile:
              pickle.dump(H20_SA_1_ZAF_0_Time, outfile, pickle.HIGHEST_PROTOCOL)
          with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_0_Time.pkl', 'rb') as infile:
              H20_SA_1_ZAF_0_Time = pickle.load(infile)
        
        
        
        
        
        
        
        
        
      # Performing nonspin adapted | zero amplitude filtered VQE
      else:
        print ('__________________Spin adaptation has NOT been applied__________________ \n')

        #Just to have a consistent order of excitations
        perform_spin_adaptation(all_excitations)

        def custom_excitation_list(num_spin_orbitals, num_particles):
          return spin_nonadapted_exct
        UCC_operator = UCC(num_spin_orbitals=nso*2, num_particles=nelec, excitations=custom_excitation_list, qubit_converter=QubitConverter(mapper=JordanWignerMapper(), two_qubit_reduction=True), initial_state = HF)
        t = UCC_operator.parameters

        # Nonspin adapted | zero-amplitude filtered VQE
        if include_zero_filteration == 'True':

          algorithm = VQE(UCC_operator, optimizer=optimizer, quantum_instance=backend)
          energy_eval_func = algorithm.get_energy_evaluation(Hamiltonian, False)

          x_0 = np.zeros(len(t))
          ordered_params = nonspin_adapted_VQE(2)

          removable_idx = []
          j = np.arange(len(ordered_params[-1, :]))
          for i in j:
            if np.abs(ordered_params[1,i] - ordered_params[0,i]) < 1e-4 and np.abs(ordered_params[1,i]) < 1e-5:
              # plt.plot(np.arange(len(ordered_params[:,0])), ordered_params[:,i], 'r-o')
              removable_idx.append(i)

          #Storing the removable indices
          H4_Linear_UCCSD_Removable_Idx.append(removable_idx)

          def custom_excitation_list(num_spin_orbitals, num_particles):
            removed_excitations = [all_excitations[i] for i in range (len(all_excitations)) if i not in removable_idx]
            return removed_excitations

          UCC_operator = UCC(num_spin_orbitals=nso*2, num_particles=nelec, excitations=custom_excitation_list, qubit_converter=QubitConverter(mapper=JordanWignerMapper(), two_qubit_reduction=True), initial_state = HF)
          t = UCC_operator.parameters
          x_0 = np.zeros(len(t))
          algorithm = VQE(UCC_operator, optimizer=optimizer, quantum_instance=backend)
          energy_eval_func = algorithm.get_energy_evaluation(Hamiltonian, False)

          print ('\n __________________Zero-amplitude filteration has been applied__________________ \n')
          ordered_params = np.zeros((1,len(t)))
          ordered_params = nonspin_adapted_VQE(35)

        # Performing Nonspin adapted | non zero-amplitude filtered VQE
        else:
          print ('__________________Zero-amplitude filteration has NOT been applied__________________ \n')

          algorithm = VQE(UCC_operator, optimizer=optimizer, quantum_instance=backend)
          energy_eval_func = algorithm.get_energy_evaluation(Hamiltonian, False)
          ordered_params = np.zeros((1,len(t)))
          x_0 = np.zeros(len(t))

          t0 = time.perf_counter()
          ordered_params = nonspin_adapted_VQE(35)
          t1 = time.perf_counter()

          H4_Linear_SA_0_ZAF_0_Params.append(ordered_params)
          H4_Linear_SA_0_ZAF_0_Energies.append(energy_eval_func(ordered_params[-1, :]) + result_exact.nuclear_repulsion_energy)
          H4_Linear_SA_0_ZAF_0_Time.append(t1-t0)

        
        
          H20_SA_0_ZAF_0_Params = copy.copy(H4_Linear_SA_0_ZAF_0_Params)
          H20_SA_0_ZAF_0_Energies = copy.copy(H4_Linear_SA_0_ZAF_0_Energies)
          H20_SA_0_ZAF_0_Time = copy.copy(H4_Linear_SA_0_ZAF_0_Time)
        
        
          with open('Final data/Pre ML Data/H20/H20_SA_0_ZAF_0_Params.pkl', 'wb') as outfile:
              pickle.dump(H20_SA_0_ZAF_0_Params, outfile, pickle.HIGHEST_PROTOCOL)
          with open('Final data/Pre ML Data/H20/H20_SA_0_ZAF_0_Params.pkl', 'rb') as infile:
              H20_SA_0_ZAF_0_Params = pickle.load(infile)

          with open('Final data/Pre ML Data/H20/H20_SA_0_ZAF_0_Energies.pkl', 'wb') as outfile:
              pickle.dump(H20_SA_0_ZAF_0_Energies, outfile, pickle.HIGHEST_PROTOCOL)
          with open('Final data/Pre ML Data/H20/H20_SA_0_ZAF_0_Energies.pkl', 'rb') as infile:
              H20_SA_0_ZAF_0_Energies = pickle.load(infile)

          with open('Final data/Pre ML Data/H20/H20_SA_0_ZAF_0_Time.pkl', 'wb') as outfile:
              pickle.dump(H20_SA_0_ZAF_0_Time, outfile, pickle.HIGHEST_PROTOCOL)
          with open('Final data/Pre ML Data/H20/H20_SA_0_ZAF_0_Time.pkl', 'rb') as infile:
              H20_SA_0_ZAF_0_Time = pickle.load(infile)
        
        
        
        
        
      Bond_Lengths.append(r)
      with open('Final data/Pre ML Data/H20/Bond_Lengths.pkl', 'wb') as outfile:
            pickle.dump(Bond_Lengths, outfile, pickle.HIGHEST_PROTOCOL)
      if len(Bond_Lengths) == len(R):
        break
    Cases_Completed.append(case_idx)
    with open('Final data/Pre ML Data/H20/Cases_Completed.pkl', 'wb') as outfile:
        pickle.dump(Cases_Completed, outfile, pickle.HIGHEST_PROTOCOL)
        
    Bond_Lengths = []
    with open('Final data/Pre ML Data/H20/Bond_Lengths.pkl', 'wb') as outfile:
        pickle.dump(Bond_Lengths, outfile, pickle.HIGHEST_PROTOCOL)
    
    print ('_______________________________Case  %i completed_______________________________'%(case_idx+1))
    if len(Cases_Completed) == len(All_cases):
        break
















#Saving the results:





# H20_True_Energies = copy.copy(H4_Linear_True_Energies)

# H20_SA_0_ZAF_0_Params = copy.copy(H4_Linear_SA_0_ZAF_0_Params)
# H20_SA_0_ZAF_0_Energies = copy.copy(H4_Linear_SA_0_ZAF_0_Energies)
# H20_SA_0_ZAF_0_Time = copy.copy(H4_Linear_SA_0_ZAF_0_Time)

# H20_SA_1_ZAF_0_Params = copy.copy(H4_Linear_SA_1_ZAF_0_Params)
# H20_SA_1_ZAF_0_Energies = copy.copy(H4_Linear_SA_1_ZAF_0_Energies)
# H20_SA_1_ZAF_0_Time = copy.copy(H4_Linear_SA_1_ZAF_0_Time)

# H20_SA_1_ZAF_1_Params = copy.copy(H4_Linear_SA_1_ZAF_1_Params)
# H20_SA_1_ZAF_1_Energies = copy.copy(H4_Linear_SA_1_ZAF_1_Energies)
# H20_SA_1_ZAF_1_Time = copy.copy(H4_Linear_SA_1_ZAF_1_Time)
# H20_spin_adapted_Removable_Idx = copy.copy(H4_Linear_spin_adapted_Removable_Idx)
# H20_nonspin_adapted_Removable_Idx = copy.copy(H4_Linear_nonspin_adapted_Removable_Idx)

# # H20_UCCSD_Removable_Idx = copy.copy(H4_Linear_UCCSD_Removable_Idx)







# with open('Final data/Pre ML Data/H20/H20_True_Energies.pkl', 'wb') as outfile:
#     pickle.dump(H20_True_Energies, outfile, pickle.HIGHEST_PROTOCOL)
# with open('Final data/Pre ML Data/H20/H20_True_Energies.pkl', 'rb') as infile:
#     H20_True_Energies = pickle.load(infile)






# with open('Final data/Pre ML Data/H20/H20_SA_0_ZAF_0_Params.pkl', 'wb') as outfile:
#     pickle.dump(H20_SA_0_ZAF_0_Params, outfile, pickle.HIGHEST_PROTOCOL)
# with open('Final data/Pre ML Data/H20/H20_SA_0_ZAF_0_Params.pkl', 'rb') as infile:
#     H20_SA_0_ZAF_0_Params = pickle.load(infile)

# with open('Final data/Pre ML Data/H20/H20_SA_0_ZAF_0_Energies.pkl', 'wb') as outfile:
#     pickle.dump(H20_SA_0_ZAF_0_Energies, outfile, pickle.HIGHEST_PROTOCOL)
# with open('Final data/Pre ML Data/H20/H20_SA_0_ZAF_0_Energies.pkl', 'rb') as infile:
#     H20_SA_0_ZAF_0_Energies = pickle.load(infile)

# with open('Final data/Pre ML Data/H20/H20_SA_0_ZAF_0_Time.pkl', 'wb') as outfile:
#     pickle.dump(H20_SA_0_ZAF_0_Time, outfile, pickle.HIGHEST_PROTOCOL)
# with open('Final data/Pre ML Data/H20/H20_SA_0_ZAF_0_Time.pkl', 'rb') as infile:
#     H20_SA_0_ZAF_0_Time = pickle.load(infile)






# with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_0_Params.pkl', 'wb') as outfile:
#     pickle.dump(H20_SA_1_ZAF_0_Params, outfile, pickle.HIGHEST_PROTOCOL)
# with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_0_Params.pkl', 'rb') as infile:
#     H20_SA_1_ZAF_0_Params = pickle.load(infile)

# with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_0_Energies.pkl', 'wb') as outfile:
#     pickle.dump(H20_SA_1_ZAF_0_Energies, outfile, pickle.HIGHEST_PROTOCOL)
# with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_0_Energies.pkl', 'rb') as infile:
#     H20_SA_1_ZAF_0_Energies = pickle.load(infile)

# with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_0_Time.pkl', 'wb') as outfile:
#     pickle.dump(H20_SA_1_ZAF_0_Time, outfile, pickle.HIGHEST_PROTOCOL)
# with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_0_Time.pkl', 'rb') as infile:
#     H20_SA_1_ZAF_0_Time = pickle.load(infile)






# with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_1_Params.pkl', 'wb') as outfile:
#     pickle.dump(H20_SA_1_ZAF_1_Params, outfile, pickle.HIGHEST_PROTOCOL)
# with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_1_Params.pkl', 'rb') as infile:
#     H20_SA_1_ZAF_1_Params = pickle.load(infile)

# with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_1_Energies.pkl', 'wb') as outfile:
#     pickle.dump(H20_SA_1_ZAF_1_Energies, outfile, pickle.HIGHEST_PROTOCOL)
# with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_1_Energies.pkl', 'rb') as infile:
#     H20_SA_1_ZAF_1_Energies = pickle.load(infile)

# with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_1_Time.pkl', 'wb') as outfile:
#     pickle.dump(H20_SA_1_ZAF_1_Time, outfile, pickle.HIGHEST_PROTOCOL)
# with open('Final data/Pre ML Data/H20/H20_SA_1_ZAF_1_Time.pkl', 'rb') as infile:
#     H20_SA_1_ZAF_1_Time = pickle.load(infile)

# with open('Final data/Pre ML Data/H20/H20_spin_adapted_Removable_Idx.pkl', 'wb') as outfile:
#     pickle.dump(H20_spin_adapted_Removable_Idx, outfile, pickle.HIGHEST_PROTOCOL)
# with open('Final data/Pre ML Data/H20/H20_spin_adapted_Removable_Idx.pkl', 'rb') as infile:
#     H20_spin_adapted_Removable_Idx = pickle.load(infile)

# with open('Final data/Pre ML Data/H20/H20_nonspin_adapted_Removable_Idx.pkl', 'wb') as outfile:
#     pickle.dump(H20_nonspin_adapted_Removable_Idx, outfile, pickle.HIGHEST_PROTOCOL)
# with open('Final data/Pre ML Data/H20/H20_nonspin_adapted_Removable_Idx.pkl', 'rb') as infile:
#     H20_nonspin_adapted_Removable_Idx = pickle.load(infile)











# with open('Final data/Pre ML Data/H20/H20_UCCSD_Removable_Idx.pkl', 'wb') as outfile:
    # pickle.dump(H20_UCCSD_Removable_Idx, outfile, pickle.HIGHEST_PROTOCOL)
# with open('Final data/Pre ML Data/H20/H20_UCCSD_Removable_Idx.pkl', 'rb') as infile:
    # H20_UCCSD_Removable_Idx = pickle.load(infile)
    
    
    
print ('\n \n \n The code has been executed successfully!')