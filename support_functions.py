#IMPORTS
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.visualization import plot_histogram
from qiskit_textbook.tools import array_to_latex
import qiskit
import datetime
import json
import random
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from scipy.special import gamma
import qiskit.providers.aer.noise as noise
from qiskit import *
from qiskit.extensions import XGate, UnitaryGate
from qiskit.compiler import transpile
from qiskit.transpiler import CouplingMap
from qiskit.providers.aer.noise import (
    NoiseModel, 
    errors, 
    QuantumError, 
    ReadoutError, 
    pauli_error, 
    depolarizing_error, 
    thermal_relaxation_error
)

#FUNCTIONS

def load_noise_thermal():
    """
    This function loads a standard thermal noise model corresponding to IBM's Perth machine.
    It manually adds a readout error and a cx error to match the performance of the Qiskit's noise model of Perth 
    and the true machine.
    """
    #Values from ibm_perth
    # T1 and T2 values (T2 <= 2 x T1)
    T1s = t1 = 151320
    T2s = t2 = 133700

    # Instruction times (in nanoseconds)
    time_id = 35
    time_sx = 140
    time_rz = 140
    time_x = 140
    time_cx = 444
    time_reset = 800  # 1 microsecond
    time_measure = 675

    
    # QuantumError objects
    errors_reset = thermal_relaxation_error(t1, t2, time_reset)
    errors_measure = thermal_relaxation_error(t1, t2, time_measure)
    errors_id  = thermal_relaxation_error(t1, t2, time_id)
    errors_rz  = thermal_relaxation_error(t1, t2, time_rz)
    errors_sx  = thermal_relaxation_error(t1, t2, time_sx)
    errors_x  = thermal_relaxation_error(t1, t2, time_x)
    errors_cx = thermal_relaxation_error(t1, t2, time_cx).expand(thermal_relaxation_error(t1, t2, time_cx))

    # Add errors to noise model
    noise_thermal = NoiseModel() 
    noise_thermal.add_all_qubit_quantum_error(errors_reset, "reset")
    noise_thermal.add_all_qubit_quantum_error(errors_measure, "measure")
    noise_thermal.add_all_qubit_quantum_error(errors_id, "id")
    noise_thermal.add_all_qubit_quantum_error(errors_rz, "rz")
    noise_thermal.add_all_qubit_quantum_error(errors_sx, "sz")
    noise_thermal.add_all_qubit_quantum_error(errors_cx, "cx")
    noise_thermal.add_all_qubit_quantum_error(errors_x, "x")
    #"""
    #Readout error
    prob = 0.010
    error_readout = pauli_error([('X', prob),('I', 1-prob)])
    noise_thermal.add_all_qubit_quantum_error(error_readout, 'measure')

    #CX-error
    prob1 = 0.006
    prob2 = 0.008
    errors_cx = pauli_error([('X', prob1),('I', 1-prob1)]).expand(pauli_error([('X', prob2),('I', 1-prob2)]))
    noise_thermal.add_all_qubit_quantum_error(errors_cx, "cx")
    #"""
    return noise_thermal

def load_noise_future():
    """
    This is a supposed future noise model of IBM's Perth.
    It has longer coherence times and less added errors.
    """
    #Values from ibm_perth
    # T1 and T2 values (T2 <= 2 x T1)
    T1s = t1 = 151320*10
    T2s = t2 = 133700*10

    # Instruction times (in nanoseconds)
    time_id = 35
    time_sx = 140
    time_rz = 140
    time_x = 140
    time_cx = 444
    time_reset = 800  # 1 microsecond
    time_measure = 675

    # QuantumError objects
    errors_reset = thermal_relaxation_error(t1, t2, time_reset)
    errors_measure = thermal_relaxation_error(t1, t2, time_measure)
    errors_id  = thermal_relaxation_error(t1, t2, time_id)
    errors_rz  = thermal_relaxation_error(t1, t2, time_rz)
    errors_sx  = thermal_relaxation_error(t1, t2, time_sx)
    errors_x  = thermal_relaxation_error(t1, t2, time_x)
    errors_cx = thermal_relaxation_error(t1, t2, time_cx).expand(thermal_relaxation_error(t1, t2, time_cx))

    # Add errors to noise model
    noise_future = NoiseModel()
    noise_future.add_all_qubit_quantum_error(errors_reset, "reset")
    noise_future.add_all_qubit_quantum_error(errors_measure, "measure")
    noise_future.add_all_qubit_quantum_error(errors_id, "id")
    noise_future.add_all_qubit_quantum_error(errors_rz, "rz")
    noise_future.add_all_qubit_quantum_error(errors_sx, "sz")
    noise_future.add_all_qubit_quantum_error(errors_cx, "cx")
    noise_future.add_all_qubit_quantum_error(errors_x, "x")
    #"""
    #Readout error
    prob = 0.005#0.010
    error_readout = pauli_error([('X', prob),('I', 1-prob)])
    noise_future.add_all_qubit_quantum_error(error_readout, 'measure')

    #CX-error
    prob1 = 0.003#0.006
    prob2 = 0.004#0.008
    errors_cx = pauli_error([('X', prob1),('I', 1-prob1)]).expand(pauli_error([('X', prob2),('I', 1-prob2)]))
    noise_future.add_all_qubit_quantum_error(errors_cx, "cx")
    #"""
    return noise_future

def dict_to_file(D, filename):
    """
    Write dictionary to a json file.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(D,f)
    return filename

def file_to_dict(filename):
    """
    Read a json file into a dctionary.
    """
    with open(filename) as f:
        data = f.read()
    D = json.loads(data)
    return D

def hardware_exc(circ, backend):
    """
    Perform a hardware execution on backend
    """
    job = execute(circ, backend, shots = Nr_shots,)
    return job.job_id()

def create_full_dictionary(n):
    """
    Create a dictionary with all possible binary vectors in n dimensions.
    """
    if( (n is not int) & (n<=0) ):
        print("Invalid argument", n)
        return None
    v = [0]*n
    d = {}
    while(v[0]<2):
        ts = ""
        for i in range(n):
            ts += str(v[i])
        d[ts]=0
       
        v[n-1] += 1
        for i in range(n-1,0,-1):
            if(v[i]>=2):
                v[i] += -2
                v[i-1] += 1
            else:
                break
    return d
    
def convert_counts_to_prob(counts1):
    """
    Convert Qiskit's counts of a state into a dictionary with measurement probabilities.
    """
    mxl = 0
    for x in counts1.keys():
        mxl = len(x)
        break
    p = create_full_dictionary(mxl)
    n1 = 0
    for x in counts1.keys():
        n1 += counts1[x]
        if(len(x)!=mxl):
            print("State length not constant.")
            return None
    for x in counts1.keys():
        if( x not in p.keys() ):
            print("Wrong dictionary created.", p, x)
            return None
        else:
            p[x] = counts1[x] / n1
    return p

def c_R2(counts1, counts2, a):
    """
    For noiseless counts1 and noisy counts2 determine how much of the difference R2 
    can be explained by our simple noise model with error probability a.
    """
    p = convert_counts_to_prob(counts1)
    q = convert_counts_to_prob(counts2)
    n_states = len(p)
    if(p.keys() != q.keys()):
        print("Inconsistent, count sets.\n", p, "\n", q )
        return None
    V0 = 0
    V1 = 0
    V1_0 = 0
    for x in p.keys():
        V0 += (p[x]-q[x])**2
        V1 += ( a*p[x] + ((1-a)/n_states) - q[x] )**2
    if( (V1 < V0) & (V1>0.0) ):
        R2 = 1 - (V1/V0)
    else:
        R2 = 0
    return R2

def analyse_noise(counts1, counts2):
    """
    We are testing the hypothesis that a random answer is returned, if an error occurs. 
    The function returns two numbers. 
    The first is the error probability under this hypothesis. 
    The second is the extend to which this hypothesis can explain the observations.
    The first argument is the noiseless results of a circuit. The second is the noisy run.
    """
    p = convert_counts_to_prob(counts1)
    q = convert_counts_to_prob(counts2)
    n_states = len(p)
    if(p.keys() != q.keys()):
        print("Inconsistent, count sets.\n", p, "\n", q )
        return None
    pp = -1
    pq = -1
    for x in p.keys():
        pp += n_states * (p[x]**2)
        pq += n_states * (p[x]*q[x])
    a = 0
    if(abs(pp) > 0.001):
        a = pq/pp
    else:
        if( abs(pq) < abs(pp) ):
            a = pq/pp
        elif(np.sign(pp*pq)>=1):
            a = 1
        elif(np.sign(pp*pq)<=-1):
            a = -1
        else: a = 0 
    if(a<-1) :
        a = -1
    elif(a>1):
        a = 1
    R2 = c_R2(counts1, counts2, a)
    return a, R2

def SSE_empty(n):
    if(n==1):
        return [0,0,0]
    SSE = []
    for i in range(3):
        SSE.append(SSE_empty(n-1))
    return SSE.copy()

def optimize_noise_model_simple(R):
    """
    A simple search algorithm to find the optimal parameters of the noise model.
    It returns  probaility of a good measurement b 
    and the deterioration base for the depth cd and the size cs of the circuit.
    Finally, it returns the best sum of remaining squared errors.
    """
    cd = 1.0
    cs = 1.0
    b = 1.0
    SSE_best = 0
    for [q,p,a,R2_avg,R2_std,add,ads,ctr] in R:
        SSE_best += ctr * ((b*(cd**(0.01*(0.01*add)))*(cs**(0.01*(0.01*ads)))-a)**2)
    cont = 1
    while( cont == 1 ):
        cont = 0
        cd1 = cd - 0.00001
        cd2 = cd + 0.00001
        cs1 = cs - 0.00001
        cs2 = cs + 0.00001
        b1 = b - 0.0001
        b2 = b + 0.0001
        SSE_b1  = 0
        SSE_b2  = 0
        SSE_cd1 = 0
        SSE_cd2 = 0
        SSE_cs1 = 0
        SSE_cs2 = 0
        for [q,p,a,R2_avg,R2_std,add,ads,ctr] in R:
            SSE_b1  += ctr*((b1*(cd**(0.01*add))*(cs**(0.01*ads))-a)**2)
            SSE_b2  += ctr*((b2*(cd**(0.01*add))*(cs**(0.01*ads))-a)**2)
            SSE_cd1 += ctr*((b*(cd1**(0.01*add))*(cs**(0.01*ads))-a)**2)
            SSE_cd2 += ctr*((b*(cd2**(0.01*add))*(cs**(0.01*ads))-a)**2)
            SSE_cs1 += ctr*((b*(cd**(0.01*add))*(cs1**(0.01*ads))-a)**2)
            SSE_cs2 += ctr*((b*(cd**(0.01*add))*(cs2**(0.01*ads))-a)**2)
        if( (b1>=0) & (b1<=1) & ( SSE_b1 < SSE_best ) ):
            cont = 1
            SSE_best = SSE_b1
            b = b1
        if( (b2>=0) & (b2<=1) & ( SSE_b2 < SSE_best ) ):
            cont = 1
            SSE_best = SSE_b2
            b = b2
        if( (cd1>=0) & (cd1<=1) & ( SSE_cd1 < SSE_best ) ):
            cont = 1
            SSE_best = SSE_cd1
            cd = cd1
        if( (cd2>=0) & (cd2<=1) & ( SSE_cd2 < SSE_best ) ):
            cont = 1
            SSE_best = SSE_cd2
            cd = cd2
        if( (cs1>=0) & (cs1<=1) & ( SSE_cs1 < SSE_best ) ):
            cont = 1
            SSE_best = SSE_cs1
            cs = cs1
        if( (cs2>=0) & (cs2<=1) & ( SSE_cs2 < SSE_best ) ):
            cont = 1
            SSE_best = SSE_cs2
            cs = cs2
    return b,cd, cs, SSE_best


def optimize_noise_model_bcd(R):
    """
    A simple search algorithm to find the optimal parameters of the noise model.
    It returns  probaility of a good measurement b 
    and the deterioration base for the depth cd.
    Finally, it returns the best sum of remaining squared errors.
    """
    cd = 1.0
    b = 1.0
    eps = 0.0001
    SSE_best = 0
    for [q,p,a,R2_avg,R2_std,add,ads,ctr] in R:
        SSE_best += ctr*((b*(cd**(0.01*(0.01*add)))-a)**2)
    cont = 1
    while( cont == 1 ):
        cont = 0
        b_p = [b-eps,b,b+eps].copy()
        cd_p = [cd-eps,cd,cd+eps].copy()
        #SSE[b_p][cd_p]
        SSE = SSE_empty(2)
        for [q,p,a,R2_avg,R2_std,add,ads,ctr] in R:
            for ib in range(3):
                for icd in range(3):
                    SSE[ib][icd] += ctr*((b_p[ib]*(cd_p[icd]**(0.01*add))-a)**2)
        for ib in range(3):
            if( (b_p[ib]>=0) & (b_p[ib]<=1) ):
                for icd in range(3):
                    if( (cd_p[icd]>=0) & (cd_p[icd]<=1) ):
                        if( SSE[ib][icd] < SSE_best ):
                            cont = 1
                            SSE_best = SSE[ib][icd]
                            b  =  b_p[ib]
                            cd = cd_p[icd]
    return b,cd, SSE_best

def optimize_noise_model_bcs(R):
    """
    A simple search algorithm to find the optimal parameters of the noise model.
    It returns  probaility of a good measurement b 
    and the deterioration base for the size cs of the circuit.
    Finally, it returns the best sum of remaining squared errors.
    """
    cs = 1.0
    b = 1.0
    eps = 0.0001
    SSE_best = 0
    for [q,p,a,R2_avg,R2_std,add,ads,ctr] in R:
        SSE_best += ctr*((b*(cs**(0.01*(0.01*ads)))-a)**2)
    cont = 1
    while( cont == 1 ):
        cont = 0
        b_p = [b-eps,b,b+eps].copy()
        cs_p = [cs-eps,cs,cs+eps].copy()
        #SSE[b_p][cd_p]
        SSE = SSE_empty(2)
        for [q,p,a,R2_avg,R2_std,add,ads,ctr] in R:
            for ib in range(3):
                for ics in range(3):
                    SSE[ib][ics] += ctr*((b_p[ib]*(cs_p[ics]**(0.01*ads))-a)**2)
        for ib in range(3):
            if( (b_p[ib]>=0) & (b_p[ib]<=1) ):
                for ics in range(3):
                    if( (cs_p[ics]>=0) & (cs_p[ics]<=1) ):
                        if( SSE[ib][ics] < SSE_best ):
                            cont = 1
                            SSE_best = SSE[ib][ics]
                            b  =  b_p[ib]
                            cs = cs_p[ics]
    return b,cs, SSE_best

def clean_exc(circ):
    """
    Perform a noiseless simulation of the circuit
    """
    result = execute(circ, Aer.get_backend('qasm_simulator'),shots = Nr_shots).result()
    counts1 = result.get_counts(0)
    return counts1

def noisy_exc(circ, noise_model):
    """
    Perform a simulation of the circuit using a sepcified noise model
    """
    result = execute(circ, Aer.get_backend('qasm_simulator'),
                     basis_gates=noise_model.basis_gates,
                     shots = Nr_shots,
                     noise_model=noise_model).result()
    counts2 = result.get_counts(0)
    return counts2

def random_circ(q,t):
    """
    Generate a circuit on q qubits 
    and with t rounds of randomly multi-controlled rotations on the final qubit.
    """
    Qreg = qiskit.QuantumRegister( q )
    Creg = qiskit.ClassicalRegister( 1 )
    circ = qiskit.QuantumCircuit(Qreg, Creg)
    n = q-1
    slc_0 = []
    for _ in range(n):
        slc_0.append(0)
    
    for _ in range(t):
        slc = []
        for _ in range(n):
            slc.append(0)
        while( slc == slc_0 ):
            slc = [ np.random.choice([0,1]) for _ in range(n) ]
        Q_slc = []
        for i in range(n):
            if(1==slc[i]):
                Q_slc.append(Qreg[i])
                circ.x(Qreg[i])
        angle = np.pi * np.random.uniform()
        circ.mcry(angle,Q_slc,Qreg[n])
        for i in range(n-1,-1,-1):
            if(1==slc[i]):
                circ.x(Qreg[i])
    
    circ.measure(Qreg[n], Creg)
    tdc = transpile(circ.decompose(reps=2), basis_gates=load_noise_thermal().basis_gates, coupling_map=get_coupling_map(q), optimization_level=3)
    return tdc

def r01c(par):
    """
    Useless circuit
    """
    Qreg = qiskit.QuantumRegister( 1 )
    Creg = qiskit.ClassicalRegister( 1 )
    circ = qiskit.QuantumCircuit(Qreg, Creg)
    if(par > 0.5):
        circ.x(Qreg[0])
    circ.measure(Qreg, Creg)
    tdc = transpile(circ.decompose(reps=2), basis_gates=load_noise_thermal().basis_gates, coupling_map=get_coupling_map(7), optimization_level=3)
    return tdc

def all_keys(n):
    """
    Generate all n-bit vectors/keys of a dictionary.
    """
    if(n==1):
        return ['0', '1']
    keys = []
    temp = all_keys(n-1)
    for x in temp:
        keys.append(x+'0')
        keys.append(x+'1')
    return keys.copy()

def counts_to_prob(counts):
    """
    Convert Qiskit's counts of a state into a dictionary with measurement probabilities.
    """
    for x in counts.keys():
        n=len(x)
        break
    keys = all_keys(n)
    prob = []
    summed = 0
    for x in keys:
        if(x in counts.keys()):
            prob.append( counts[x] )
            summed += counts[x]
        else:
            prob.append(0.0)
    for i in range(len(keys)):
        prob[i] *= 1/summed
    return prob.copy()

def rsq(a0, a1, a2, a3):
    """
    Another useless circuit
    """
    Qreg = qiskit.QuantumRegister( 1 )
    Creg = qiskit.ClassicalRegister( 1 )
    circ = qiskit.QuantumCircuit(Qreg, Creg)
    if(a0>0.5):
        circ.x(Qreg[0])
    circ.u(a1, a2, a3,0)
    circ.measure(Qreg, Creg)
    tdc = transpile(circ.decompose(reps=2), basis_gates=load_noise_thermal().basis_gates, coupling_map=get_coupling_map(7), optimization_level=3)
    return tdc

def null_circ():
    """
    Empty circuit
    """
    Qreg = qiskit.QuantumRegister( 1 )
    Creg = qiskit.ClassicalRegister( 1 )
    circ = qiskit.QuantumCircuit(Qreg, Creg)
    circ.measure(Qreg, Creg)
    tdc = transpile(circ.decompose(reps=2), basis_gates=load_noise_thermal().basis_gates, coupling_map=get_coupling_map(7), optimization_level=3)
    return tdc
    

def correct_for_meas_error(p_meas_err, s):
    """
    A single-qubit circuit is prepared in '0' with probability q. 
    The probability of a correct measurement is 1-p and of a flipped measurement p.
    We will measure '0' with probability (1-q)p+(1-p)q=s, so that $q=(s-p)/(1-2p)$.
    """
    p = p_meas_err
    s0 = s[0]
    s1 = s[1]
    if( (s0>p) & ((1-2*p)>0) ):
        q0 = (s0-p)/(1-2*p)
    else:
        return [-1,-1]
    q1 = 1-q0
    return [q0,q1]

def prob_dist_2(pd1,pd2):
    """
    Compute the euclidean length between 2 probability distributions.
    """
    d=0
    if(len(pd1)!=len(pd2)):
        print("Probability distributions of unequal length.", pd1, pd2)
        return None
    n = len(pd1)
    for i in range(n):
        d += (pd1[i]-pd2[i])**2
    return np.sqrt(d)

def L(theta,counts_lst):
    """
    The likelihood z of an unknown angle theta given the measurements in counts_lst.
    """
    z = 0
    for [m,c0,c1] in counts_lst:
        z += (c1/(c0+c1)) * np.log( np.sin( (2*m+1) * theta ) ** 2 ) + (c0/(c0+c1)) * np.log( ( np.cos( (2*m+1) * theta ) ) ** 2 )
    return z

def dL(theta,counts_lst):
    """
    The derivative wrt theta of the likelihood function.
    """
    z = 0.0
    for [m,c0,c1] in counts_lst:
        z += m*c1*( np.cos( (2*m+1) * theta ) / np.sin( (2*m+1) * theta ) ) - m*c0*( np.sin( (2*m+1) * theta ) / np.cos( (2*m+1) * theta ) )
    return z

def maximize_L(counts_lst):
    """
    Scan the domain [0,pi/2] too look for points of zero derivative.
    For these points the likelihood is computed to find the best angle.
    """
    eps = 0.00001
    delta = 0.01
    x0 = eps
    y0 = dL(x0, counts_lst)
    xM = np.pi/2
    
    zeros = [eps, xM-eps]
    if( abs(y0) < eps ):
        zeros.append(x0)
    
    intervals = []
    x = x0 + delta
    while( x < xM ):
        y = dL(x,counts_lst)
        if( ( y < -eps ) & ( y0 > 0.0 ) ):
            intervals.append([x0,x])
        elif( ( y > eps ) & ( y0 < 0.0 ) ):
            intervals.append([x0,x])
        elif(abs(y) < eps):
            zeros.append(x)
        x0 = x
        x += delta
        y0 = y
    
    for [x0,x1] in intervals:
        y0 = dL(x0,counts_lst)
        y1 = dL(x1,counts_lst)
        while( abs(x1-x0) > eps ):
            x = (x1+x0)/2
            y = dL(x,counts_lst)
            if( abs(y) < eps ):
                break
            elif( ( y < -eps ) & ( y0 > eps ) ):
                y1 = y
                x1 = x
            elif( ( y > eps ) & ( y0 < -eps ) ):
                y1 = y
                x1 = x
            elif( ( y < -eps ) & ( y1 > eps ) ):
                y0 = y
                x0 = x
            elif( ( y > eps ) & ( y1 < -eps ) ):
                y0 = y
                x0 = x
            else:
                print("Error!")
                break
        x = (x1+x0)/2
        zeros.append(x)
    
    maxi = 1
    z = 0.0
    for x in zeros:
        y = L(x,counts_lst)
        if( (y>maxi) | (maxi > 0.0) ):
            maxi=y
            z = x
            
    return z

def check(n, prob_lst, f_lst):
    """
    Check if the integration problem fits on the circuit.
    """
    if( n != 1+ceil(np.log2(len(f_lst))) ):
        print("The f_lst requires {0:d} qubits, but received {1:d}.".format(1+ceil(np.log2(len(f_lst))), n))
        return False
    if( n-1 != ceil(np.log2(len(prob_lst))) ):
        print("The prob_lst requires {0:d} qubits, but received {1:d}.".format(1+ceil(np.log2(len(prob_lst))), n))
        return False
    return True

def MLE_MCI_f(nr_of_qubits,powers, prob_lst, f_lst):
    """
    Function that generates a circuit of nr_of_qubits qubits with powers amplification rounds 
    for the maximum likelihood estimator for monte carlo integration of the function f_lst 
    against the probability distribution prob_lst.
    """
    n = nr_of_qubits
    qc = QuantumCircuit(n,1)
    all_qubits = [i for i in range(n)]
    sample_qubits = [i for i in range(n-1)]

    if( check(n, prob_lst, f_lst) == False ):
        print("Parameters do not specify a quantum circuit.")
        return None
    
    #Prepare samping points / state
    qc.append(uni_prob_mat(prob_lst), sample_qubits)

    #qc.barrier()

    #Evaluate function
    qc.append(uni_integration_mat(f_lst),all_qubits)#f(x)=f(x)
    

    for _ in range(powers):
        #qc.barrier()    
        qc.z(n-1)
        #qc.barrier()
        #Invert function
        qc.append(inv_uni_integration_mat(f_lst),all_qubits)#f(x)=f(x)
        
        #qc.barrier()
        qc.x(n-1)
        qc.h(n-1)
        qc.append(inv_uni_prob_mat(prob_lst), sample_qubits)
        for i in range(n-1):
            qc.x(i)
        qc.mcx(sample_qubits,n-1)
        qc.h(n-1)
        qc.x(n-1)
        for i in range(n-1):
            qc.x(i)
        qc.append(uni_prob_mat(prob_lst), sample_qubits)
        #qc.barrier()
        #Evaluate function
        qc.append(uni_integration_mat(f_lst),all_qubits)#f(x)=f(x)

    qc.measure(n-1,0)
    return qc

def MLE_MCI_x(nr_of_qubits,powers):
    """
    Function that generates a circuit of nr_of_qubits qubits with powers amplification rounds 
    for the maximum likelihood estimator for monte carlo integration of the function f(x)=x 
    against the uniform probability distribution.
    """
    n = nr_of_qubits
    qc = QuantumCircuit(n,1)
    all_qubits = [i for i in range(n)]
    sample_qubits = [i for i in range(n-1)]

    #Prepare samping points / state
    qc.append(uniform_dist(n-1), sample_qubits)

    #qc.barrier()

    #Evaluate function
    qc.append(x_function(n),all_qubits)#f(x)=x

    for _ in range(powers):
        #qc.barrier()    
        qc.z(n-1)
        #qc.barrier()
        #Invert function
        qc.append(inv_x_function(n),all_qubits)#f(x)=x
        
        #qc.barrier()
        qc.x(n-1)
        qc.h(n-1)
        qc.append(inv_uniform_dist(n-1), sample_qubits)
        for i in range(n-1):
            qc.x(i)
        qc.mcx(sample_qubits,n-1)
        qc.h(n-1)
        qc.x(n-1)
        for i in range(n-1):
            qc.x(i)
        qc.append(uniform_dist(n-1), sample_qubits)
        #qc.barrier()
        #Evaluate function
        qc.append(x_function(n),all_qubits)#f(x)=x

    qc.measure(n-1,0)
    return qc

def MLE_MCI_sin2(nr_of_qubits,powers):
    """
    Function that generates a circuit of nr_of_qubits qubits with powers amplification rounds 
    for the maximum likelihood estimator for monte carlo integration of the function f(t)=sin^2(t) 
    against the uniform probability distribution.
    """
    n = nr_of_qubits
    qc = QuantumCircuit(n,1)
    all_qubits = [i for i in range(n)]
    sample_qubits = [i for i in range(n-1)]

    #Prepare samping points / state
    qc.append(uniform_dist(n-1), sample_qubits)

    #qc.barrier()

    #Evaluate function
    qc.append(sin2_function(n),all_qubits)#f(t)=sin^{2}(t)

    for _ in range(powers):
        #qc.barrier()    
        qc.z(n-1)
        #qc.barrier()
        #Invert function
        qc.append(inv_sin2_function(n),all_qubits)#f(t)=sin^{2}(t)
        
        #qc.barrier()
        qc.x(n-1)
        qc.h(n-1)
        qc.append(inv_uniform_dist(n-1), sample_qubits)
        for i in range(n-1):
            qc.x(i)
        qc.mcx(sample_qubits,n-1)
        qc.h(n-1)
        qc.x(n-1)
        for i in range(n-1):
            qc.x(i)
        qc.append(uniform_dist(n-1), sample_qubits)
        #qc.barrier()
        #Evaluate function
        qc.append(sin2_function(n),all_qubits)#f(t)=sin^{2}(t)

    qc.measure(n-1,0)
    return qc

def uniform_dist(domain_qubits):
    """
    Create the uniform distribution on the _domain_qubits using some Hadamard gates.
    """
    qc = QuantumCircuit(domain_qubits)
    for i in range(domain_qubits):
        qc.h(i)
    return qc.to_gate()

def inv_uniform_dist(domain_qubits):
    """
    Invert the uniform distribution
    """
    qc = QuantumCircuit(domain_qubits)
    for i in range(domain_qubits):
        qc.h(i)
    return qc.to_gate()

def uni_prob_mat(prob_lst):
    """
    This function creates a gate that rotates the zero state (1,0,0,0,0,...) to the specified prob_list.
    """
    n = ceil(np.log2(len(prob_lst)))
    #Copy parameters
    prob_square_sum = 0.0
    for x in prob_lst:
        prob_square_sum += abs(x)
    X = []
    for x in prob_lst:
        X.append( np.sqrt( abs( x / prob_square_sum) ) )
    while( len(X) < 2**n ):
        X.append(0)
    
    #Compute the arguments
    prob = []
    for _ in range(n+1):
        prob.append([])    
    for j in range(0,2**n):
        prob[n].append( X[j]**2 )
    for i in range(n-1,-1,-1):
        for ctr in range(0,len(prob[i+1]),2):
            x = prob[i+1][ctr] + prob[i+1][ctr+1]
            prob[i].append(x)
    arg = []
    for _ in range(n):
        arg.append([])
    for i in range(1,n+1):
        for j in range(0,len(prob[i]),2):
            arg[i-1].append( np.arccos( np.sqrt( prob[i][j] / prob[i-1][j//2] ) ) )
    
    #Build the circuit
    register = qiskit.QuantumRegister( n )
    qc = qiskit.QuantumCircuit(register)
    
    ni = (2**(n-1))-1
    
    for i in range(0,n):
        control_reg = [ register[j] for j in range(n-1,-1,-1) ]
        rotation_reg = control_reg.pop(i)
        
        for j in range(2**i):    
            ind = [int(b) for b in np.binary_repr(ni-(2**(n-1-i))*j,width=n-1) ]
            ps = []
            for s in range(0,n-1):
                if( ind[s] == 1 ):
                    qc.x(control_reg[s])
                    ps.append(s)
            
            qc.mcry(2*arg[i][j], control_reg, rotation_reg)

            for s in range(n-2,-1,-1):
                if( ind[s] == 1 ):
                    qc.x(control_reg[s])
    
    return qc.to_gate()

def inv_uni_prob_mat(prob_lst):
    """
    This function creates a gate that rotates the specified list back to the zero state.
    It is the inverse of uni_prob_mat.
    """
    n = ceil(np.log2(len(prob_lst)))
    
    #Copy parameters
    prob_square_sum = 0.0
    for x in prob_lst:
        prob_square_sum += abs(x)
    X = []
    for x in prob_lst:
        X.append( np.sqrt( abs( x / prob_square_sum) ) )
    while( len(X) < 2**n ):
        X.append(0)
    
    #Compute the arguments
    prob = []
    for _ in range(n+1):
        prob.append([])    
    for j in range(0,2**n):
        prob[n].append( X[j]**2 )
    for i in range(n-1,-1,-1):
        for ctr in range(0,len(prob[i+1]),2):
            x = prob[i+1][ctr] + prob[i+1][ctr+1]
            prob[i].append(x)
    arg = []
    for _ in range(n):
        arg.append([])
    for i in range(1,n+1):
        for j in range(0,len(prob[i]),2):
            arg[i-1].append( np.arccos( np.sqrt( prob[i][j] / prob[i-1][j//2] ) ) )
    
    #Build the circuit
    register = qiskit.QuantumRegister( n )
    qc = qiskit.QuantumCircuit(register)
    
    ni = (2**(n-1))-1
    
    for i in range(n-1,-1,-1):
        control_reg = [ register[j] for j in range(n-1,-1,-1) ]
        rotation_reg = control_reg.pop(i)
        
        for j in range((2**i)-1,-1,-1):    
            ind = [int(b) for b in np.binary_repr(ni-(2**(n-1-i))*j,width=n-1) ]
            ps = []
            for s in range(0,n-1):
                if( ind[s] == 1 ):
                    qc.x(control_reg[s])
                    ps.append(s)
            
            qc.mcry(-2*arg[i][j], control_reg, rotation_reg)

            for s in range(n-2,-1,-1):
                if( ind[s] == 1 ):
                    qc.x(control_reg[s])
            #print(i, j, control_reg, rotation_reg, ni-(2**(n-1-i))*j, ind, ps, 2*arg[i][j])
    
    return qc.to_gate()

def uni_integration_mat(lst_ori):
    """
    This generates a matrix simulating a function with values as in lst_ori 
    on the states 0..00, 0..01, 0..10, 0..11, ..., 1..11 .
    """    
    lst = lst_ori.copy()
    n = 1+ceil(np.log2(len(lst)))
    
    while( len(lst) < 2**(n-1) ):
        lst.append(0)
    
    register = qiskit.QuantumRegister( n )
    qc = qiskit.QuantumCircuit(register)

    for i in range(0,2**(n-1)):
        
        ind = [int(b) for b in np.binary_repr((2**(n-1))-1-i,width=n-1) ]
        for j in range(0,n-1):
            if( ind[j] == 1 ):
                qc.x(register[n-2-j])
        angle = np.arcsin(-np.sqrt(lst[i]))
        qc.mcry( 2*angle, [register[j] for j in range(0,n-1)], register[n-1] )
        for j in range(n-2,-1,-1):
            if( ind[j] == 1 ):
                qc.x(register[n-2-j])

    qc.z(n-1)
    return qc.to_gate()

def inv_uni_integration_mat(lst_ori):
    """
    This is the inverse of the function uni_integration_mat.
    """
    lst = lst_ori.copy()
    n = 1+ceil(np.log2(len(lst)))
    
    while( len(lst) < 2**(n-1) ):
        lst.append(0)
    
    register = qiskit.QuantumRegister( n )
    qc = qiskit.QuantumCircuit(register)

    qc.z(n-1)
    for i in range((2**(n-1))-1,-1,-1):
        
        ind = [int(b) for b in np.binary_repr((2**(n-1))-1-i,width=n-1) ]
        for j in range(0,n-1):
            if( ind[j] == 1 ):
                qc.x(register[n-2-j])
                
        angle = np.arcsin(-np.sqrt(lst[i]))
        qc.mcry( -2*angle, [register[j] for j in range(0,n-1)], register[n-1] )
        for j in range(n-2,-1,-1):
            if( ind[j] == 1 ):
                qc.x(register[n-2-j])

    
    return qc.to_gate()

def x_function(nr_of_qubits):
    """
    Gate implementation of the function f(x)=x
    """
    qc = QuantumCircuit(nr_of_qubits)
    n = nr_of_qubits
    p = n-1
    qc.ry( 2* (np.pi/2) * (2**(-n) ) , n-1)
    for i in range(p):
        qc.cry( 2 * (np.pi/2) * (2**(i-p) ) , i, n-1)
    gate = qc.to_gate()
    return gate

def inv_x_function(nr_of_qubits):
    """
    Inverse of x_function
    """
    qc = QuantumCircuit(nr_of_qubits)
    n = nr_of_qubits
    p = n-1
    qc.ry( -2* (np.pi/2) * (2**(-n) ) , n-1)
    for i in range(p):
        qc.cry( -2 * (np.pi/2) * (2**(i-p) ) , i, n-1)
    gate = qc.to_gate()
    return gate

def sin2_function(nr_of_qubits):
    """
    Gate implementation of the function f(t)=sin^2(t)
    """
    qc = QuantumCircuit(nr_of_qubits)
    n = nr_of_qubits
    global b
    p = n-1
    qc.ry(2*b*(2**(-n)), n-1)
    for i in range(p):
        qc.cry(2*b*(2**(i-p)), i, n-1)
    gate = qc.to_gate()
    return gate

def inv_sin2_function(nr_of_qubits):
    """
    Inverse of sin2_function
    """
    qc = QuantumCircuit(nr_of_qubits)
    n = nr_of_qubits
    global b
    p = n-1
    qc.ry(-2*b*(2**(-n)), n-1)
    for i in range(p):
        qc.cry(-2*b*(2**(i-p)), i, n-1)
    gate = qc.to_gate()
    return gate

def get_coupling_map(q):
    if(q==4):
        coupling_map = CouplingMap([[0,1], [1,0], [1,2], [2,1], [1,3], [3,1]])
    elif(q==5):
        coupling_map = CouplingMap([[0,1], [1,0], [1,2], [2,1], [1,3], [3,1], [3,4], [4,3]])
    elif(q==6):
        coupling_map = CouplingMap([[0,1], [1,0], [1,2], [2,1], [1,3], [3,1], [3,5], [5,3], [4,5], [5,4]])
    elif(q==7):
        coupling_map = CouplingMap([[0,1], [1,0], [1,2], [2,1], [1,3], [3,1], [3,5], [5,3], [4,5], [5,4], [5,6], [6,5]])
    elif(q==8):
        coupling_map = CouplingMap([[0,1], [1,0], [1,2], [2,1], [1,3], [3,1], [3,5], [5,3], [4,5], [5,4], [5,6], [6,5], [2,7], [7,2]])
    elif(q==9):
        coupling_map = CouplingMap([[0,1], [1,0], [1,2], [2,1], [1,3], [3,1], [3,5], [5,3], [4,5], [5,4], [5,6], [6,5], [2,7], [7,2], [6,8], [8,6]])
    elif(q==10):
        coupling_map = CouplingMap([[0,1], [1,0], [1,2], [2,1], [1,3], [3,1], [3,5], [5,3], [4,5], [5,4], [5,6], [6,5], [2,7], [7,2], [6,8], [8,6], [7,9], [9,7]])
    elif(q==11):
        coupling_map = CouplingMap([[0,1], [1,0], [1,2], [2,1], [1,3], [3,1], [3,5], [5,3], [4,5], [5,4], [5,6], [6,5], [2,7], [7,2], [6,8], [8,6], [7,9], [9,7], [8,10], [10,8]])
    elif(q==12):
        coupling_map = CouplingMap([[0,1], [1,0], [1,2], [2,1], [1,3], [3,1], [3,5], [5,3], [4,5], [5,4], [5,6], [6,5], [2,7], [7,2], [6,8], [8,6], [7,9], [9,7], [8,10], [10,8], [11,9], [9,11], [10,11], [11,10]])
    elif(q==13):
        coupling_map = CouplingMap([[0,1], [1,0], [1,2], [2,1], [1,4], [4,1], [2,3], [3,2], [3,5], [5,3], [5,8], [8,5], [4,6], [6,4], [6,7], [7,6], [8,9], [9,8], [9,11], [11,9], [7,10], [10,7], [10,12], [12,10], [11,12], [12,11]])
    elif(q==14):
        coupling_map = CouplingMap([[0,1], [1,0], [1,2], [2,1], [1,4], [4,1], [2,3], [3,2], [3,5], [5,3], [5,8], [8,5], [4,7], [7,4], [6,7], [7,6], [7,10], [10,7], [8,9], [9,8], [9,11], [11,9], [10,12], [12,10], [11,13], [13,11], [12,13], [13,12]])
    elif(q==15):
        coupling_map = CouplingMap([[0,1], [1,0], [1,2], [2,1], [1,4], [4,1], [2,3], [3,2], [3,5], [5,3], [5,8], [8,5], [4,7], [7,4], [6,7], [7,6], [8,9], [9,8], [7,10], [10,7], [8,11], [11,8], [10,12], [12,10], [12,13], [13,12], [13,14], [14,13], [11,14], [14,11]])
    elif(q==16):
        coupling_map = CouplingMap([[0,1], [1,0], [1,2], [2,1], [1,4], [4,1], [2,3], [3,2], [3,5], [5,3], [5,8], [8,5], [4,7], [7,4], [6,7], [7,6], [8,9], [9,8], [7,10], [10,7], [8,11], [11,8], [10,12], [12,10], [12,13], [13,12], [12,15], [15,12], [13,14], [14,13], [11,14], [14,11]])
    else:
        print("Invalid q. It must be 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 or 16..")
        return None
    return coupling_map

#PARAMETERS
Nr_shots = 10001
N_t = 100
itr = 100


