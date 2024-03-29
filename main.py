## File can be run with command: python main.py

import numpy as np
import math 
from math import sqrt
from multiprocessing import cpu_count, Pool
from timeit import default_timer as timer


def UCB_grid(par, p_0, p_1, tau, time):

    opt_grid_reward = ComputeOptReward(p_0, p_1, tau) 

    num_arm = par+1
    # Array indices correspond to arm, sample total, number of pulls for arm, mu, ucb (in order)
    arr=np.zeros((5,num_arm), dtype=np.float32)
    ucb_grid_regret=0
    
    
    # Step 1: Divide into partition
    partition_size = 1/par
    for i in range(num_arm):
            arr[0][i] = partition_size*i
            arr[4][i] = np.inf
            
            
    # Step 2: UCB
    X = rng.uniform(0, 1, time) 
    for t in range(time):

        # Pull arm with highest UCB
        arm_index = np.argmax(arr[4])
        arm = arr[0][arm_index]
        arr[1][arm_index] += (X[t] <= ((arm < tau) * p_0 + (arm >= tau) * p_1)) - arm
        arr[2][arm_index] += 1
                    
        # Update UCB
        n = arr[2][arm_index]
        mu_hat = arr[1][arm_index]/n
        arr[3][arm_index] = mu_hat

        ucb = mu_hat + sqrt((2  * np.log(time)) / n)
        arr[4][arm_index] = ucb

        # Theoretical regret
        ucb_grid_reward = (arm >= tau) * (p_1-p_0) + p_0 - arm
        ucb_grid_regret += opt_grid_reward - ucb_grid_reward 
                
    tau_hat = arr[0][np.argmax(arr[4])]

    return tau, tau_hat, ucb_grid_regret


def UCB(tau_hat,N,p_0, p_1, tau, total_reg):
    # UCB with only 2 arms for Leftist
    
    opt_ucb_reward = ComputeOptReward(p_0, p_1, tau) 

    # Indices 0,1 are arm 0, arm tau_hat respectively.
    ucb=np.ones(2, dtype=np.float32)*np.inf 
    samp_num=np.zeros(2)
    mu_hat= np.zeros(2, dtype=np.float32)
    samples=np.zeros(2)
    ucb_reg = 0

    if N < 1:
        return tau_hat, ucb_reg, total_reg

    X = rng.uniform(0, 1, N) 
    for i in range(N):
        # Sampling the arm with the largest upper confidence bound
        arm_index = int(ucb[1] > ucb[0])    # break ties in favor of arm 0
        arm = (arm_index > 0)*tau_hat
        samples[arm_index] += (X[i] <= ((arm < tau)*p_0 + (arm >= tau)*p_1))-arm

        # Updating UCB
        samp_num[arm_index] += 1
        mu_hat[arm_index]=samples[arm_index]/samp_num[arm_index]
        ucb[arm_index] = mu_hat[arm_index]+ sqrt((2*np.log(N))/samp_num[arm_index])
            
        # Theoretical regret  
        ucb_reward = (arm >= tau)*(p_1-p_0) + p_0 - arm
        ucb_reg += opt_ucb_reward-ucb_reward
        total_reg += opt_ucb_reward-ucb_reward
    
    if ucb[0]>=ucb[1]:
        return 0, ucb_reg, total_reg
    else:
        return tau_hat, ucb_reg, total_reg


def NBS(delta_hat, R, T_remain, p_0, p_1, tau, total_regret, time):
    # Noisy Binary Search

    opt_NBS_reward = ComputeOptReward(p_0, p_1, tau) 
    epsilon_nbs = 1/time

    N = math.ceil(4*math.log((time*np.log2(time)))/(delta_hat**2))  # 4log(1/delta)/epsilon^2
    L = 0

    NBS_regret = 0

    if N > T_remain:
        return R, T_remain, NBS_regret, total_regret
    
   # Initial sample arm R
    p_R = (R < tau) * p_0 + (R >= tau) * p_1 
    hat_p_R = np.mean(rng.binomial(1, p_R, N)) 
    NBS_reward = (R >= tau) * (p_1 - p_0) + p_0 - R
    total_regret += N * (opt_NBS_reward - NBS_reward)    

    NBS_regret += N * (opt_NBS_reward - NBS_reward)
    T_remain -= N 
        
    while R-L >= (2*epsilon_nbs * (np.log2(time)**2) / delta_hat) and T_remain > 0: 
        # Get the new midpoint
        m = (L+R)/2
        if (T_remain - N < 0):
            N = T_remain


        p_m = (m < tau) * p_0 + (m >= tau) * p_1 
        hat_p_m = np.mean(rng.binomial(1, p_m, N)) 
        NBS_reward = (m >= tau) * (p_1 - p_0) + p_0 - m
        total_regret += N * (opt_NBS_reward - NBS_reward)    

        NBS_regret += N * (opt_NBS_reward - NBS_reward)
        T_remain -= N 

        # Replacing arm L or R with m based on which side of tau we estimate m to be on.
        if hat_p_R - hat_p_m >= delta_hat/2:
            L = m

        else: 
            R = m
            hat_p_R=hat_p_m
        
    return R, T_remain, NBS_regret, total_regret


def Leftist(p_0, p_1, tau, time):
    # One Dimensional Leftist

    opt_reward = ComputeOptReward(p_0, p_1, tau) 

    tau_hat = 1
    delta_hat = time
    T_remain = time
    epsilon = 1/8
    ar = 1
    N=math.ceil(math.log(2*(time*np.log2(time)))/(2*(epsilon**2)))   # log(2/delta)/2epsilon^2
    a = math.log(time) / sqrt(time)
    best_arm = 0
    
    leftist_regret, NBS_regret, UCB_regret, total_regret = 0, 0, 0, 0
        
    while epsilon >= a:

        # Check to see if there are enough pulls lefts to sample arm
        if (2*N > T_remain):
            break
        
        # Pull arms 0 and ar, N times each
        hat_p_zero = np.mean(rng.binomial(1, p_0, N)) 
        leftist_regret += N * (opt_reward - p_0)    
        p_ar = (ar < tau) * p_0 + (ar >= tau) * p_1 
        
        hat_p_ar = np.mean(rng.binomial(1, p_ar, N)) 
        ar_reward = (ar >= tau) * (p_1 - p_0) + p_0 - ar 
        leftist_regret += N * (opt_reward - ar_reward)    
        
        T_remain -= 2*N
        
        # Estimate of Delta
        delta_hat = hat_p_ar-hat_p_zero

        if (delta_hat-epsilon) >= epsilon:
            tau_hat, T_remain, NBS_regret, total_regret = NBS(epsilon,ar,T_remain, p_0, p_1, tau, leftist_regret, time) 
            best_arm, UCB_regret, total_regret = UCB(tau_hat,T_remain, p_0, p_1, tau, total_regret)
            T_remain = 0
            break
            
        else:
            if ar >= 8*epsilon:
                ar = ar/2
            epsilon = epsilon/2
            N = 4*N

        total_regret = leftist_regret

    if epsilon < a:   
        # Theoretical regret
        leftist_regret += (opt_reward - p_0)*T_remain
        total_regret = leftist_regret

    # tau returned for troubleshooting purposes
    return tau, best_arm, leftist_regret, NBS_regret, UCB_regret, total_regret


def EG(p_0, p_1, tau, time):
    # EG

    opt_reward = ComputeOptReward(p_0, p_1, tau) 

    tau_hat = 1
    delta_hat = time
    T_remain = time
    epsilon = 1/8
    ar = 1
    N=math.ceil(math.log(2*(time*np.log2(time)))/(2*(epsilon**2)))   # log(2/delta)/2epsilon^2
    a = math.log(time) / sqrt(time)
    best_arm = 0
    
    leftist_regret, NBS_regret, UCB_regret, total_regret = 0, 0, 0, 0
        
    while True:

        ##Check to see if there are enough pulls lefts to sample arm
        ## TODO is it needed?
        if (2*N > T_remain):
            break
        
        # Pull arms 0 and ar, N times each
        hat_p_zero = np.mean(rng.binomial(1, p_0, N)) 
        leftist_regret += N * (opt_reward - p_0)    
        p_ar = (ar < tau) * p_0 + (ar >= tau) * p_1 
        
        hat_p_ar = np.mean(rng.binomial(1, p_ar, N)) 
        ar_reward = (ar >= tau) * (p_1 - p_0) + p_0 - ar 
        leftist_regret += N * (opt_reward - ar_reward)    
        
        T_remain -= 2*N
        
        # Estimate of Delta
        delta_hat = hat_p_ar-hat_p_zero

        if (delta_hat-epsilon) >= epsilon:
            tau_hat, T_remain, NBS_regret, total_regret = NBS(epsilon,ar,T_remain, p_0, p_1, tau, leftist_regret, time) 
            best_arm, UCB_regret, total_regret = UCB(tau_hat,T_remain, p_0, p_1, tau, total_regret)
            T_remain = 0
            break
            
        else:
            epsilon = epsilon/2
            N = 4*N

        total_regret = leftist_regret

    # this will never happen b/c the while loop condition is now always True
    if epsilon < a:   
        # Theoretical regret
        leftist_regret += (opt_reward - p_0)*T_remain
        total_regret = leftist_regret

    # tau returned for troubleshooting purposes
    return tau, best_arm, leftist_regret, NBS_regret, UCB_regret, total_regret


def ComputeOptReward(p_0, p_1, tau):
    if p_1 - p_0 >= tau:
        return p_1 - tau
    else:
        return p_0


def run(i, seed):
    print('running experiment 1: ' + str(i))
    K=int(T**(0.5))                 # number of partitions
    global rng
    rng = np.random.default_rng(seed)

    # Experiment 1: Set p0 and p1 values, with a changing Tau value. 
    exp1_leftist_results=Leftist(p_0, p_1, tau[i], T)
    exp1_EG_results=EG(p_0, p_1, tau[i], T)
    results1 = UCB_grid(K, p_0, p_1, tau[i], T)

    return exp1_leftist_results[:6], exp1_EG_results[:6], results1[:3]


def run2(i, seed):
    print('running experiment 2: ' + str(i))
    K=int(T**(0.5))                 # number of partitions
    global rng
    rng = np.random.default_rng(seed)

    # Experiment 2: Set p0 and Tau values, with a changing Delta (p1=p0+Delta) value.
    leftist_results=Leftist(p_0,varying_p1[i], static_tau, T)
    results2 = UCB_grid(K, p_0, varying_p1[i], static_tau, T)
    EG_results_exp2=EG(p_0,varying_p1[i], static_tau, T)

    return leftist_results[:6], EG_results_exp2[:6], results2[:3]

def run3(i, seed):
    print('running experiment 3: ' + str(i))
    K=int(varying_time[i]**(0.5))
    global rng
    rng = np.random.default_rng(seed)

    # Experiment 3: Set p0, p1, and Tau values, with a changing time horizon, T.
    leftist_results=Leftist(p_0,p_1, static_tau, varying_time[i])
    results3 = UCB_grid(K, p_0, p_1, static_tau, varying_time[i])
    EG_results_exp3=EG(p_0, p_1, static_tau, varying_time[i])

    return leftist_results[:6], EG_results_exp3, results3[:3]


# Global variables
T=1000000

p_0 = 0.25
static_delta = 0.1
p_1 = p_0+static_delta
static_tau = 0.1

reps=25                             # Number of repetitions 
x=40                                # Number of unique values sampled
n=x*reps

tau = np.empty(n)
taulin = np.linspace(0.01,0.4,x)

x2=30
n2=x2*reps

varying_p1 = np.empty(n2)
delta = np.geomspace(5e-3, 0.55, x2)

x3 = 6
n3 = x3*reps
varying_time = np.zeros(n3,dtype=int)
time = np.geomspace(5e5, 16e6, x3, dtype=int)

y=0
for i in range(x):
    tau[y:y+reps]= taulin[i]
    y=y+reps

y=0
for i in range(x2):
    varying_p1[y:y+reps]=delta[i]+p_0  
    y=y+reps      

y=0
for i in range(x3):
    varying_time[y:y+reps]=time[i]
    y=y+reps      


# Record keeping

exp1_matrix_to_save_leftist = np.empty((n, 6))
exp1_EG = np.empty((n, 6))
exp1_grid_UCB=np.empty((n, 3))

exp2_matrix_to_save_leftist = np.empty((n2, 6))
exp2_grid_UCB = np.empty((n2, 3))
exp2_EG = np.empty((n2,6))

exp3_matrix_to_save=np.empty((n3, 6))
exp3_grid_UCB=np.empty((n3, 3))
exp3_EG = np.empty((n3,6))

exp4_matrix_to_save=np.empty((n3, 6))
exp4_grid_UCB=np.empty((n3, 3))
exp4_EG = np.empty((n3,6))


if __name__ == "__main__":
    
    cpu = cpu_count()
    print("Running on ", cpu, " cores")
    start = timer()

    entropy = 42
    seed_sequence = np.random.SeedSequence(entropy)
    seeds = seed_sequence.spawn(n)


    # Experiment 1 - varying tau
    with Pool(cpu) as pool:
        exp1_left, results1_EG, exp1_GUCB = zip(*pool.starmap(run, zip(range(n), seeds)))

    pool.close()
    pool.join()

    # Experiment 2 - varying delta
    s = seeds[0].spawn(n2)
    with Pool(cpu) as pool2:
            exp2_left, results2_EG, exp2_GUCB = zip(*pool2.starmap(run2, zip(range(n2), s)))

    pool2.close()
    pool2.join()

    for i in range(n):
        exp1_grid_UCB[i,:] = exp1_GUCB[i]
        exp1_matrix_to_save_leftist[i,:] = exp1_left[i]
        exp1_EG[i,:] = results1_EG[i]

    for i in range(n2):   
        exp2_matrix_to_save_leftist[i,:] = exp2_left[i]
        exp2_grid_UCB[i,:] = exp2_GUCB[i]
        exp2_EG[i,:] = results2_EG[i]

    static_tau = 0.1
    p_1 = 0.3

    # Experiment 3.2 - varying T, tau > Delta
    s = seeds[2].spawn(n3)
    with Pool(cpu) as pool3:
        exp4_left, results4_EG, exp4_GUCB = zip(*pool3.starmap(run3, zip(range(n3), s)))

    pool3.close()
    pool3.join()

    p_0 = 0.05
    static_tau = 0.075
    p_1 = 0.15

    # Experiment 3.1 - varying T, Delta > tau
    s = seeds[1].spawn(n3)
    with Pool(cpu) as pool4:
        exp3_left, results3_EG, exp3_GUCB = zip(*pool4.starmap(run3, zip(range(n3), s)))

    pool4.close()
    pool4.join()

    for i in range(n3):   
        exp3_matrix_to_save[i,:] = exp3_left[i]
        exp3_grid_UCB[i,:] = exp3_GUCB[i]
        exp4_matrix_to_save[i,:] = exp4_left[i]
        exp4_grid_UCB[i,:] = exp4_GUCB[i]
        exp3_EG[i:,] = results3_EG[i]
        exp4_EG[i:,] = results4_EG[i]

    end = timer()

    print(f'elapsed time: {end - start}')

    np.savetxt("exp1_grid_UCB.csv", exp1_grid_UCB, delimiter=',',header ='tau, best arm, UCB regret') 
    np.savetxt("exp1_leftist.csv",exp1_matrix_to_save_leftist, delimiter=',',header ='tau, best arm, Leftist regret, NSB regret, UCB regret, total regret')
    np.savetxt("exp1_EG.csv",exp1_EG, delimiter=',',header ='tau, best arm, Leftist regret, NSB regret, UCB regret, total regret')

    np.savetxt("exp2_leftist.csv",exp2_matrix_to_save_leftist, delimiter=',',header ='tau, best arm, Leftist regret, NSB regret, UCB regret, total regret')
    np.savetxt("exp2_grid_UCB.csv",exp2_grid_UCB, delimiter=',',header ='tau, best arm, UCB regret')
    np.savetxt("exp2_EG.csv",exp2_EG, delimiter=',',header ='tau, best arm, Leftist regret, NSB regret, UCB regret, total regret')

    np.savetxt("exp3_grid_UCB.csv", exp3_grid_UCB,  delimiter=',', header ='tau, best arm, UCB regret')
    np.savetxt("exp3_saveddata.csv", exp3_matrix_to_save,  delimiter=',', header ='tau, best arm, Leftist regret, NSB regret, UCB regret, total regret')
    np.savetxt("exp3_EG.csv",exp3_EG, delimiter=',',header ='tau, best arm, Leftist regret, NSB regret, UCB regret, total regret')

    np.savetxt("exp4_grid_UCB.csv", exp4_grid_UCB,  delimiter=',', header ='tau, best arm, UCB regret')
    np.savetxt("exp4_saveddata.csv", exp4_matrix_to_save,  delimiter=',', header ='tau, best arm, Leftist regret, NSB regret, UCB regret, total regret')  
    np.savetxt("exp4_EG.csv",exp4_EG, delimiter=',',header ='tau, best arm, Leftist regret, NSB regret, UCB regret, total regret')
