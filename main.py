import numpy as np
import math 
from math import sqrt
from multiprocessing import cpu_count, Pool
from timeit import default_timer as timer


def UCB_grid(par, p_0, p_1, tau):

    opt_grid_reward = ComputeOptReward(p_0, p_1, tau) # effiency change

    num_arm = par+1
    # Array indices correspond to arm, sample total, number of pulls for arm, mu, ucb (in order)
    arr=np.zeros((5,num_arm), dtype=np.float32)
    ucb_grid_regret=0
    cumu_reg = []
    
    
    # Step 1: Divide into partition
    partition_size = 1/par
    for i in range(num_arm):
            arr[0][i] = partition_size*i
            arr[4][i] = np.inf
            
            
    # Step 2: UCB
    X = rng.uniform(0, 1, T) # effiency change
    for t in range(T):

        # Pull arm with highest UCB
        arm_index = np.argmax(arr[4])
        arm = arr[0][arm_index]
        arr[1][arm_index] += (X[t] <= ((arm < tau) * p_0 + (arm >= tau) * p_1)) - arm
        arr[2][arm_index] += 1
                    
        # Update UCB
        n = arr[2][arm_index]
        mu_hat = arr[1][arm_index]/n
        arr[3][arm_index] = mu_hat

        ucb = mu_hat + sqrt((2  * np.log(t + 1)) / n)
        arr[4][arm_index] = ucb

        # Theoretical regret
        ucb_grid_reward = (arm >= tau) * (p_1-p_0) + p_0 - arm
        ucb_grid_regret += opt_grid_reward - ucb_grid_reward # effiency change

        # Flag for cumulative regret experiment
        if (reg and (t+1) % 10 == 0):
            cumu_reg.append(ucb_grid_regret) 
                
    tau_hat = arr[0][np.argmax(arr[4])]

    return tau, tau_hat, ucb_grid_regret, cumu_reg


def UCB(tau_hat,N,p_0, p_1, tau, total_reg, left_cumreg, timelapse):
    # UCB with only 2 arms for Leftist
    
    opt_ucb_reward = ComputeOptReward(p_0, p_1, tau) # efficiency change

    # Indices 0,1 are arm 0, arm tau_hat respectively.
    ucb=np.ones(2, dtype=np.float32)*np.inf 
    samp_num=np.zeros(2)
    mu_hat= np.zeros(2, dtype=np.float32)
    samples=np.zeros(2)
    ucb_reg = 0

    if N < 1:
        return tau_hat, ucb_reg, total_reg, left_cumreg

    X = rng.uniform(0, 1, N) # efficiency change
    for i in range(N):
        # Sampling the arm with the largest upper confidence bound
        arm_index = int(ucb[1] > ucb[0])    # break ties in favor of arm 0
        arm = (arm_index > 0)*tau_hat
        samples[arm_index] += (X[i] <= ((arm < tau)*p_0 + (arm >= tau)*p_1))-arm

        # Updating UCB
        samp_num[arm_index] += 1
        mu_hat[arm_index]=samples[arm_index]/samp_num[arm_index]
        ucb[arm_index] = mu_hat[arm_index]+ sqrt((2*np.log(i+1))/samp_num[arm_index])
            
        # Theoretical regret  
        ucb_reward = (arm >= tau)*(p_1-p_0) + p_0 - arm
        ucb_reg += opt_ucb_reward-ucb_reward
        total_reg += opt_ucb_reward-ucb_reward

        # Flag for cumulative regret experiment
        if reg:
            timelapse += 1
            if (timelapse % 10 == 0):
                left_cumreg.append(total_reg)
    
    if ucb[0]>=ucb[1]:
        return 0, ucb_reg, total_reg, left_cumreg
    else:
        return tau_hat, ucb_reg, total_reg, left_cumreg


def NBS(delta_hat, R, T_remain, p_0, p_1, tau, total_regret, left_cumreg, timelapse):
    # Noisy Binary Search

    opt_NBS_reward = ComputeOptReward(p_0, p_1, tau) # efficiency change

    N = math.ceil(4*math.log(1/err_delta)/(delta_hat**2)) 
    L = 0

    NBS_regret = 0

    if N > T_remain:
        return R, T_remain, NBS_regret, total_regret, left_cumreg, timelapse
    
   # Initial sample arm R
    p_R = (R < tau) * p_0 + (R >= tau) * p_1 # efficiency change
    hat_p_R = np.mean(rng.binomial(1, p_R, N)) # efficiency change
    NBS_reward = (R >= tau) * (p_1 - p_0) + p_0 - R

    if reg:
        for i in range(N): 
            total_regret += (opt_NBS_reward - NBS_reward)
            timelapse += 1
            if (timelapse % 10 == 0):
                left_cumreg.append(total_regret) 
    else:
        total_regret += N * (opt_NBS_reward - NBS_reward) # efficiency change   

    NBS_regret += N * (opt_NBS_reward - NBS_reward)
    T_remain -= N # efficiency change
        
    while R-L >= (epsilon_nbs * (np.log2(T)**2) / delta_hat) and T_remain > 0: 
        # Get the new midpoint
        m = (L+R)/2
        if (T_remain - N < 0):
            N = T_remain


        p_m = (m < tau) * p_0 + (m >= tau) * p_1 # efficiency change
        hat_p_m = np.mean(rng.binomial(1, p_m, N)) # efficiency change
        NBS_reward = (m >= tau) * (p_1 - p_0) + p_0 - m

        if reg:
            for i in range(N): 
                total_regret += (opt_NBS_reward - NBS_reward)
                timelapse += 1
                if (timelapse % 10 == 0):
                    left_cumreg.append(total_regret) 
        else:
            total_regret += N * (opt_NBS_reward - NBS_reward) # efficiency change   

        NBS_regret += N * (opt_NBS_reward - NBS_reward)
        T_remain -= N # efficiency change

        # Replacing arm L or R with m based on which side of tau we estimate m to be on.
        if hat_p_R - hat_p_m >= delta_hat/2:
            L = m

        else: 
            R = m
            hat_p_R=hat_p_m
        
    return R, T_remain, NBS_regret, total_regret, left_cumreg, timelapse


def Leftist(p_0, p_1, tau):
    # One Dimensional Leftist

    opt_reward = ComputeOptReward(p_0, p_1, tau) # efficiency change

    timelapse = 0
    tau_hat = 1
    delta_hat = T
    T_remain = T
    epsilon = 1/8
    ar = 1
    N=math.ceil(math.log(2/err_delta)/(2*(epsilon**2)))   
    a = 1/sqrt(T)
    best_arm = 0
    
    leftist_regret, NBS_regret, UCB_regret, total_regret = 0, 0, 0, 0
    left_cumreg = []
        
    while epsilon >= a:

        ##Check to see if there are enough pulls lefts to sample arm
        ## TODO is it needed?
        if (2*N > T_remain):
            break
        
        # Pull arms 0 and ar, N times each
        hat_p_zero = np.mean(rng.binomial(1, p_0, N)) # efficiency change

        if reg:
            for i in range(N): 
                leftist_regret += (opt_reward - p_0)
                timelapse += 1
                if (timelapse % 10 == 0):
                    left_cumreg.append(leftist_regret) 
        else:
            leftist_regret += N * (opt_reward - p_0) # efficiency change   

        p_ar = (ar < tau) * p_0 + (ar >= tau) * p_1 # efficiency change
        hat_p_ar = np.mean(rng.binomial(1, p_ar, N)) # efficiency change
        ar_reward = (ar >= tau) * (p_1 - p_0) + p_0 - ar # efficiency change

        if reg:
            for i in range(N): 
                leftist_regret += (opt_reward - ar_reward)
                timelapse += 1
                if (timelapse % 10 == 0):
                    left_cumreg.append(leftist_regret) 
        else:
            leftist_regret += N * (opt_reward - ar_reward) # efficiency change   
        
        T_remain -= 2*N
        
        # Estimate of Delta
        delta_hat = hat_p_ar-hat_p_zero

        if (delta_hat-epsilon) >= epsilon:
            tau_hat, T_remain, NBS_regret, total_regret, left_cumreg, time_from_NBS = NBS(epsilon,ar,T_remain, p_0, p_1, tau, leftist_regret, left_cumreg, timelapse) 
            best_arm, UCB_regret, total_regret, left_cumreg = UCB(tau_hat,T_remain, p_0, p_1, tau, total_regret, left_cumreg, time_from_NBS)
            T_remain = 0
            break
            
        else:
            if ar >= 8*epsilon:
                ar = ar/2
            epsilon = epsilon/2
            N = 4*N

        total_regret = leftist_regret

    if epsilon < a:   
        # Commit to zero for remaining rounds (never actually occurs)
        if reg:
            for i in range(T_remain):
                # Theoretical regret
                leftist_regret += (opt_reward - p_0)
                timelapse += 1
                if (timelapse % 10 == 0):
                    left_cumreg.append(leftist_regret)
        else:
            # Theoretical regret
            leftist_regret += (opt_reward - p_0)*T_remain

        total_regret += leftist_regret

        print("leftist commit 0 ", leftist_regret)

    if timelapse < 10:
        left_cumreg = leftist_regret

    # tau returned for troubleshooting purposes
    return tau, best_arm, leftist_regret, NBS_regret, UCB_regret, total_regret, left_cumreg


# efficiency change
def ComputeOptReward(p_0, p_1, tau):
    if p_1 - p_0 >= tau:
        return p_1 - tau
    else:
        return p_0


def run(i, seed):
    K=int(T**(0.5))                 # number of partitions
    global rng
    rng = np.random.default_rng(seed)

    # Experiment 1: Set p0 and p1 values, with a changing Tau value. 
    exp1_leftist_results=Leftist(p_0, p_1, tau[i])
    results1 = UCB_grid(K, p_0, p_1, tau[i])

    return exp1_leftist_results[:6],  results1[:3] #, leftist_results[:6]

def run2(i, seed):
    K=int(T**(0.5))                 # number of partitions
    global rng
    rng = np.random.default_rng(seed)

    # Experiment 2: Set p0 and Tau values, with a changing Delta (p1=p0+Delta) value.
    leftist_results=Leftist(p_0,varying_p1[i], static_tau)
    results2 = UCB_grid(K, p_0, varying_p1[i], static_tau)

    return leftist_results[:6], results2[:3]
    # return results2[:3]

def run3(reps, seed):

    K=int(T**(0.5))
    global rng
    rng = np.random.default_rng(seed)

    # Experiment 3: Cumulative regret
    leftist_results=Leftist(p_0,p_1, static_tau)
    results1 = UCB_grid(K, p_0, p_1, static_tau)

    return leftist_results[-1], results1[-1]


# Global variables
T=1000000
epsilon_nbs = 1/T
err_delta = 1/(T*np.log2(T))

p_0 = 0.25
static_delta = 0.1
p_1 = p_0+static_delta
static_tau = 0.1

reps=1
x=40                                # Number of unique values sampled
n=x*reps

tau = np.empty(n)
taulin = np.linspace(0.01,0.4,x)

reg = False                         # Flag for experiment 3

x2=15                                # Number of unique values sampled
n2=x2*reps

varying_p1 = np.empty(n2)
delta = np.geomspace(1e-3, 0.5, x2)


y=0
for i in range(x):
    tau[y:y+reps]= taulin[i]
    y=y+reps

y=0
for i in range(x2):
    varying_p1[y:y+reps]=delta[i]+p_0  
    y=y+reps      

# Record keeping
exp1_matrix_to_save=np.empty((n, 6))
exp2_matrix_to_save=np.empty((n2, 6))
exp1_grid_UCB=np.empty((n, 3))
exp2_grid_UCB=np.empty((n2, 3))

if __name__ == "__main__":
    
    cpu = cpu_count()
    print("Running on ", cpu, " cores")
    start = timer()

    entropy = 42
    seed_sequence = np.random.SeedSequence(entropy)
    seeds = seed_sequence.spawn(n)

    # with Pool(cpu) as pool:
    #     exp1_left, exp1_GUCB = zip(*pool.starmap(run, zip(range(n), seeds)))

    # pool.close()
    # pool.join()

    # s = seeds[0].spawn(n2)
    # with Pool(cpu) as pool2:
    #     exp2_left, exp2_GUCB = zip(*pool2.starmap(run2, zip(range(n2), s)))
    #     # exp2_GUCB = pool2.starmap(run2, zip(range(n2), s))

    # pool2.close()
    # pool2.join()

    # for i in range(n):
    #     exp1_grid_UCB[i,:] = exp1_GUCB[i]
    #     exp1_matrix_to_save[i,:] = exp1_left[i]

    # for i in range(n2):   
    #     exp2_matrix_to_save[i,:] = exp2_left[i]
    #     exp2_grid_UCB[i,:] = exp2_GUCB[i]

    reg = True

    # Delta (.425) > tau (.3)
    static_tau = 0.3
    p_1 = 0.625

    s = seeds[1].spawn(reps)
    with Pool(cpu) as pool3:
        left_reg, reg_GUCB = zip(*pool3.starmap(run3, zip(range(reps), s)))

    pool3.close()
    pool3.join()

    end = timer()

    print(f'elapsed time: {end - start}')

    # np.savetxt("exp1_grid_UCB.csv", exp1_grid_UCB, delimiter=',',header='tau, best arm, UCB regret') 
    # np.savetxt("exp1_saveddata.csv",exp1_matrix_to_save, delimiter=',',header='tau, best arm, Leftist regret, NSB regret, UCB regret, total regret')
    # np.savetxt("exp2_saveddata.csv",exp2_matrix_to_save, delimiter=',',header='tau, best arm, Leftist regret, NSB regret, UCB regret, total regret')
    # np.savetxt("exp2_grid_UCB.csv",exp2_grid_UCB, delimiter=',',header='tau, best arm, UCB regret')
    np.savetxt("reg_UCB.csv", reg_GUCB, header = 'Cumulative regret') 
    np.savetxt("reg_left.csv", left_reg, header = 'Cumulative regret')  
