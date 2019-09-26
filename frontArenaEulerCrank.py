# A simple prop FD local vola solver

# FiniteDifferencePropExample

import acm
import FiniteDifferenceLVSpecificFuncs

def GetLocalVol(volInfo, isCall, forwardArray, strikeArray, timeGrid, valuationDate, skewExpiries):
    """Python method for retrieving local volatility matrix to have more control over the input parameters"""
    if timeGrid.Size() == 0:
        return acm.FRealMatrix()
   
    return volInfo.LocalVolatilityMatrix(isCall, forwardArray, strikeArray, timeGrid, 0.001, valuationDate, skewExpiries)

'''
The below functions that re commented out have been moved to another 
script. They are only shown here for completeness. 
#--------------------------------------------- 
# Calculate the time step size 
#---------------------------------------------
def time_step(t_exp, n_steps):
    return t_exp / n_steps

#--------------------------------------------- 
# Calculate the strike array from 0 to 4*K.
#---------------------------------------------

def construct_strike_array(strike, n_steps):
    ret_array = [0.001]
    step = strike * 4.0 / n_steps
    for j in range(1, n_steps+1):
        ret_array.append(j * step)
    
    return ret_array
    
    
def construct_strike_array_centered(S0, strike, n_steps):
    dS = strike * 4.0/(n_steps-1)
    S  = [dS*i for i in range(n_steps)]
        
    # find smallest distance to spot
    distLast = 9e99
    distI    = 0

    for i in range(n_steps):
        dist = (S0 - S[i])
        if (0 <= dist and dist < distLast):
            distLast = dist
            distI    = i
  
    # add dsitLast to all points
    if distLast > 0:
        for i in range(n_steps):
            S[i] = S[i] + distLast
    else:
        for i in range(n_steps-1):
            S[i]=S[i+1]
        S.append(S[len(S)-1] + dS)
        distI += 1

    return [S, distI]


#--------------------------------------------- 
# Calculate the time array (the time-points) 
#--------------------------------------------- 
def construct_time_array(t_exp, n_steps):
    if t_exp <= 0:
        return []
    ret_array = []
    step = float(t_exp) / float(n_steps)
    for i in range(n_steps + 1):
        ret_array.append(i * step)
    return ret_array
'''

#--------------------------------------------- 
# Find the index of the strike K
#--------------------------------------------- 
def get_strike_index(s_array, goal_strike):
    count = 0
    for k in s_array:
        if k > goal_strike:
            return count
        count += 1
    
#--------------------------------------------- 
# Calculate a dummy array of constant local volatilities
# with one value for each strike point
#--------------------------------------------- 
def dummy_loc_vol_array(s_index, sigma):
    r_array = []
    for j in range(s_index):
        r_array.append(sigma)
    return r_array
    
#--------------------------------------------- 
# Calculate a dummy matrix of constant local volatilities
# with one value for each strike and time point
#---------------------------------------------
def create_dummy_loc_vol(t_index, s_index, sigma):
    lv_matrix = []
    for i in range(t_index):
        lv_matrix.append(dummy_loc_vol_array(s_index, sigma))
    return lv_matrix

#---------------------------------------------
# Pay-off function - These can be simplified
#---------------------------------------------
def payoff_function(s, k, is_call):   
    if is_call == True:
        return s - k  # return s - k.Number()
    else:
        return k - s # return k.Number() - s
        
#---------------------------------------------
# Helper functions: construct result and wrapper, 
# note that the wrapper is defined in FCustomFunctions 
#--------------------------------------------- 
def construct_result_dict(result,instrumentSpotDate, unit ):
    dict = {}
    dict['result'] = acm.DenominatedValue(result, unit ,instrumentSpotDate)
    return dict
 
# wrapper
def fin_diff_wrapper(underlying_price, strike_array, strike, is_call, interest_rate, dividends, local_vol_matrix, dividendPV, time_size, time_step, instrumentSpotDate):
    # Use this for a flat vola surface
    #local_vol_matrix = create_dummy_loc_vol(time_size, len(strike_array), 0.4)
    res = FD_solver(underlying_price.Number() - dividendPV, strike_array, strike, is_call, interest_rate, dividends, local_vol_matrix, time_size, time_step)
    return construct_result_dict(res, instrumentSpotDate, underlying_price.Unit())
    
'''  ---------------------------------------------------------------------------------------------------
'''
#--------------------------------------------- 
# Creates the FD matrix A in u_t = Au
#---------------------------------------------
def create_FD_matrix(local_vol_matrix, carry_cost, interest_rate, nInner, number_of_timesteps):
    
    s_Index = range(1,nInner+1)
    
    alpha = []
    beta  = []
    gamma = []
    
    for i in range(number_of_timesteps):
        tempA = []
        tempB = []
        tempG = []
        for j in s_Index: 
            sigma = local_vol_matrix[i][j]
            tempA.append(0.5*(sigma*sigma*j*j-carry_cost*j))
            tempB.append(-sigma*sigma*j*j - interest_rate)
            tempG.append(0.5*(sigma*sigma*j*j+carry_cost*j))
        alpha.append(tempA)
        beta.append(tempB)
        gamma.append(tempG)

    # Handle the boundary conditions using a linear approximation near the boundary, i.e. d2f/dx2 = 0
    # => F0 = 2F1 - F2 at lower boundary and Fn = 2Fn-1 - Fn-2 at the upper boundary
    for i in range(number_of_timesteps):
        
        #at lower boundary
        beta[i][0] = beta[i][0] + 2.0*alpha[i][0]
        gamma[i][0] = gamma[i][0] - alpha[i][0]
        #at upper boundary
        endIndex = len(s_Index)-1
        
        alpha[i][endIndex] = alpha[i][endIndex] - gamma[i][endIndex]
        beta[i][endIndex] = beta[i][endIndex] + 2.0*gamma[i][endIndex]
    
    return alpha, beta, gamma
    
def create_initial_data(strike_array, strike, is_call):
    payoff = range(len(strike_array))
    N = len(strike_array)
    for i in range(N):  
        if is_call == True:
            payoff[i]=max(strike_array[i]-strike,0.0)
        else:
            payoff[i]=max(strike-strike_array[i],0.0)
    # return only the "inner" part of the pay off vector, i.e. not the boundary points        
    return payoff[1:N-1]



# The two following *_step functions are helper functions.    
def euler_forward_step(alpha, beta, gamma, Fold, time_step, time_pos):        
    N    = len(Fold)
    Fnew = range(N)
    
    for i in range(N):
        Fnew[i] = 0.0
         
    # Do first row separately    
    Fnew[0] = Fold[0] + time_step*(beta[time_pos][0]*Fold[0] + gamma[time_pos][0]*Fold[1])  
    
    # Do all inner values
    for j in range(1,N-1):            
        Fnew[j] = Fold[j] + time_step*(alpha[time_pos][j]*Fold[j-1] + beta[time_pos][j]*Fold[j] + gamma[time_pos][j]*Fold[j+1])
        
    # Do last row separately  
    Fnew[N-1] = Fold[N-1] + time_step*(alpha[time_pos][N-1]*Fold[N-2] + beta[time_pos][N-1]*Fold[N-1])
         
    return Fnew

def euler_forward(alpha, beta, gamma, Fold, number_of_timesteps, time_step):        
    N    = len(Fold)
    Fnew = range(N)
    
    for i in range(N):
        Fnew[i] = 0.0
       
    for i in range(number_of_timesteps): 
       
        # Do first row separately    
        Fnew[0] = Fold[0] + time_step*(beta[i][0]*Fold[0] + gamma[i][0]*Fold[1])  
        
        # Do all inner values
        for j in range(1,N-1):            
            Fnew[j] = Fold[j] + time_step*(alpha[i][j]*Fold[j-1] + beta[i][j]*Fold[j] + gamma[i][j]*Fold[j+1])
            
        # Do last row separately  
        Fnew[N-1] = Fold[N-1] + time_step*(alpha[i][N-1]*Fold[N-2] + beta[i][N-1]*Fold[N-1])
             
        #copy new to old
        for jj in range(N):
            Fold[jj] = Fnew[jj]
    
    return Fnew
    
def euler_backward(alpha, beta, gamma, Fold, number_of_timesteps, time_step):
    N         = len(Fold)
    Fnew      = range(N)
    alpha_tmp = range(N)
    beta_tmp  = range(N)
    gamma_tmp = range(N)
        
    
    for i in range(number_of_timesteps):
        
        for j in range(N):
            
            alpha_tmp[j] = -time_step*alpha[i][j]
            beta_tmp[j]  = 1.0-time_step*beta[i][j]
            gamma_tmp[j] = -time_step*gamma[i][j]
        #print i,Fold
        Fnew = thomas_algorithm(alpha_tmp, beta_tmp, gamma_tmp, Fold)
        
        #copy new to old
        for j in range(N):
            Fold[j] = max(Fnew[j],0.0) 
            
    for j in range(N):
        Fnew[j] = max(Fnew[j],0.0)
         
    return Fnew


def euler_backward_step(alpha, beta, gamma, Fold, time_step, time_pos):
    N         = len(Fold)
    Fnew      = range(N)
    alpha_tmp = range(N)
    beta_tmp  = range(N)
    gamma_tmp = range(N)
        
    
    for j in range(N):        
        alpha_tmp[j] = -time_step*alpha[time_pos][j]
        beta_tmp[j]  = 1.0-time_step*beta[time_pos][j]
        gamma_tmp[j] = -time_step*gamma[time_pos][j]
    
    Fnew = thomas_algorithm(alpha_tmp, beta_tmp, gamma_tmp, Fold)
    
    for j in range(N):
        Fnew[j] = max(Fnew[j],0.0)
         
    return Fnew
    
    
#----------------------------------------- 
# Thomas algorithm for a tridigonal matrix
def thomas_algorithm(a,b,c,d):
    
    #solves a tridiagonal matrix system [a,b,c]=d
        
    N = len(d)
    endIndex = N-1
    res = range(N)
    
    #forward sweep      
    c[0] = c[0]/b[0]
    d[0] = d[0]/b[0]
    
    for i in range(1,endIndex+1):
        #print 'fs', i
        temp = 1.0/(b[i]-c[i-1]*a[i])
        c[i] = c[i]*temp
        d[i] = (d[i]-d[i-1]*a[i])*temp
       
    # backward sweep
    res[endIndex] = d[endIndex]
    
    for i in range(endIndex-1,-1,-1):
        #print 'bs', i
        res[i] = d[i]-c[i]*res[i+1]
    
    return res
   
def extract_solution_to_boundaries(F):
    N = len(F)
    Nall = N+2
    Fall = range(Nall)
    #print N, Nall
    for i in range(Nall):
        Fall[i] = 0.0
        
    # Use boundary conditions to extract the solution to the boundaries
    Fall[0]      = max(2.0*F[0] - F[1],0.0)
    Fall[Nall-1] = max(2.0*F[N-1] - F[N-2],0.0)
    
    # Copy data from F to Fall (with 0.0 at the boundaries)    
   
    for i in range(1,Nall-1):
        Fall[i] = F[i-1]
   
    return Fall
    
#--------------------------------------------- 
# Interpolate in the FD solution for the spot price (goal_strike)
#--------------------------------------------- 
def get_result_from_grid(s_array, grid, goal_strike):
    ind = get_strike_index(s_array, goal_strike)
    s1 = s_array[ind - 1]
    s2 = s_array[ind]
    f1 = grid[ind - 1]
    f2 = grid[ind]
    val = f1 + (goal_strike - s1)*( (f2-f1)/(s2-s1))
    return val

def Crank_Nicolson(alpha, beta, gamma, F0, number_of_timesteps, time_step):
    #Our function
    Fold=F0
    for i in range(number_of_timesteps-1):
        Fnew = euler_forward_step(alpha, beta, gamma, Fold, time_step/2, i)
        Fold = euler_backward_step(alpha, beta, gamma, Fnew, time_step/2, i+1)
    return Fold

def FD_solver(underlying_price, price_array, strike, is_call, interest_rate, dividends, local_vol_matrix, number_of_timesteps, time_step):
    nInner = len(price_array)-2
    """
    # create the approximation matrix _including_ the boundary conditions, i.e NOT the full FD matrix incl. time-step etc.
    alpha, beta, gamma = create_FD_matrix(local_vol_matrix, interest_rate, interest_rate, nInner, number_of_timesteps)
    
    # create the initial condition (final condition)
    F0    = create_initial_data(price_array, strike, is_call)
       
    # Do the time stepping (Euler Forward or backward or Crank-Nicolson)
    
    Fni   = euler_forward(alpha, beta, gamma, F0, number_of_timesteps, time_step)
    Falli = extract_solution_to_boundaries(Fni)
    res   = get_result_from_grid(price_array, Falli, underlying_price)
    
    print "Explicit:", res

    alpha, beta, gamma = create_FD_matrix(local_vol_matrix, interest_rate, interest_rate, nInner, number_of_timesteps)
    F0    = create_initial_data(price_array, strike, is_call)

    Fni   = euler_backward(alpha, beta, gamma, F0, number_of_timesteps, time_step)
    Falli = extract_solution_to_boundaries(Fni)
    res   = get_result_from_grid(price_array, Falli, underlying_price)
    
    print "Implicit:", res
    """
    alpha, beta, gamma = create_FD_matrix(local_vol_matrix, interest_rate, interest_rate, nInner, number_of_timesteps)
    F0    = create_initial_data(price_array, strike, is_call)
    
    Fni   = Crank_Nicolson(alpha, beta, gamma, F0, number_of_timesteps, time_step)
    Falli = extract_solution_to_boundaries(Fni)
    res   = get_result_from_grid(price_array, Falli, underlying_price)
    
    
    #print "Result      :", res
    print"wrong code"
    return res
