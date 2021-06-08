import numpy as np
import tools

def EGM (sol,h,k,t,par): 

    # Prepare
    epsi = np.tile(par.epsi,par.Na)
    k = np.tile(k, par.Na*par.Nw)
    a = np.repeat(par.grid_a[t],par.Nw) 
    w = np.tile(par.epsi_w,(par.Na,1))
    
    # Income/transfers
    wage = par.kappa*k*epsi
    S = par.rho*k

    # Next period states
    k_plus = (1-par.delta)*k+par.phi1*par.hlist[h]**par.phi2
    
    if t < par.To:
        m_plus = (1+par.r)*a + par.hlist[h]*wage
    if t >= par.Tp:
        if h == 0:
            m_plus = (1+par.r)*a + par.P + S
        else:
            m_plus = (1+par.r)*a + par.hlist[h]*wage   
    else:
        if h == 0:
            m_plus = (1+par.r)*a + S
        else:
            m_plus = (1+par.r)*a + par.hlist[h]*wage

    # Value, consumption, marg_util
    shape = (6,m_plus.size)
    v_plus = np.nan+np.zeros(shape)
    c_plus = np.nan+np.zeros(shape)
    marg_u_plus = np.nan+np.zeros(shape)
    

    for i in range(6): #Range over working full-time, part-time and not working next period
        # Choice specific value
        v_plus[i,:] = tools.interp_2d_vec(par.grid_m,par.grid_k,sol.v[t+1,i], m_plus, k_plus)
    
        # Choice specific consumption    
        c_plus[i,:] = tools.interp_2d_vec(par.grid_m,par.grid_k,sol.c[t+1,i], m_plus, k_plus)
       
        # Choice specific Marginal utility
        marg_u_plus[i,:] = marg_u(c_plus[i,:], par) 
       
    # Expected value
    V_plus, prob = logsum(v_plus[0],v_plus[1],v_plus[2],v_plus[3],v_plus[4],v_plus[5],par.sigma_epsilon) 
    w_raw = w*np.reshape(V_plus,(par.Na,par.Nw))
    w_raw = np.sum(w_raw,1)
    marg_u_plus = prob[0,:]*marg_u_plus[0] + prob[1,:]*marg_u_plus[1] + prob[2,:]*marg_u_plus[2] + prob[3,:]*marg_u_plus[3] + prob[4,:]*marg_u_plus[4] + prob[5,:]*marg_u_plus[5]

    #Expected  average marg. utility
    avg_marg_u_plus = w*np.reshape(marg_u_plus,(par.Na,par.Nw))
    avg_marg_u_plus = np.sum(avg_marg_u_plus ,1)
    

    # raw c, m and v
    c_raw = inv_marg_util(par.beta*(1+par.r)*avg_marg_u_plus,par)
    m_raw = c_raw + par.grid_a[t,:]
   
    # Upper Envelope
    c,v = upper_envelope(t,h,c_raw,m_raw,w_raw,par)
    
    return c,v




def upper_envelope(t,h,c_raw,m_raw,w_raw,par):
    
    # Add a point at the bottom
    c_raw = np.append(1e-6,c_raw)  
    m_raw = np.append(1e-6,m_raw) 
    a_raw = np.append(0,par.grid_a[t,:]) 
    w_raw = np.append(w_raw[0],w_raw)

    # Initialize c and v   
    c = np.nan + np.zeros((par.Nm))
    v = -np.inf + np.zeros((par.Nm))
    
    # Loop through the endogenous grid
    size_m_raw = m_raw.size
    for i in range(size_m_raw-1):    

        c_now = c_raw[i]        
        m_low = m_raw[i]
        m_high = m_raw[i+1]
        c_slope = (c_raw[i+1]-c_now)/(m_high-m_low)
        
        w_now = w_raw[i]
        a_low = a_raw[i]
        a_high = a_raw[i+1]
        w_slope = (w_raw[i+1]-w_now)/(a_high-a_low)


        # Loop through the common grid
        for j, m_now in enumerate(par.grid_m):

            interp = (m_now >= m_low) and (m_now <= m_high) 
            extrap_above = (i == size_m_raw-2) and (m_now > m_high)

            if interp or extrap_above:
                # Consumption
                c_guess = c_now+c_slope*(m_now-m_low)
                
                # post-decision values
                a_guess = m_now - c_guess
                w = w_now+w_slope*(a_guess-a_low)
                
                # Value of choice
                v_guess = util(c_guess,h,par)+par.beta*w
                
                # Update
                if v_guess >v[j]:
                    v[j]=v_guess
                    c[j]=c_guess

    return c,v


# FUNCTIONS
def util(c,h,par):
    return (c**(1.0-par.zeta)-1)/(1.0-par.zeta)-(par.b*par.hlist[h]**par.alpha)/par.alpha
            
def inv_marg_util(u,par):
    return u**(-1/par.zeta)

def marg_u(c, par):
    return c**(-par.zeta)


def logsum(v1,v2,v3,v4,v5,v6,sigma):

    # setup
    V = np.array([v1, v2, v3, v4, v5, v6])

    # Maximum over the discrete choices
    mxm = V.max(0)

    # check the value of sigma
    if abs(sigma) > 1e-10:

        # numerically robust log-sum
        log_sum = mxm + sigma*(np.log(np.sum(np.exp((V - mxm) / sigma),axis=0)))
    
        # d. numerically robust probability
        prob = np.exp((V- log_sum) / sigma)    

    else: # No smmothing --> max-operator
        id = V.argmax(0)    #Index of maximum
        log_sum = mxm
        prob = np.zeros((v1.size*6))
        I = np.cumsum(np.ones((v1.size,1)))+id*(v1.size)-1
        I = I.astype(int)  # change type to integer
        prob[I] = 1

        prob = np.reshape(prob,(6,v1.size),'A')

    return log_sum,prob

