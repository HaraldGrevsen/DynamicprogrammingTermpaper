# import packages 
import DCEGM as DCEGM 
import numpy as np
import tools as tools
from types import SimpleNamespace
import math

    # Constructing Gauss_hermite for the wage shocks
def gauss_hermite(n):
    # a. calculations
    i = np.arange(1,n)
    a = np.sqrt(i/2)
    CM = np.diag(a,1) + np.diag(a,-1)
    L,V = np.linalg.eig(CM)
    I = L.argsort()
    V = V[:,I].T
    # b. nodes and weights
    x = L[I]
    w = np.sqrt(math.pi)*V[:,0]**2

    return x,w
def GaussHermite_lognorm(sigma,n):
    x, w = gauss_hermite(n)
    x = np.exp(x*math.sqrt(2)*sigma - 0.5*sigma**2)
    w = w / math.sqrt(math.pi)
    # assert a mean of one
    assert(1 - np.sum(w*x) < 1e-8 ), 'The mean in GH-lognorm is not 1'
    return x, w

class TheModel():
    def __init__(self,name=None):
        """ defines default attributes """
        # a. name
        self.par = SimpleNamespace()
        self.sol = SimpleNamespace() 
        self.sim = SimpleNamespace() 
    ############
    # setup    #
    ############

    def setup(self):
        par = self.par
        par.T = 60  # Number for years, from 20 to 80
        par.Ts = 45  # Number of years before public pension payouts
        par.To = 42  # Number of years before occupational pension payouts
    
        # Our Model parameters
        par.zeta = 0.8
        par.beta = 0.95
        par.rho = 0.1
        par.b = 0.8
        par.phi1 = 1.1
        par.phi2 = 1.1
        par.alpha = 1.8
        par.sigma_xi = 0.1   # THIS IS FOR WAGE (MIGHT BE CALLED SIGMA_W)
        par.sigma_epsilon = 0.3
        par.kappa = 0.1
        par.r = 0.04
        par.P = 1
        
        # Grids and numerical integration for our 2 state variables
        par.a_max = 100
        par.a_phi = 1.1
        par.Na  = 150
        par.k_max = 60
        par.k_phi = 1.1
        par.Nk  = 150
        par.m_max = 100
        par.m_phi = 1.1
        par.Nm  = 150
        #points in Gauss_Hermite
        par.Nxi = 4
        par.Nm_b = 50
        
        
        #Grid for different h
        par.h = [0,0.5,1]

    def create_grids(self):

        par = self.par
        # Check parameters
        assert (par.rho >= 0), 'not rho > 0'
        # need more checks?

        # Shocks for wage, this gives us the shock and weight!
        par.xi,par.xi_w = GaussHermite_lognorm(par.sigma_xi,par.Nxi)
        
        # Setting up grids 
        # We set up a grid for A, which is the exogeneously fixed monotonic grid over savings
        # End pf period assets
        par.grid_a = np.nan + np.zeros([par.T,par.Na])
        for t in range(par.T):
            par.grid_a[t,:] = tools.nonlinspace(0+1e-8,par.a_max,par.Na,par.a_phi)
        
        # We need a grid for human capital (K)
        par.grid_k =  tools.nonlinspace(0+1e-4,par.k_max,par.Nk,par.k_phi)
        #par.grid_k = np.nan + np.zeros([par.T,par.Nk])
        #for t in range(par.T):
        #    par.grid_k[t,:] = tools.nonlinspace(0+1e-8,par.k_max,par.Nk,par.k_phi)
        
        #Grid for m?
        #par.grid_m = np.nan + np.zeros([par.T,par.Nm])
        #for t in range(par.T):
        #    par.grid_m[t,:] = tools.nonlinspace(0+1e-8,par.m_max,par.Nm,par.m_phi)
        par.grid_m =  np.concatenate([np.linspace(0+1e-6,1-1e-6,par.Nm_b), tools.nonlinspace(1+1e-6,par.m_max,par.Nm-par.Nm_b,par.m_phi)])
        #par.grid_m =  tools.nonlinspace(0+1e-4,par.m_max,par.Nm,par.m_phi)
        # Set seed
        np.random.seed(2021)
                
    def solve(self):
        # Initialize
        par = self.par
        sol = self.sol

        shape=(par.T,par.Na,par.Nk,3,par.Nxi)
        sol.m = np.nan+np.zeros(shape)
        sol.c = np.nan+np.zeros(shape)
        sol.v = np.nan+np.zeros(shape)


        #Last period, consume all
        #for h in range(3):
        #    sol.m[par.T-1,h,:] = par.grid_m
        #    sol.c[par.T-1,h,:] = par.grid_m
        #    sol.v[par.T-1,h,:] = DCEGM.util(sol.c[par.T-1,h,:],h,par)

        # In our last period the agent will consume all!
        # Last period, (= consume all) 
        for i_a in range(par.Na):
            for i_k in range(par.Nk):
                for i_h in range(3):
                    #Her forsøg uden P og S!
                    #m = par.grid_a[i_a] * (1+par.r)  + par.h[i_h] * par.kappa * par.grid_k[i_k] * par.xi + par.P + par.rho*par.grid_k[i_k]
                    #m = par.grid_a[par.T-1,i_a] * (1+par.r)  + par.h[i_h] * par.kappa * par.grid_k[i_k]*par.xi + par.P + par.rho*par.grid_k[i_k]
                    sol.m[par.T-1,i_a,i_k,i_h] = par.grid_m
                    sol.c[par.T-1,i_a,i_k,i_h] = par.grid_m
                    sol.v[par.T-1,i_a,i_k,i_h] = DCEGM.util(sol.c[par.T-1,i_a,i_k,i_h],0,par)

                #sol.m[par.T-1,i_m,h] = par.grid_m[i_m,:]
                #sol.c[par.T-1,i_a,i_k,h] = par.grid_m[i_m,:]
                #sol.v[par.T-1,i_a,i_k,h] = egm.util(sol.c[par.T-1,i_m,h],h,par) 
                #SOLVE USING EGM:
                #[c, v, m] = model.EGM(par.T-1,h,k,par)   
                #sol.m[par.T-1,i_a,i_k,i_h] = m
                #sol.c[par.T-1,i_a,i_k,i_h] = c
                #sol.v[par.T-1,i_a,i_k,i_h] = v
                #=egm.util(sol.c[par.T-1,i_a,i_k,i_h],i_h,par)
        #sol.m_logsum = DCEGM.logsum(sol.m[par.T-1,i_a,i_k,0],sol.m[par.T-1,i_a,i_k,1],sol.m[par.T-1,i_a,i_k,2],par.sigma_epsilon)
                    #m=par.grid_a[i_a,:]*(1+par.r)+i_h*par.kappa*par.grid_k[i_k]*xi+par.P+par.rho*par.grid_k[i_k]
                    #Her forsøg uden P og S!
                    #m = par.grid_a[i_a,:] * (1+par.r) + i_h * par.kappa * par.grid_k[i_k,:] * par.xi                 
        #Before last period
        #for t in range(par.T-2,-1,-1): #range(start, stop, step)
        #    for i_a in range(par.Na):            
        #        for i_k in range(par.Nk):
        #        #INTERPOLATE?
        #            # Solve model with EGM
        #            c,v,m = DCEGM.DCEGM_(sol,h,k,t,par)
        #            sol.m[par.t,i_a,i_k] = m
        #            sol.c[par.t,i_a,i_k] = c
        #            sol.v[par.t,i_a,i_k] = v                        
    
    def solve22(self):
        # Initialize
        par = self.par
        sol = self.sol
        xi = np.tile(par.xi,par.Na)

        shape=(par.T,par.Na,par.Nk,3,par.Nxi)
        sol.m = np.nan+np.zeros(shape)
        sol.c = np.nan+np.zeros(shape)
        sol.v = np.nan+np.zeros(shape)
        
        # Last period, (= consume all) 
        for i_a in range(par.Na):
            for i_k in range(par.Nk):
                for i_h in range(3):
                    m=par.grid_a[i_a,:]*(1+par.r)+i_h*par.kappa*par.grid_k[i_k]*xi+par.P+par.rho*par.grid_k[i_k]
                    sol.m[par.T-1,i_a,i_k,i_h] = m
                    sol.c[par.T-1,i_a,i_k,i_h] = m
                    sol.v[par.T-1,i_a,i_k,i_h] = egm.util(sol.c[par.T-1,i_a,i_k,i_h],i_h,par)

        # Before last period
        for t in range(par.T-2,-1,-1):
            for i_a in range(par.Na):
                #Choice specific function
                for i_k, k in enumerate(par.grid_k):
                    for i_h in range(3):
                        # Solve model with EGM
                        c,v = egm.EGM(sol,i_h,k,t,par)
                        sol.c[t,z_plus,:,i_p] = c
                        sol.v[t,z_plus,:,i_p] = v