# import packages 
import numpy as np
import tools
from types import SimpleNamespace
import egm_dc_multidim as egm


class model_dc_multidim():

    def __init__(self,name=None):
        """ defines default attributes """

        # a. name
        self.par = SimpleNamespace()
        self.sol = SimpleNamespace() 
        self.sim = SimpleNamespace() 

    def setup(self):

        par = self.par

        par.T = 65  # Number for years, from 20 to 85
        par.Tp = 45  # Number of years before public pension payouts
        par.To = 42  # Number of years before occupational pension payouts
        
        # Discrete choices
        par.hlist = [0,0.5,1]

        # Model parameters
        par.zeta = 0.79
        par.beta = 0.95
        par.rho = 0.08
        par.b = 0.8
        par.phi1 = 0.2
        par.phi2 = 0.6
        par.alpha = 1.5
        par.sigma_w = 0.2   
        par.sigma_epsilon = 0.3
        par.kappa = 0.9
        par.r = 0.04
        par.P = 0.5

        # Grids and numerical integration
        par.m_max = 10
        par.m_phi = 1.1 # Curvature parameters
        par.a_max = 10
        par.a_phi = 1.1  # Curvature parameters
        #par.k_max = par.phi1*par.T
        par.k_max = 13
        par.k_phi = 1.1 # Curvature parameters

        par.Nw = 4
        par.Nm = 150
        par.Na = 150
        par.Nk = 150

        par.Nm_b = 50
        
        #Simulation
        par.m_ini = 2.5 # initial m in simulation
        par.k_ini = 2.5
        par.simN = 500000 # number of persons in simulation
        par.simT = 100 # number of periods in simulation
        par.simlifecycle = 0 # = 0 simulate infinite horizon model
        par.simT = par.T
        par.simlifecycle = 1

    def create_grids(self):

        par = self.par

        # Check parameters
        assert (par.rho >= 0), 'not rho > 0'

        # Shocks
        par.epsi,par.epsi_w = tools.GaussHermite_lognorm(par.sigma_w,par.Nw)
        
        # End of period assets
        par.grid_a = np.nan + np.zeros([par.T,par.Na])
        for t in range(par.T):
            par.grid_a[t,:] = tools.nonlinspace(0+1e-6,par.a_max,par.Na,par.a_phi)

        # Cash-on-hand
        par.grid_m =  np.concatenate([np.linspace(0+1e-6,1-1e-6,par.Nm_b), tools.nonlinspace(1+1e-6,par.m_max,par.Nm-par.Nm_b,par.m_phi)])    

        # Human capital
        par.grid_k = tools.nonlinspace(0+1e-4,par.k_max,par.Nk,par.k_phi)

        # Set seed
        np.random.seed(2020)

    def solve(self):
        
        # Initialize
        par = self.par
        sol = self.sol

        shape=(par.T,3,par.Nm,par.Nk)
        sol.m = np.nan+np.zeros(shape)
        sol.c = np.nan+np.zeros(shape)
        sol.v = np.nan+np.zeros(shape)
        
        # Last period, (= consume all) 
        
        for i_k in range(par.Nk):
            for h in range(3):
                sol.m[par.T-1,h,:,i_k] = par.grid_m
                sol.c[par.T-1,h,:,i_k] = par.grid_m
                sol.v[par.T-1,h,:,i_k] = egm.util(sol.c[par.T-1,h,:,i_k],h,par)

        # Before last period
        for t in range(par.T-2,-1,-1):

            #Choice specific function
            for i_k, k in enumerate(par.grid_k):
            
                for h in range(3):
                
                    # Solve model with EGM
                    c,v = egm.EGM(sol,par.hlist[h],k,t,par)
                    sol.c[t,h,:,i_k] = c
                    sol.v[t,h,:,i_k] = v
                
