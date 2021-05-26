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
        par.beta = 0.97
        par.rho = 0.1
        par.b = 1.2
        par.phi1 = 0.2
        par.phi2 = 0.6
        par.alpha = 1.5
        par.sigma_w = 0.1   # THIS IS FOR WAGE (MIGHT BE CALLED SIGMA_W)
        par.sigma_epsilon = 0.3
        par.kappa = 1
        par.r = 0.04
        par.P = 1

        # Grids and numerical integration
        par.m_max = 10
        par.m_phi = 1.1 # Curvature parameters
        par.a_max = 10
        par.a_phi = 1.1  # Curvature parameters
        par.k_max = 2
        par.k_phi = 1.1 # Curvature parameters

        par.Nw = 4
        par.Nm = 150
        par.Na = 150
        par.Nk = 100

        par.Nm_b = 50
        

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
                

    def simulate (self):

        par = self.par
        sol = self.sol
        sim = self.sim

        # Initialize
        shape = (par.simT, par.simN)
        sim.m = np.nan +np.zeros(shape)
        sim.c = np.nan +np.zeros(shape)
        sim.a = np.nan +np.zeros(shape)
        sim.p = np.nan +np.zeros(shape)
        sim.y = np.nan +np.zeros(shape)

        # Shocks
        shocki = np.random.choice(par.Nshocks,(par.T,par.simN),replace=True,p=par.w) #draw values between 0 and Nshocks-1, with probability w
        sim.psi = par.psi_vec[shocki]
        sim.xi = par.xi_vec[shocki]

            #check it has a mean of 1
        assert (abs(1-np.mean(sim.xi)) < 1e-4), 'The mean is not 1 in the simulation of xi'
        assert (abs(1-np.mean(sim.psi)) < 1e-4), 'The mean is not 1 in the simulation of psi'

        # Initial values
        sim.m[0,:] = par.sim_mini
        sim.p[0,:] = 0.0

        # Simulation 
        for t in range(par.simT):
            if par.simlifecycle == 0:
                sim.c[t,:] = tools.interp_linear_1d(sol.m[0,:],sol.c[0,:], sim.m[t,:])
            else:
                sim.c[t,:] = tools.interp_linear_1d(sol.m[t,:],sol.c[t,:], sim.m[t,:])
            
            sim.a[t,:] = sim.m[t,:] - sim.c[t,:]

            if t< par.simT-1:
                if t+1 > par.Tr: #after pension
                    sim.m[t+1,:] = (1+par.r)*sim.a[t,:]/(par.G*par.L[t])+1
                    sim.p[t+1,:] = np.log(par.G)+np.log(par.L[t])+sim.p[t,:]
                    sim.y[t+1,:] = sim.p[t+1,:]
                else:       #before pension
                    sim.m[t+1,:] = par.R*sim.a[t,:]/(par.G*par.L[t]*sim.psi[t+1,:])+sim.xi[t+1,:]
                    sim.p[t+1,:] = np.log(par.G)+np.log(par.L[t])+sim.p[t,:]+np.log(sim.psi[t+1,:])
                    sim.y[t+1,:] = sim.p[t+1,:]+np.log(sim.xi[t+1,:])
        
        #Renormalize 
        sim.P = np.exp(sim.p)
        sim.Y = np.exp(sim.y)
        sim.M = sim.m*sim.P
        sim.C = sim.c*sim.P
        sim.A = sim.a*sim.P
