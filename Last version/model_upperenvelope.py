# import packages 
import numpy as np
import tools
from types import SimpleNamespace
import dc_egm_upperenvelope as egm


class model_u():

    def __init__(self,name=None):
        """ defines default attributes """

        # a. name
        self.par = SimpleNamespace()
        self.sol = SimpleNamespace() 
        self.sim = SimpleNamespace() 

    def setup(self):

        par = self.par

        par.T = 61  # Number for years, from 25 to 80
        par.Tp = 40  # Number of years before public pension payouts, age 65 
        par.To = 35  # Number of years before occupational pension payouts, age 60
        
        # Discrete choices
        par.hlist = [0,0.5,1]

        # Model parameters
        par.zeta = 0.79
        par.beta = 0.97
        par.rho = 0.2
        par.b = 2
        par.delta = 0.03
        par.phi1 = 0.2
        par.phi2 = 0.6
        par.alpha = 1.5
        par.sigma_w = 0.1 
        par.sigma_epsilon = 0.01
        par.kappa = 1
        par.r = 0.04
        par.P = 1.2

        # Grids and numerical integration
        par.m_max = 30
        par.m_phi = 1.1 # Curvature parameters
        par.a_max = 30
        par.a_phi = 1.1  # Curvature parameters
        par.k_max = 30
        par.k_phi = 1.1 # Curvature parameters

        par.Nw = 4
        par.Nm = 150
        par.Na = 150
        par.Nk = 150

        par.Nm_b = 10
        
        # Simulation
        par.m_start = 2.5 # initial m in simulation
        par.k_start = 2 #initial k in simulation
        par.simN = 100000 # number of persons in simulation
        par.simT = par.T # number of periods in simulation        
        

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
        np.random.seed(3)

    def solve(self):
        
        # Initialize
        par = self.par
        sol = self.sol

        shape=(par.T,3,par.Nm,par.Nk)
        sol.m = np.nan+np.zeros(shape)
        sol.c = np.nan+np.zeros(shape)
        sol.v = np.nan+np.zeros(shape)
        
        sol.c_raw=np.nan+np.zeros(shape)
        sol.m_raw=np.nan+np.zeros(shape)
        sol.v_raw=np.nan+np.zeros(shape)
        
        
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
                    c,v,c_raw,m_raw,v_raw = egm.EGM(sol,h,k,t,par,i_k)    
                    sol.c[t,h,:,i_k] = c
                    sol.v[t,h,:,i_k] = v
                    sol.c_raw[t,h,:,i_k]=c_raw
                    sol.m_raw[t,h,:,i_k]=m_raw
                    sol.v_raw[t,h,:,i_k]=v_raw
                    
                
    def simulate (self):

        par = self.par
        sol = self.sol
        sim = self.sim

        # Initialize
        shape = (par.simT, par.simN)
        sim.m = np.nan +np.zeros(shape)
        sim.k = np.nan +np.zeros(shape)
        sim.c = np.nan +np.zeros(shape)
        sim.h = np.nan +np.zeros(shape)
        sim.a = np.nan +np.zeros(shape)
        sim.s = np.zeros(shape)
        sim.p = np.zeros(shape)
        sim.wage = np.nan +np.zeros(shape)
        sim.disp = np.nan +np.zeros(shape)

        # Shocks used
        par.eps_w = np.random.lognormal(0,par.sigma_w,shape)
        par.eps_ts = np.random.rand(par.simT, par.simN)

        # Initial values
        sim.m[0,:] = par.m_start
        sim.k[0,:] = par.k_start
        
        shape_inter = (3,par.simN)
        v_interp = np.nan + np.zeros(shape_inter)
        c_interp = np.nan + np.zeros(shape_inter)
        
        for t in range(par.simT):
            #for i in range(3): #Range over working full-time, part-time and not working next period  [t,:]
                    # Choice specific value
            v_interp[0,:] = tools.interp_2d_vec(par.grid_m,par.grid_k,sol.v[t,0], sim.m[t,:], sim.k[t,:])
            v_interp[1,:] = tools.interp_2d_vec(par.grid_m,par.grid_k,sol.v[t,1], sim.m[t,:], sim.k[t,:])
            v_interp[2,:] = tools.interp_2d_vec(par.grid_m,par.grid_k,sol.v[t,2], sim.m[t,:], sim.k[t,:])
    
                    # Choice specific consumption    
            c_interp[0,:] = tools.interp_2d_vec(par.grid_m,par.grid_k,sol.c[t,0], sim.m[t,:], sim.k[t,:])
            c_interp[1,:] = tools.interp_2d_vec(par.grid_m,par.grid_k,sol.c[t,1], sim.m[t,:], sim.k[t,:])
            c_interp[2,:] = tools.interp_2d_vec(par.grid_m,par.grid_k,sol.c[t,2], sim.m[t,:], sim.k[t,:])
       
        
            # Probabilities
            _, prob = egm.logsum(v_interp[0],v_interp[1],v_interp[2],par.sigma_epsilon) 
            
            
            sim.prob = prob
            
            # Chooce h and the corresponding consumption policy function
            for n in range(par.simN):
                if par.eps_ts[t,n] <= prob[2,n]:
                    sim.h[t,n] = 1 
                    sim.c[t,n] = c_interp[2,n]
                elif par.eps_ts[t,n] <= prob[2,n]+prob[1,n] and par.eps_ts[t,n] > prob[2,n]:
                    sim.h[t,n] = 0.5
                    sim.c[t,n] = c_interp[1,n]
                else:
                    sim.h[t,n] = 0
                    sim.c[t,n] = c_interp[0,n]
                    if t >= par.Tp:
                        sim.p[t,n] = par.P
                        sim.s[t,n] = par.rho*sim.k[t,n]
                    elif t < par.To:
                        sim.p[t,n] = 0
                        sim.s[t,n] = 0
                    else:
                        sim.p[t,n] = 0
                        sim.s[t,n] = par.rho*sim.k[t,n]


            # Calculate wages, savings, total income and next period states (human capital and assets)
            sim.wage[t,:] = par.kappa * sim.k[t,:] * par.eps_w[t,:]
            sim.a[t,:] = sim.m[t,:]-sim.c[t,:]
            
            sim.disp[t,:] = sim.h[t,:]*sim.wage[t,:]+sim.s[t,:]+sim.p[t,:]

            
            if t < par.T-1:
                sim.k[t+1,:] = (1-par.delta)*sim.k[t,:]+par.phi1*sim.h[t,:]**par.phi2
                sim.m[t+1,:] = (1+par.r)*sim.a[t,:]+sim.h[t,:]*sim.wage[t,:]+sim.s[t,:]+sim.p[t,:]
                

