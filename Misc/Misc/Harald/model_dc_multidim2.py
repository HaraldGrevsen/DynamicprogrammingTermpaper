# import packages 
import numpy as np
import tools
from types import SimpleNamespace
import egm_dc_multidim2 as egm


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
        par.rho = 0.1
        par.b = 1.2
        par.phi1 = 0.2
        par.phi2 = 0.6
        par.alpha = 1.5
        par.sigma_w = 0.1   # THIS IS FOR WAGE (MIGHT BE CALLED SIGMA_W)
        par.sigma_epsilon = 0.3
        par.kappa = 1
        par.r = 0.04
        par.P = 0.20

        # Grids and numerical integration
        par.m_max = 10
        par.m_phi = 1.1 # Curvature parameters
        par.a_max = 10
        par.a_phi = 1.1  # Curvature parameters
        par.k_max = 4
        par.k_phi = 1.1 # Curvature parameters

        par.Nw = 4
        par.Nm = 150
        par.Na = 150
        par.Nk = 150

        par.Nm_b = 10
        
        # Simulation
        par.m_start = 1.5 # initial m in simulation
        par.k_start = 0.5 #initial k in simulation
        par.simN = 10000 # number of persons in simulation
        par.simT = par.T # number of periods in simulation
        par.simlifecycle = 1 # = 0 simulate infinite horizon model
        
        

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
                    c,v = egm.EGM(sol,h,k,t,par)
                    sol.c[t,h,:,i_k] = c
                    sol.v[t,h,:,i_k] = v
                    
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
        sim.s = np.nan +np.zeros(shape)
        sim.p = np.nan +np.zeros(shape)
        sim.wage = np.nan +np.zeros(shape)
        
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

            for n in range(par.simN):
                if par.eps_ts[t,n] <= prob[2,t]:
                    sim.h[t,n] = 1 
                    sim.c[t,n] = c_interp[2,n]
                    sim.p[t,n] = 0
                elif par.eps_ts[t,n] <= prob[2,t]+prob[1,t] and par.eps_ts[t,n] > prob[2,t]:
                    sim.h[t,n] = 0.5
                    sim.c[t,n] = c_interp[1,n]
                    sim.p[t,n] = 0
                else:
                    sim.h[t,n] = 0
                    sim.c[t,n] = c_interp[0,n]
                    if t >= par.Tp-1:
                        sim.p[t,n] = par.P
                    else:
                        sim.p[t,n] = 0
            
            
            sim.wage[t,:] = par.kappa * sim.k[t,:] * par.eps_w[t,:]
            
            if t >= par.To-1:
                sim.s[t,:] = par.rho*sim.k[t,:]
            else:
                sim.s[t,:] = 0
            
            sim.a[t,:] = sim.m[t,:]-sim.c[t,:]
            
            if t < par.T-1:
                sim.k[t+1,:] = sim.k[t,:]+par.phi1*pow(sim.h[t,:],par.phi2)
                sim.m[t+1,:] = (1+par.r)*sim.a[t,:]+sim.h[t,:]*sim.wage[t,:]+sim.s[t,:]+sim.p[t,:]
        return sim


    def simulate2 (self):

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
        sim.s = np.nan +np.zeros(shape)
        sim.p = np.nan +np.zeros(shape)
        sim.wage = np.nan +np.zeros(shape)
        
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
            
            if par.eps_ts[t,:] <= prob[2,:]:
                F = 1
            if par.eps_ts[t,:] <= prob[2,:] +  prob[1,:]:
                E = 1
            if E==1 and F!=1:
                P = 1
            if E!= 1:
                NW = 1
            
            sim.c[t,F] = c_interp[2,:]
            sim.c[t,P] = c_interp[1,:]
            sim.c[t,NW] = c_interp[0,:]
            
            sim.h[t,F] = 1
            sim.h[t,P] = 0.5
            sim.h[t,NW] = 0
            
            sim.wage[t,:] = par.kappa * sim.k[t,:] * par.eps_w[t,:]
            
            if t >= par.To:
                sim.s[t,:] = par.rho*sim.k[t,:]
            else:
                sim.s[t,:] = 0
            
            if t >= par.Tp and sim.h[t,:]==0:
                sim.p[t,:] = par.P
            else:
                sim.p[t,:] = 0
            
            sim.a[t,:] = sim.m[t,:]-sim.c[t,:]
            
            if t < par.T:
                sim.k[t+1,:] = sim.k[t,:]+par.phi1*pow(sim.h[t,:],par.phi2)
                sim.m[t+1,:] = (1+par.r)*sim.a[t,:]+sim.h[t,:]*sim.wage[t,:]+sim.s[t,:]+sim.p[t,:]
        return sim           