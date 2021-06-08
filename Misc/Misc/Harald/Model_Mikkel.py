# import packages 
import numpy as np
import tools
from types import SimpleNamespace
import DC_EGM as egm

class model_dc_multidim():

    def __init__(self,name=None):
        """ defines default attributes """

        # a. name
        self.par = SimpleNamespace()
        self.sol = SimpleNamespace() 
        self.sim = SimpleNamespace() 
        
        
    def setup(self):

        par = self.par

        par.T = 60
        par.T_s=65
        par.T_o= par.T_s-3
        # Model parameters
        
        par.sigma_eps=0.3
        par.beta=0.95
        par.r=0.04
        par.phi=1
        par.kappa=0.1
        par.rho=0.1
        par.zeta=0.8
        par.b=0.5
        par.alpha=1
        par.sigma_w=1
        par.P=1
        
        par.h=[0, 0.5, 1]
        
        # Grids and numerical integration
        par.a_max = 150
        par.a_phi = 1.1  # Curvature parameters
        par.k_max = par.phi*par.T
        par.k_phi = 1.1 # Curvature parameters

        par.Nxi = 8
        par.Na = 150
        par.Nk = 150

        par.Nm_b = 50
        
    def create_grids(self):

        par = self.par

        # Check parameters
        #assert (par.rho >= 0), 'not rho > 0'
        #indsæt evt. andre parameter krav som stående herover
        
        # Shocks
        par.xi,par.xi_w = tools.GaussHermite_lognorm(par.sigma_w,par.Nxi)
        
        # End of period assets
        par.grid_a = np.nan + np.zeros([par.T,par.Na])
        for t in range(par.T):
            par.grid_a[t,:] = tools.nonlinspace(0+1e-6,par.a_max,par.Na,par.a_phi)

        # Human capital
        par.grid_k = tools.nonlinspace(0+1e-4,par.k_max,par.Nk,par.k_phi)

        # Set seed
        np.random.seed(2021)
        
        
        
    def solve(self):
        
        # Initialize
        par = self.par
        sol = self.sol

        shape=(par.T,par.Na,par.Nk,3,par.Nxi)#So the nestorder of our array are as follows:Time,A,K,h,eps_w
        sol.m = np.nan+np.zeros(shape)
        sol.c = np.nan+np.zeros(shape)
        sol.v = np.nan+np.zeros(shape)
        
        # Last period, (= consume all) 
        for i_a in range(par.Na): #i_A are corresponding to index in A grid 
            for i_k in range(par.Na): #i_k are corresponding to index in K grid
                for i_h in range(3): #i_h are corresponding to index in par.h
                    m=par.grid_a[i_a]*(1+par.r)+par.h[i_h]*par.kappa*par.grid_k[i_k]*par.xi+par.P+par.rho*par.grid_k[i_ks]
                    sol.m[par.T-1, i_a, i_k, i_h] = m
                    sol.c[par.T-1, i_a, i_k, i_h] = m
                    sol.v[par.T-1, i_a, i_k, i_h] = egm.util(sol.c[par.T-1,z_plus,:,i_p],z_plus,par) #skal læige have rette til
                

        # Before last period
        for t in range(par.T-2,-1,-1):

            #Choice specific function
            for i_p, p in enumerate(par.grid_p):
            
                for z_plus in range(2):

                    # Solve model with EGM
                    c,v = egm.EGM(sol,z_plus,p,t,par)
                    sol.c[t,z_plus,:,i_p] = c
                    sol.v[t,z_plus,:,i_p] = v