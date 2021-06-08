############# HERE WE SIMULATE

    def simulate (self):

        par = self.par
        sol = self.sol
        sim = self.sim

        # Initialize
        shape = (par.simT, par.simN)
        #sim.m = par.m_ini + np.zeros(shape)
        #sim.k = par.k_ini +np.zeros(shape)
        sim.m = np.nan + np.zeros(shape)
        sim.k = np.nan + np.zeros(shape)
        sim.c = np.nan + np.zeros(shape)
        sim.h = np.nan + np.zeros(shape)

        # Shocks, for wage (w) and labour-choices (epsilon)
        shocki = np.random.choice(par.Nshocks,(par.T,par.simN),replace=True,p=par.epsi_w) #draw values between 0 and Nshocks-1, with probability w
        sim.epsi = par.epsi_vec[shocki]
        #draw T uniformly distributed variables uni:
        
        
        #check it has a mean of 1
        assert (abs(1-np.mean(sim.xi)) < 1e-4), 'The mean is not 1 in the simulation of xi'
        assert (abs(1-np.mean(sim.psi)) < 1e-4), 'The mean is not 1 in the simulation of psi'

        # Initial values
        sim.m[0,:] = par.m_ini
        sim.k[0,:] = par.k_ini


        for t in range(par.T):
            for h in range(3):
                sim.c[t,:] = tools.interp_2d_vec(sol.m[t,:],sol.k[t,:],sol.c[t,h], sim.m[t,:], sim.k[t,:])
           
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




                    V_plus, prob = logsum(v_plus[0],v_plus[1],v_plus[2],par.sigma_epsilon)
