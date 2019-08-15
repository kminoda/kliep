import numpy as np
import matplotlib.pylab as plt

class DensityRatioEstimator:
    """
    A class for conducting direct density-ratio estimation, or KLIEP.
    This is a non-parametric approach for solving change-point detection problem. 
    By computing likelihood ratio S, we can calculate whether any changes have occurred at a certain point.
    """
    def __init__(self, y, n_rf, n_te, k, eta=0.1, lam=0.1, mu=0.01):
        self.y = y
        self.n_rf = n_rf
        self.n_te = n_te
        self.k = k
        self.eta = eta
        self.lam = lam
        self.mu = mu
        
        self.t_rf = 0
        self.t_te = n_rf
        self.S = [np.nan]*(self.n_rf + self.n_te + self.k)
        
        self.Y = np.array([y[i:i+k,:] for i in range(n_rf+n_te)])
        self.Y_rf = self.Y[self.t_rf:self.t_te]
        self.Y_te = self.Y[self.t_te:]
        
        self.alpha = np.random.rand(n_te)*1
        self.eps = 0.01
        self.t = self.n_rf + self.n_te + self.k -1
        self.t_list = np.arange(0,self.t+1,1)
        
        self.sigma = self._KLIEP_model_selection()
        
        self.func = lambda x, y: np.exp(-np.sum((x-y)**2)/2/(self.sigma**2))
        
        self.K = self._get_K()
        self.b = self._get_b()
        
        self._batch_KLIEP() # TODO: ここを_KLIEP_model_selectionの行と統合したい。
        
        self.change_point_list = []
        
    def __call__(self,y):
        self.t += 1
        self.t_list = np.append(self.t_list,self.t)
        
        # if not just after any change point.
        if len(self.change_point_list)==0 or self.t > self.change_point_list[-1]+self.n_rf+self.n_te:
            self._update(y)
            S = self._compute_S()
            
            # if S is greater than threshold, regard this as change point
            if S > self.mu:
                self.S.append(np.nan)
                print('CHANGE OCCURRED!!')
                print('time:{0}'.format(self.t))
                self.change_point_list.append(self.t) # report the change point
                self._batch_KLIEP()
                
            # if S is smaller than threshold, not a change point
            else:
                self.S.append(S)
        
        # if just after any change point, skip.
        else:
            self.S.append(np.nan)
            self._update(y,update_alpha=False) # update Y_rf and Y_te without updating alpha.
        
    def plot_likelihood_ratio(self):
        """
        plot input data and likelihood ratio S with detected change points.
        """
        fig, ax = plt.subplots(2, 1,  figsize=(12, 8))
        
        # plot time series data
        ax[0].plot(self.t_list,self.y)
        
        # plot S
        ax[1].plot(self.t_list,self.S)
        ax[1].plot(self.t_list,[self.mu]*len(self.t_list),linestyle='dashed',label='threshold')
        ax[1].vlines(self.change_point_list,
                       np.nanmin(self.S)*1.2,
                       np.nanmax(self.S)*1.2,
                       color='red')
        plt.legend()
        plt.show()
    
    def _batch_KLIEP(self):
        """
        calculate KLIEP with batch algorithm
        """
        count = 0
        
        # continue until alpha converge.
        while(True):
            # calculate next alpha
            alpha_next = self.alpha + self.eps*self.K@(1/(self.K@self.alpha))
            alpha_next += (1-self.b.T*alpha_next)@self.b/(self.b.T@self.b)
            alpha_next = np.max([np.zeros_like(alpha_next),alpha_next],axis=0)
            alpha_next /= self.b.T@alpha_next
            
            # if update of alpha converges, break.
            if np.linalg.norm(alpha_next - self.alpha) < self.eps:
                break
                
            # update alpha and count
            self.alpha = alpha_next
            count += 1
            
            # if the loop is too long, break.
            if count>1000:
                print('self._batch_KLIEP(): DID NOT CONVERGE.')
                break
                
        # define w_hat(Y)
        def get_w_hat(Y):
            w_hat = sum([self.alpha[l]*self.func(Y,self.Y_te[l]) for l in range(self.n_te)])
            return w_hat
        self._get_w_hat = get_w_hat
    
    def _KLIEP_model_selection(self):
        """
        calculate KLIEP and select best value of sigma
        TODO: change this function
        """
        sigma_list = [1e-1,1e0,1e1,1e2,1e3,1e4,1e5]
        return 1e4
    
    def _compute_S(self):
        """
        calculate S using w_hat(Y)
        """
        return sum([np.log(self._get_w_hat(self.Y_te[i])) for i in range(self.n_te)])
    
    def _update(self,new_y,update_alpha=True):
        """
        update y,Y,Y_te,Y_rf,and alpha. (Algorithm 3)
        
        Parameters:
        ------------------
        new_y: np.array
            new input data
        update_alpha: boolean(default:True)
            update alpha using _update_alpha(), if True
        """
        # update ys,Ys,Y_tes
        self.y = np.vstack([self.y,new_y])
        self.Y = np.vstack([self.Y,self.y[np.newaxis,-self.k:,:]])
        self.Y_te = np.vstack([self.Y_te,self.Y[-1][np.newaxis,:]])
        
        # update alpha
        if update_alpha:
            self._update_alpha()
        
        # move from test to reference interval
        self.Y_rf = np.vstack([self.Y_rf,self.Y_te[0][np.newaxis,:]])
        self.Y_te = np.delete(self.Y_te,0,axis=0)
        
        # discard first data of reference interval
        self.Y = np.delete(self.Y,0,axis=0)
        self.Y_rf = np.delete(self.Y_te,0,axis=0)
        
        
    def _update_alpha(self):
        """
        update alpha with the idea of SGD
        """
        # calculate c
        c = sum([self.alpha[l]*self.func(self.Y_te[-1],self.Y_te[l]) for l in range(self.n_te)])
        
        # calculate next alpha
        self.alpha = np.append(np.delete(self.alpha,0)*(1-self.eta*self.lam),c)
        
        # calculate next alpha 
        self.b = self._get_b()
        self.alpha += (1-self.b.T*self.alpha)@self.b/(self.b.T@self.b)
        self.alpha = np.max([np.zeros_like(self.alpha),self.alpha],axis=0)
        self.alpha /= self.b.T@self.alpha
        
    def _get_K(self):
        """
        calculate matrix K
        
        Returns:
        ---------------
        K: np.matrix
        """
        return np.array([[self.func(self.Y_te[i],
                             self.Y_te[l]) for i in range(self.n_te)] for l in range(self.n_te)])
        
    def _get_b(self):
        """
        calculate vector b
        
        Returns:
        ---------------
        b: np.array
        """
        return np.array([sum([self.func(self.Y_rf[i],
                                 self.Y_te[l]) for i in range(self.n_rf)])/self.n_rf for l in range(self.n_te)])