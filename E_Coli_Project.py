#import autograd.numpy as np
#from autograd import grad 
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from numpy import random

class Simulation():
    
    def __init__(self, num_ecoli: int, dist: str):
        """num_ecoli: number ecoli
            dist: distribution from the selection of {"gaussian", "expon", "fisk", "t", "lin"}
        """
        self.num_ecoli = num_ecoli
        #concentration strength
        #linear doesn't require concentration strength since values are large
        if dist != 'lin':
            self.strength = 100000
        else:
            self.strength = 1
        #ecoli run gradient magnitude scaling
        if dist != 'lin':
            self.learning_rate = 1000
        else:
            #adjust learning rate for linear
            self.learning_rate = 0.1
        #grid range; defines the range of the ecoli and distribution placement
        self.x_max = 200
        self.y_max = 200
        self.x_min = 0
        self.y_min = 0
        
        #hard coding variance
        init_con_var = 4000
        #randomly generating the distribution center
        init_con_center = np.random.uniform(50, 150, size=(2,))
        self.con_center = init_con_center
        self.run_sizes = []
        self.tumble_sizes = []
        #initialize the distribution
        self.conc_dist = self.__get_distribution(dist)(self.con_center, init_con_var)
        self.dist_desc = dist #distribution description

    def run_simulation(self, iterations: int, hist=False):
        """iterations: number of iterations
            hist: boolean indicating if the data is used for histogram generation
        """
        self.run_sizes = []
        self.tumble_sizes = []
        
        #concentration plotting resolution set to be 0.5
        mesh_min_bound = np.array((0, 0))
        mesh_max_bound = np.array((self.x_max, self.y_max))
        phi_m = np.linspace(mesh_min_bound[0], mesh_max_bound[0], int((self.x_max - self.x_min) / 0.5))
        phi_p = np.linspace(mesh_min_bound[1], mesh_max_bound[1], int((self.y_max - self.y_min) / 0.5))
        X,Y = np.meshgrid(phi_p, phi_m)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X; pos[:, :, 1] = Y
        concent = self.conc_dist(pos)
        
        non_zero_ind = np.nonzero(concent)
        x_vals = pos[non_zero_ind][:, 0]
        y_vals = pos[non_zero_ind][:, 1]
        x_min = x_vals.min()
        x_max = x_vals.max()
        y_min = y_vals.min()
        y_max = y_vals.max()
        init_ecoli_x = np.zeros((iterations, self.num_ecoli))
        init_ecoli_y = np.zeros((iterations, self.num_ecoli))
        init_ecoli_x[0, :] = np.random.uniform(x_min, x_max, size=(self.num_ecoli))
        init_ecoli_y[0, :] = np.random.uniform(y_min, y_max, size=(self.num_ecoli))
        #set all equal to the same position for hist
        if hist:
            init_ecoli_x[0, :] = init_ecoli_x[0, 0]
            init_ecoli_y[0, :] = init_ecoli_y[0, 0]
        #iteration
        i = 1
        run_turn = False
        while i < iterations:
            if run_turn:
                #print("running...")
                init_ecoli_x, init_ecoli_y = self.__run(init_ecoli_x, init_ecoli_y, i, step_size=5)
                run_turn = False
                i += 1
            else:
                num_steps = min(iterations - i, 4)
                init_ecoli_x, init_ecoli_y = self.__tumble(init_ecoli_x, init_ecoli_y, i, step_size=1, num_iterations=num_steps)
                i += num_steps
                run_turn = True

            if i % 100 == 0:
                fig, ax = plt.subplots()
                #ax.plot(init_ecoli_x[0, 0], init_ecoli_y[0, 0], color = 'red', marker = 'or')
                #p = ax.pcolor(X, Y, concent, cmap=plt.cm.RdBu, vmin=abs(concent).min(), vmax=abs(concent).max())
                
                
                end_point = min(iterations - 1, i)
                e_coli__min_bound = np.array(np.min(init_ecoli_x[end_point, :]), np.min(init_ecoli_y[end_point, :]))
                e_coli__max_bound = np.array(np.max(init_ecoli_x[end_point, :]), np.max(init_ecoli_y[end_point, :]))

                change_max = e_coli__max_bound > mesh_max_bound
                if change_max.any():
                    #import pdb; pdb.set_trace()
                    mesh_max_bound[change_max] = e_coli__max_bound[change_max]
                change_min = e_coli__min_bound < mesh_min_bound
                if change_min.any():
                    mesh_min_bound[change_min] = e_coli__min_bound[change_min]
                #change mesh grid if e. coli travel beyond current bounds
                if change_max.any() or change_min.any():
                    phi_m = np.linspace(mesh_min_bound[0], mesh_max_bound[0], 400)
                    phi_p = np.linspace(mesh_min_bound[1], mesh_max_bound[1], 400)
                    X,Y = np.meshgrid(phi_p, phi_m)
                    pos = np.empty(X.shape + (2,))
                    pos[:, :, 0] = X; pos[:, :, 1] = Y
                    concent = self.conc_dist(pos)
                #plot the individual paths of the e. coli
                p = ax.contourf(X, Y, concent, levels = 50, cmap = plt.cm.RdBu, vmin=abs(concent).min(), vmax=abs(concent).max())
                cb = fig.colorbar(p, ax=ax)
                ax.scatter(init_ecoli_x[0, :], init_ecoli_y[0, :], color='black', s=15, label='E. coli Start')
                ax.scatter(init_ecoli_x[end_point, :], init_ecoli_y[end_point, :], color='green', s=15, label='E. coli end')
                for j in range(init_ecoli_x.shape[1]):
                    ax.plot(init_ecoli_x[:end_point, j], init_ecoli_y[:end_point, j], color='gray', linewidth=1, alpha=0.7)
                # Add scatter for current positions
                ax.legend()
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.title(f"E. coli Trajectories at iteration {i}")
                plt.show()
            #print(i)
        #return the positions after simulation
        return init_ecoli_x, init_ecoli_y

        #if self.num_ecoli > 10:
            #h = self.__graph_histograms(self.con_center, all_ecoli_x , all_ecoli_y)
            #for i in range(10, len(self.tumble_sizes), 10):
                #plt.hist(self.tumble_sizes[i], label=f"tumble sizes {i}")
                ##plt.xlabel("Tumble Sizes")
                #plt.ylabel("Frequency")
                #plt.show()'''
            
            #print("number of runs: ", len(self.run_sizes))
            #for i in range(len(self.run_sizes)):
                #plt.hist(self.run_sizes[i], label=f"run sizes {i}")
                #plt.xlabel("Run Sizes")
                #plt.ylabel("Frequency")
                #plt.show()
    def plot_single_ecoli(self, ecoli_index: int, x: np.array, y: np.array):
        """
        ecoli_index: inddex of the ecoli being plot
        x: x-positions of all ecoli
        y: y-positions of all ecoli
        """
        mesh_min_bound = (self.x_min, self.y_min)
        mesh_max_bound = (self.x_max, self.y_max)
        phi_m = np.linspace(mesh_min_bound[0], mesh_max_bound[0], int((self.x_max - self.x_min) / 0.5))
        phi_p = np.linspace(mesh_min_bound[1], mesh_max_bound[1], int((self.y_max - self.y_min) / 0.5))
        X,Y = np.meshgrid(phi_p, phi_m)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X; pos[:, :, 1] = Y
        concent = self.conc_dist(pos)
        fig, ax = plt.subplots()
        p = ax.contourf(X, Y, concent, levels = 50, cmap = plt.cm.RdBu, vmin=abs(concent).min(), vmax=abs(concent).max())
        cb = fig.colorbar(p, ax=ax)
        ax.scatter(x[0, ecoli_index], y[0, ecoli_index], color='black', s=15, label='E. coli Start')
        ax.scatter(x[-1, ecoli_index], y[-1, ecoli_index], color='green', s=15, label='E. coli end')
        ax.plot(x[:, ecoli_index], y[:, ecoli_index], color='gray', linewidth=1, alpha=0.7)
        ax.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"E. coli {ecoli_index} trajectory")
        plt.show()

    def __tumble(self, x: np.array, y: np.array, time_step: int, num_iterations=4, step_size=2) -> (np.array, np.array):
        """simulate tumble for specified num_iterations steps of size step_size"""
        #get num_iteration independent tumbles steps of step_size with an offset of 1 at beginning for cumulative sum i.e. dx[0] = dy[0] = 0
        dx = np.zeros((num_iterations, *x[time_step, :].shape))
        dy = np.zeros((num_iterations, *y[time_step, :].shape))
        dx = step_size * np.random.uniform(-1, 1, (num_iterations, *x[time_step, :].shape))
        dy = step_size * np.random.uniform(-1, 1, (num_iterations, *y[time_step, :].shape))
        #perform cumulative sum from the step prior to the tumble start to the last tumble to get the aggregate position after each tumble
        new_x = np.cumsum(dx, axis=0)
        new_y = np.cumsum(dy, axis=0)
        #save tumble step magnitude to for relative scaling of gradients
        self.tumble_sizes.append(np.sqrt(new_x[-1, :]**2 + new_y[-1, :]**2)) 
        new_x += x[time_step - 1, :]
        new_y += y[time_step - 1, :]
        #ensure the ecoli don't tumble out of bound
        x[time_step : time_step + num_iterations, :] = np.clip(new_x, self.x_min, self.x_max)
        y[time_step : time_step + num_iterations, :] = np.clip(new_y, self.y_min, self.y_max)
        return x, y


    def __run(self, x: np.array, y: np.array, time_step: int, step_size: int=3, delta: int=4):
        #will perform a gradient based run using change in concentration from time_step - delta - 1 to time_step - 1 to determine the position at time_step
        
        delta_x = x[time_step - delta - 1, :]
        prev_x = x[time_step - 1, :]
        
        delta_y = y[time_step - delta - 1, :]
        prev_y = y[time_step - 1, :]

        delta_x_y = np.stack((delta_x, prev_y), axis=1)
        prev_x_y = np.stack((prev_x, prev_y), axis=1)
        #dz / dx * dx / dt where dz is change in concentration
        #the gradient is multiplied by the sign of x to account for backward tumbles
        dz_dx_dt = self.strength * step_size * ( self.conc_dist(prev_x_y) - self.conc_dist(delta_x_y)) / delta * np.sign(prev_x_y[:, 0]-delta_x_y[:,0])

        delta_y_x = np.stack((prev_x, delta_y), axis=1)
        prev_y_x = np.stack((prev_x, prev_y), axis=1)
        #dz / dy * dy / dt
        #the gradient is multiplied by the sign of y to account for backward tumbles
        dz_dy_dt = self.strength * step_size * (self.conc_dist(prev_y_x) - self.conc_dist(delta_y_x)) / delta * np.sign(prev_y_x[:, 1] - delta_y_x[:,1])
        
        #min_step = np.ones(self.run_sizes[-1].shape) * min(self.x_max - self.x_min, self.y_max - self.y_min) * 0.1
        self.run_sizes.append(np.sqrt(dz_dx_dt**2 + dz_dy_dt**2))
        #clip and miniumum are used to ensure that gradients don't explode out of bound 
        #ensure that the gradient is only an order magnitude bigger than tumble for smooth simulation
        grad_scaling = np.minimum(10 * self.tumble_sizes[-1], self.run_sizes[-1] * self.learning_rate)
        x[time_step, :] = np.clip(prev_x + dz_dx_dt * grad_scaling, self.x_min, self.x_max)
        y[time_step, :] = np.clip(prev_y + dz_dy_dt * grad_scaling, self.y_min, self.y_max)

        return x, y


    def __multivariate_gaussian(self, mu: np.array, sigma: int):
        #uniform gaussian
        init_con_var = np.eye(2)  * sigma
        gaussian_pdf = multivariate_normal(mu, init_con_var).pdf

        return gaussian_pdf
    
    def __multivariate_exponential(self, mu: np.array, sigma: np.array):
        #exponential
        from scipy.stats import expon
        x_mu, y_mu = mu
        x_sigma = y_sigma = sigma

        def sampler(x_y):
            x = x_y[..., 0]
            y = x_y[..., 1]
            x_sample = expon.pdf(x, loc=x_mu, scale=x_sigma)
            y_sample = expon.pdf(y, loc=y_mu, scale=y_sigma)
            return x_sample + y_sample
        
        return sampler
    
    def __multivariate_fisk(self, mu: np.array, sigma: np.array, **kwargs):
        from scipy.stats import fisk
        x_mu, y_mu = mu
        c_x = (random.rand()+0.0001)*10 # added to eliminate 0 possibility for shape parameter
        c_y = (random.rand()+0.0001)*10
        x_sigma = y_sigma = sigma

        def sampler(x_y, **kwargs):
            x = x_y[..., 0]
            y = x_y[..., 1]
            x_sample = fisk.pdf(x, c = c_x, loc = x_mu, scale = x_sigma)
            y_sample = fisk.pdf(y, c = c_y, loc = y_mu, scale = y_sigma)
            return x_sample + y_sample
        return sampler
    
    def __multivariate_t(self, mu: np.array, sigma: int):
        from scipy.stats import multivariate_t
        #t-distribution with covariance
        cov_var = np.random.uniform(0, 1, (2, 2))
        cov_var = np.maximum(cov_var, cov_var.T)
        init_con_var = np.maximum(np.eye(2), cov_var)  * sigma
        
        t_pdf = multivariate_t(mu, init_con_var).pdf

        return t_pdf
    
    def __linear_dist(self, mu: np.array, sigma: int):
        #simple linear distribution
        #reset concentration center to be the coordinates with max value of X + Y
        self.con_center = np.array([self.x_max, self.y_max])
        def sampler(x_y):
            x = x_y[..., 0]
            y = x_y[..., 1]
            mask = (x >= self.x_min) & (x <= self.x_max) & (y >= self.y_min) & (y <= self.y_max)
            #mask the out of bound values so e.coli don't move forever along line
            result = np.where(mask, x + y, -5000)
            return result 
        
        return sampler

    __supoorted_dist = {"gaussian", "expon", "fisk", "t", "lin"}

    def __get_distribution(self, name: str):
        #retrieves distribution from supported set of distributions
        assert name in self.__supoorted_dist, f"{name} is an unsupported distribution. Please select one of {self.__supoorted_dist}"
        if name == "gaussian":
            return self.__multivariate_gaussian
        elif name == "expon":
            return self.__multivariate_exponential
        elif name == "fisk" or name == "log-logistic":
            return self.__multivariate_fisk
        elif name == "t":
            return self.__multivariate_t
        elif name == "lin":
            return self.__linear_dist
    

    def get_grid(self, min_lim: float, max_lim: float, res: float = 0.01) -> np.meshgrid:
        """return a square mesh grid with the specified range and resolution <= res"""
        num = np.ceil((min_lim - max_lim) / res)
        x = np.linspace(min_lim, max_lim, num)
        y = np.linspace(min_lim, max_lim, num)
        return np.meshgrid(x, y)
    
    def get_histogram(self, ecoli_x_pos, ecoli_y_pos):
        """
            ecoli_x_pos: ecoli x-positions
            ecoli_y_pos: ecoli y-positions
        """
        #y_distances_from_food = np.zeros(ecoli_y_pos.shape[0], self.num_ecoli)
        t = ecoli_x_pos.shape[0]
        stamp = 0
        time_stamps = [i for i in range(0, t, 10)]
        #while 10**stamp - 1 < t:
        #    time_stamps.append(10**stamp - 1)
        #    stamp += 1
        
        distances_from_food = np.zeros((len(time_stamps), self.num_ecoli))
        for i in range (len(time_stamps)):
            x_distances_from_food = ecoli_x_pos[time_stamps[i]] - self.con_center[0]
            y_distances_from_food = ecoli_y_pos[time_stamps[i]] - self.con_center[1]
            distances_from_food[i, :] = np.sqrt(x_distances_from_food**2 + y_distances_from_food**2)
        #hist_data = pd.DataFrame(distances_from_food, columns = ['Initial', 'I = 10', 'I = 100', 'I = 500', 'I = 1000'])
        num_bins = 200
        #make sure each iteration is spaced evenly
        fig, axs = plt.subplots(len(time_stamps), 1, figsize=(7, 2 * len(time_stamps)))
        for i in range (len(time_stamps)):
            #ax = plt.subplot(int(len(time_stamps)), 1, i+1)
            #plt.hist(distances_from_food[i], num_bins, range = (0, int(np.sqrt(self.x_max**2+ self.y_max**2))))
            axs[i].hist(distances_from_food[i], num_bins, range = (0, int(np.max(distances_from_food))))
            
            axs[i].set_title('I = ' + str(time_stamps[i]), loc = 'left', fontsize = 'x-small', pad = -4.5)
        #fig = plt.gcf()
        #plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        #print(fig.get_size_inches())
        plt.tight_layout()
        plt.suptitle('Distance from Food Source')
        plt.show()
        return distances_from_food
        


    

