import numpy as np
import matplotlib.pyplot as plt
import corner
from mh_sampler import metropolis_hastings

class mh_examples(object):
    """ Display examples of functions fitted via the Metropolis-Hastings algorithm"""

    def __init__(self):
        
        self.linear_data = np.genfromtxt('data_linear.csv', delimiter=',')
        self.quad_data = np.genfromtxt('data_quad.csv', delimiter=',')
        

    def log_likelihood(self, data, gen_func, params):
        """ Return the natural logarithm of the Likelihood. Constructed from a single gaussian.

        Parameters
        ----------

        data :  numpy.ndarray
            Dataset containing x, y and y_uncert values arranged in 3 rows.

        gen_func : function
            Generative function used to model the data.

        params : numpy.ndarray
            Free parameters of the generative model.
        """

        # Given Data
        xi, yi, yisig = data[0], data[1], data[2]

        # Likelihood of belonging to the 'good' gaussian
        li = (1/(np.sqrt(2*np.pi*(yisig**2)))) * np.exp((-(yi - gen_func(xi, params))**2) / (2*(yisig**2)))
    
        log_li = np.log(li)
        # Sum likelihood and take logarithm (note sometimes recieve underflow error due to small likelihoods)
        log_l_total = np.sum(log_li)
        
        return(log_l_total)

    def linear_f(self, x, par):
        """ Linear function.

        Parameters
        ----------

        x : numpy.ndarray
            One dimensional dataset of x-values.
        
        par : numpy.ndarray
            Free parameters.
        """

        y = par[0] + (par[1]*x)
        return(y)

    def quad_f(self, x, par):
        """ Quadratic function.

        Parameters
        ----------

        x : numpy.ndarray
            One dimensional dataset of x-values.
        
        par : numpy.ndarray
            Free parameters.
        """

        y = par[0] + (par[1]*x) + (par[2]*(x**2))
        return(y)

    def linear_example(self, N, guess, stepsizes):
        """ Run MH algorithm on linear dataset and plot results.

        Parameters
        ----------

        N : integer
            Number of algorithm iterations.

        guess : numpy.ndarray
            Starting point in parameter space for MH algorithm.

        stepsizes : numpy.ndarray
            Width of the gaussian proposal distributions. Higher values cause the sampler to traverse larger distances in parameter space.
        
        """

        # Run MH algorithm for a Markov Chain
        mcmc = metropolis_hastings(lambda params: self.log_likelihood(self.linear_data, self.linear_f, params))
        chain = mcmc.getchain(N, guess, stepsizes)
        clean_chain = chain[int(N/2):].T

        # Pick optimised parameter values as the mean of each parameter's marginalised posterior distribution (approximated by histograms)
        params_fit = np.mean(clean_chain, axis=1)
        uncert_fit = np.std(clean_chain, axis=1)

        # Plot results
        xtest = np.linspace(0,10,1000)
        ytest = self.linear_f(xtest, params_fit)
        fig, ax = plt.subplots()
        ax.scatter(self.linear_data[0], self.linear_data[1], s=1, c='black')
        ax.errorbar(self.linear_data[0], self.linear_data[1], self.linear_data[2], ls='none', lw=1, c='black', capsize=1)
        ax.plot(xtest, ytest, c='red')
        title = ' f(x) = '
        for i in range(len(params_fit)):
            title += f'$({params_fit[i].round(3)} \pm {uncert_fit[i].round(3)})x^{i}$ '

            if i != len(params_fit)-1:
                title += '+'
        ax.set_title(title)
        corner.corner(clean_chain.T, labels=['$c$', '$m$'])
        plt.show()

    def quadratic_example(self, N, guess, stepsizes):
        """ Run MH algorithm on quadratic dataset and plot results.

        Parameters
        ----------

        N : integer
            Number of algorithm iterations.

        guess : numpy.ndarray
            Starting point in parameter space for MH algorithm.

        stepsizes : numpy.ndarray
            Width of the gaussian proposal distributions. Higher values cause the sampler to traverse larger distances in parameter space.
        
        """

        # Run MH algorithm for a Markov Chain
        mcmc = metropolis_hastings(lambda params: self.log_likelihood(self.quad_data, self.quad_f, params))
        chain = mcmc.getchain(N, guess, stepsizes)
        clean_chain = chain[int(N/2):].T

        # Pick optimised parameter values as the mean of each parameter's marginalised posterior distribution (approximated by histograms)
        params_fit = np.mean(clean_chain, axis=1)
        uncert_fit = np.std(clean_chain, axis=1)

        # Plot results
        xtest = np.linspace(0,10,1000)
        ytest = self.quad_f(xtest, params_fit)
        fig, ax = plt.subplots()
        ax.scatter(self.quad_data[0], self.quad_data[1], s=1, c='black')
        ax.errorbar(self.quad_data[0], self.quad_data[1], self.quad_data[2], ls='none', lw=1, c='black', capsize=1)
        ax.plot(xtest, ytest, c='red')
        title = ' f(x) = '
        for i in range(len(params_fit)):
            title += f'$({params_fit[i].round(3)} \pm {uncert_fit[i].round(3)})x^{i}$ '

            if i != len(params_fit)-1:
                title += '+'
        ax.set_title(title)
        corner.corner(clean_chain.T, labels=['$p0$', '$p1$', '$p2$'])
        plt.show()



examples = mh_examples()
examples.linear_example(int(1e6), [0,0], [1,1])
examples.quadratic_example(int(1e7), [0,0,0], [1,1,1])