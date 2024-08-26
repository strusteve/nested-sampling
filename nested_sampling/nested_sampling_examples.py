import numpy as np
import matplotlib.pyplot as plt
import corner
import scipy as scp
from mh_sampler import metropolis_hastings
from nested_sampler import nested_sampler


class nested_examples(object):
    """ Display examples of functions fitted via a Nested Sampling algorithm (Skilling 2004) """

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
        #log_li = np.log(scale * (1/(np.sqrt(2*np.pi*(yisig**2)))) * np.exp((-(yi - gen_func(xi, params))**2) / (2*(yisig**2))))
        log_li = np.log((1/(np.sqrt(2*np.pi*(yisig**2))))) + ((-(yi - gen_func(xi, params))**2) / (2*(yisig**2)))

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

    def linear_example(self, N, prior_low, prior_high, log_evidence_tol):
        """ Run nested sampling algorithm on a linear dataset and plot results.

        Parameters
        ----------

        N : integer
            Number of samples.

        prior_low : numpy.ndarray
            Array of lower bounds for each parameter's uniform prior distribution.

        prior_high : numpy.ndarray
            Array of upper bounds for each parameter's uniform prior distribution.
        
        """

        # Run nested sampling algorithm
        nest = nested_sampler(N, lambda params: self.log_likelihood(self.linear_data, self.linear_f, params), log_evidence_tol)
        samples = nest.run_sampler(prior_low, prior_high)

        # Compute the posterior and best fit model (MAP values)
        best_fit = np.median(samples, axis=0)
        uncertainties = np.std(samples, axis=0)

        # Plot results
        fig, ax = plt.subplots(figsize=(10,3))
        ax.scatter(self.linear_data[0], self.linear_data[1], s=1, c='black')
        ax.errorbar(self.linear_data[0], self.linear_data[1], self.linear_data[2], ls='none', lw=1, color='black')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        xtest = np.linspace(0,10,100)
        ytest = self.linear_f(xtest, best_fit)
        ax.plot(xtest, ytest, color='red')
        title = ' f(x) = '
        for i in range(len(best_fit)):
            title += f'$({best_fit[i].round(2)} \pm {uncertainties[i].round(2)})x^{i}$ '

            if i != len(best_fit)-1:
                title += '+'
        ax.set_title(title)

        corner.corner(samples, labels=['$b$', '$m$'])

        plt.show()

    def quadratic_example(self, N, prior_low, prior_high, log_evidence_tol):
        """ Run MH algorithm on quadratic dataset and plot results.

        Parameters
        ----------

        N : integer
            Number of samples.

        prior_low : numpy.ndarray
            Array of lower bounds for each parameter's uniform prior distribution.

        prior_high : numpy.ndarray
            Array of upper bounds for each parameter's uniform prior distribution.
        
        """

        # Run nested sampling algorithm
        nest = nested_sampler(N, lambda params: self.log_likelihood(self.quad_data, self.quad_f, params), log_evidence_tol)
        samples = nest.run_sampler(prior_low, prior_high)

        # Compute the posterior and best fit model (MAP values)
        best_fit = np.median(samples, axis=0)
        uncertainties = np.std(samples, axis=0)

        # Plot results
        fig, ax = plt.subplots(figsize=(10,3))
        ax.scatter(self.quad_data[0], self.quad_data[1], s=1, c='black')
        ax.errorbar(self.quad_data[0], self.quad_data[1], self.quad_data[2], ls='none', lw=1, color='black')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        xtest = np.linspace(0,10,100)
        ytest = self.quad_f(xtest, best_fit)
        ax.plot(xtest, ytest, color='red')
        title = ' f(x) = '
        for i in range(len(best_fit)):
            title += f'$({best_fit[i].round(3)} \pm {uncertainties[i].round(3)})x^{i}$ '

            if i != len(best_fit)-1:
                title += '+'
        ax.set_title(title)

        corner.corner(samples, labels=['$p0$', '$p1$', '$p2$'])
        plt.show()




N = 1000

examples = nested_examples()
examples.linear_example(N, [-10, -10], [10, 10], np.log(1e-6))
examples.quadratic_example(N, [0, 0, -5], [10, 10, 5], np.log(1e-6))

