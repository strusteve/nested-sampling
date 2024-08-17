import numpy as np
import matplotlib.pyplot as plt
import corner
from mh_sampler import metropolis_hastings
from nested_sampler import nested_sampler


class nested_examples(object):
    """ Display examples of functions fitted via a Nested Sampling algorithm (Skilling 2004) """

    def __init__(self):
        
        self.linear_data = np.genfromtxt('data_linear.csv', delimiter=',')
        self.quad_data = np.genfromtxt('data_quad.csv', delimiter=',')
        
    def log_likelihood(self, data, gen_func, scale, params):
        """ Return the natural logarithm of the Likelihood. Constructed from a single gaussian.

        Parameters
        ----------

        data :  numpy.ndarray
            Dataset containing x, y and y_uncert values arranged in 3 rows.

        gen_func : function
            Generative function used to model the data.

        scale : float
            Direct multiplication of the likelihood to help with underflow/overflow errors

        params : numpy.ndarray
            Free parameters of the generative model.
        """

        # Given Data
        xi, yi, yisig = data[0], data[1], data[2]

        # Likelihood of belonging to the 'good' gaussian
        li = scale * (1/(np.sqrt(2*np.pi*(yisig**2)))) * np.exp((-(yi - gen_func(xi, params))**2) / (2*(yisig**2)))
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

    def linear_example(self, N, prior_low, prior_high, scale):
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
        nest = nested_sampler(N, lambda params: self.log_likelihood(self.linear_data, self.linear_f, scale, params))
        samples, likelihood, evidence, priors = nest.run_sampler(prior_low, prior_high)

        # Compute the posterior and best fit model (MAP values)
        posterior = evidence / np.sum(evidence)
        best_fit = samples[np.argmax(posterior)]
        uncertainties = samples.T.std(1)

        # Plot results
        fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10,3))
        ax.plot(np.log(priors), likelihood, color='black')
        ax.set_xlabel('log(Enclosed Prior Volume)')
        ax.set_ylabel('log(Lower Likelihood Bound)')
        ax1.scatter(self.linear_data[0], self.linear_data[1], s=1, c='black')
        ax1.errorbar(self.linear_data[0], self.linear_data[1], self.linear_data[2], ls='none', lw=1, color='black')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        xtest = np.linspace(0,10,100)
        ytest = self.linear_f(xtest, best_fit)
        ax1.plot(xtest, ytest, color='red')
        title = ' f(x) = '
        for i in range(len(best_fit)):
            title += f'$({best_fit[i].round(2)} \pm {uncertainties[i].round(2)})x^{i}$ '

            if i != len(best_fit)-1:
                title += '+'
        ax1.set_title(title)
        plt.show()

    def quadratic_example(self, N, prior_low, prior_high, scale):
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
        nest = nested_sampler(N, lambda params: self.log_likelihood(self.quad_data, self.quad_f, scale, params))
        samples, likelihood, evidence, priors = nest.run_sampler(prior_low, prior_high)

        # Compute the posterior and best fit model (MAP values)
        posterior = evidence / np.sum(evidence)
        best_fit = samples[np.argmax(posterior)]
        uncertainties = samples.T.std(1)

        # Plot results
        fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10,3))
        ax.plot(np.log(priors), likelihood, color='black')
        ax.set_xlabel('log(Enclosed Prior Volume)')
        ax.set_ylabel('log(Lower Likelihood Bound)')
        ax1.scatter(self.quad_data[0], self.quad_data[1], s=1, c='black')
        ax1.errorbar(self.quad_data[0], self.quad_data[1], self.quad_data[2], ls='none', lw=1, color='black')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        xtest = np.linspace(0,10,100)
        ytest = self.quad_f(xtest, best_fit)
        ax1.plot(xtest, ytest, color='red')
        title = ' f(x) = '
        for i in range(len(best_fit)):
            title += f'$({best_fit[i].round(3)} \pm {uncertainties[i].round(3)})x^{i}$ '

            if i != len(best_fit)-1:
                title += '+'
        ax1.set_title(title)
        plt.show()


N = 10000
scale=100

examples = nested_examples()
examples.linear_example(N, [0, 0], [10, 10], scale)
examples.quadratic_example(N, [0, 0, -5], [10, 10, 5], scale)

