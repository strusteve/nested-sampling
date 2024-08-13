import numpy as np
import matplotlib.pyplot as plt

# Generative function to be fit
def f(x, par):
    y = par[0] + (par[1]*x)
    return(y)

# Objective function via a likelihood distribution
def log_likelihood(data,  params, scale):

    # Given Data
    xi, yi, yisig = data[0], data[1], data[2]

    # Likelihood of belonging to the 'good' gaussian
    li = scale * (1/(np.sqrt(2*np.pi*(yisig**2)))) * np.exp((-(yi - f(xi, params))**2) / (2*(yisig**2)))
 
    log_li = np.log(li)
    # Sum likelihood and take logarithm (note sometimes recieve underflow error due to small likelihoods)
    log_l_total = np.sum(log_li)
    
    return(log_l_total)

# Metropolis-hastings MCMC algorithm - returns Markov chain
def metropolis_hastings(data, guess, stepsizes, N, scale):

    # Markov Chain
    chain = np.array(([guess]))
    accepted = 0

    # Metropolis-Hastings Algorithm
    for i in range(N):

        #Get current parameters & subsequent likelihood
        current_params = chain[len(chain)-1]
        current_l = log_likelihood(data, current_params, scale)

        #Propose new parameters (step in parameter space) from gaussian proposal distributions
        new_params = np.zeros(len(current_params))

        for j in range(len(current_params)):
            new_params[j] = np.random.normal(current_params[j], stepsizes[j])

        #Get likelihood for proposed step
        new_l = log_likelihood(data, new_params, scale)

        #Calculate acceptance ratio (NB symmetric proposal distributions)
        log_prob_ratio = new_l - current_l

        # Generate random number from 0-1
        mu = np.random.random_sample()

        #Accept or reject step (kept in logarithm due to underflow errors)
        if (np.log(mu) < min(np.log(1), log_prob_ratio)):
            chain = np.vstack((chain, new_params))
            accepted += 1
        else:
            chain = np.vstack((chain, current_params))

    # Print acceptance ratio for optimisation of stepsizes
    print('Acceptance rate = ' + str((accepted/N)*100) + '%')

    # Discard first half of Markov Chain to ignore burn-in
    clean_chain = chain[int(N/2):].T

    return clean_chain


'''

User defined variables:

Guess = initial starting point for the mcmc algorithm
Stepsizes = width of the gaussian proposal functions 
N = Number of steps taken in parameter space
scale = direct multiplication with each point's likelihood (to avoid underflow/overflow errors)

'''

guess = np.array(([0, 0]))
stepsizes = np.array(([1, 1]))
N=10000
scale = 1

#######################################################

data = np.genfromtxt('data_linear.csv', delimiter=',')

# Run MH algorithm for a Markov Chain
chain = metropolis_hastings(data, guess, stepsizes, N, scale)

# Pick optimised parameter values as the mean of each parameter's marginalised posterior distribution (approximated by histograms)
params_fit = np.mean(chain, axis=1)
uncert_fit = np.std(chain, axis=1)

# Plot results
xtest = np.linspace(0,10,1000)
ytest = f(xtest, params_fit)
fig, ax = plt.subplots()
ax.scatter(data[0], data[1], s=1, c='black')
ax.errorbar(data[0], data[1], data[2], ls='none', lw=1, c='black', capsize=1)
ax.plot(xtest, ytest, c='red')
ax.set_title(f'$f(x) = ({params_fit[0].round(2)}\pm{uncert_fit[0].round(2)}) + ({params_fit[1].round(2)}\pm{uncert_fit[1].round(2)})x$')

plt.show()

