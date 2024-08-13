import numpy as np
import matplotlib.pyplot as plt

# Objective function via a likelihood distribution
def log_likelihood(data, gen_func,  params, scale):

    # Given Data
    xi, yi, yisig = data[0], data[1], data[2]

    # Likelihood of belonging to the 'good' gaussian
    '''
    li_good = scale * (1/(np.sqrt(2*np.pi*(yisig**2)))) * np.exp((-(yi - gen_func(xi, params))**2) / (2*(yisig**2)))
    log_li = np.log(li_good)

    The above produces underflow errors but will eventually have to be reconsidered when using the outlier model
    '''
    li_good = np.log(scale * (1/(np.sqrt(2*np.pi*(yisig**2))))) + ((-(yi - gen_func(xi, params))**2) / (2*(yisig**2)))

    # Sum likelihood and take logarithm (note sometimes recieve underflow error due to small likelihoods)
    log_l_total = np.sum(li_good)
    
    return(log_l_total)

# Distribution of successive prior volume ratios (i.e the probability distribution for the largest of N samples drawn uniformly from the interval [0, 1])
def prob_t(t, N):
    log_p = np.log(N) + ((N-1) * np.log(t))
    return log_p

# Sample the distribution of successive prior volume ratios via metropolis-hastings MCMC
def sample_prob_t(N, samps, stepsize):

    # Uniform prior for guesses
    t_init = 0.9
    
    # Initialise markov chain
    chain = np.array(([t_init]))
    accepted = 0

    for i in range(samps):
        current_t = chain[len(chain)-1]
        current_p = prob_t(current_t, N)

        new_t = np.random.normal(current_t, stepsize)
        if new_t > 1:
            new_t = current_t

        new_p = prob_t(new_t, N)

        # Calculate acceptance ratio (NB symmetric proposal distributions)
        posterior_ratio = new_p - current_p
        mu = np.random.uniform(0,1)

        #Accept or reject step (kept in logarithm due to underflow errors)
        if (np.log(mu) < min(np.log(1), posterior_ratio)):
            chain = np.vstack((chain, new_t))
            accepted += 1
        else:
            chain = np.vstack((chain, current_t))

    return chain.flatten()

# Sample a likelihood bounded prior via the metropolis algorithm (adapted from Feroz 2008 Section 6)
def metropolis_prior_sampling(data, gen_func, sorted_prior_samples, sorted_likelihoods, sigmas, scale):

    likelier_point = sorted_prior_samples[1:][np.random.randint(0, N-1)]

    accepted=0
    rejected=0
    chain = np.array((likelier_point), ndmin=2)

    for i in range(20):

        current_point = chain[-1]

        trial_point = np.zeros(len(current_point))
        for j in range(len(trial_point)):
            trial_point[j] = np.random.normal(current_point[j], sigmas[j])
        trial_likelihood = log_likelihood(data, gen_func, trial_point, scale)

        #Calculate prior ratio (NB uniform priors)
        prior_ratio = 1 #Included here for completeness

        #Acceptance ratio (NB symmetric proposal distributions)
        if trial_likelihood > sorted_likelihoods[0]:
            alpha = min(1, prior_ratio)
        else:
            alpha = 0
            
        # Generate random number from 0-1
        mu = np.random.random_sample()

        #Accept or reject step (kept in logarithm due to underflow errors)
        if (mu < alpha):
            chain = np.vstack((chain, trial_point))
            accepted += 1
        else:
            chain = np.vstack((chain, current_point))
            rejected += 1

    return chain[-1], accepted, rejected

# Run a nested sampling algorithm (Skilling 2004, Feroz 2008)
def nested_sampling_algorithm(data, gen_func, N, prior_low, prior_high, scale):

    # Sample 'successive prior ratios' distribution
    samps=100000
    chain_t = sample_prob_t(N, samps, 0.01)

    # Variables
    sigmas = np.ones(len(prior_high))

    # Memory
    like_layers = np.array(())
    prior_volumes = np.array(([1]))
    evidence_layers = np.array(([0]))
    discarded_points = []

    ################################################# 

    '''
    Set the Base layer (outer nest)
    '''

    # Draw N samples from the full prior
    prior_samples = np.zeros((N, len(prior_high)))
    for i in range(len(prior_high)):
        prior_samples.T[i] = np.random.uniform(prior_low[i], prior_high[i], N)

    # Evaluate the likelihood for each sample
    log_likes = np.array(())
    for j in range(N):
        log_like = log_likelihood(data, gen_func, prior_samples[j], scale)
        log_likes = np.append(log_likes, log_like)

    # Sort samples in order of their likelihoods
    sorted_likelihoods = log_likes[np.argsort(log_likes)]
    like_layers = np.append(like_layers, sorted_likelihoods[0])
    sorted_prior_samples = prior_samples[np.argsort(log_likes)]
    discarded_points.append(sorted_prior_samples[0])

    # Loop over nested liklihood layers
    while True:

        '''
        Jump to next layer in the nest
        '''

        # Replace the lowest-likelihood point with another from bounded prior via metropolis algorithm
        new_point, accepted, rejected = metropolis_prior_sampling(data, gen_func, sorted_prior_samples, sorted_likelihoods, sigmas, scale)
        new_likelihood = log_likelihood(data, gen_func, new_point, scale)

        sorted_prior_samples[0] = new_point
        sorted_likelihoods[0] = new_likelihood

        sorted_prior_samples = sorted_prior_samples[np.argsort(sorted_likelihoods)]
        sorted_likelihoods = sorted_likelihoods[np.argsort(sorted_likelihoods)]

        like_layers = np.append(like_layers, sorted_likelihoods[0])
        discarded_points.append(sorted_prior_samples[0])

        
        # Update metropolis stepsizes for bounded prior sampling
        if accepted > rejected:
            sigmas = sigmas * np.exp(1/accepted)
        elif accepted <= rejected:
            sigmas = sigmas * np.exp(-1/rejected)
        

        # Update the prior volume for the new layer
        prior_volume = chain_t[np.random.randint(0, samps-1)] * prior_volumes[-1]
        prior_volumes = np.append(prior_volumes, prior_volume)

        # Accumulate evidence
        weight = prior_volumes[-2] - prior_volumes[-1]
        lowest_nonlog_likelihood = np.exp(sorted_likelihoods[0])
        evidence_contrib = lowest_nonlog_likelihood * weight
        evidence_layers = np.append(evidence_layers, evidence_contrib)

        # Stopping criterion
        max_contrib = np.exp(sorted_likelihoods[-1]) * weight
        log_max_contrib_ratio = np.log(max_contrib) - np.log(np.sum(evidence_layers))
        print(f'Remaining log-evidence: {log_max_contrib_ratio}')

        if log_max_contrib_ratio < 0.1:
            # Return the posterior
            return np.array((discarded_points)), like_layers, evidence_layers, prior_volumes

# Simple function for running the algorithm and plotting
def run_nested_sampling(data, gen_func, N, prior_low, prior_high, scale):

    # Run the nested sampling algorithm
    samples, likelihood, evidence, priors = nested_sampling_algorithm(data, gen_func, N, prior_low, prior_high, scale)

    # Compute the posterior and best fit model (MAP values)
    posterior = evidence / np.sum(evidence)
    best_fit = samples[np.argmax(posterior)]
    uncertainties = samples.T.std(1)


    # Plot results
    fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10,3))
    ax.plot(np.log(priors), likelihood, color='black')
    ax.set_xlabel('log(Enclosed Prior Volume)')
    ax.set_ylabel('log(Lower Likelihood Bound)')
    ax1.scatter(data[0], data[1], s=1, c='black')
    ax1.errorbar(data[0], data[1], data[2], ls='none', lw=1, color='black')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    xtest = np.linspace(0,10,100)
    ytest = gen_func(xtest, best_fit)
    ax1.plot(xtest, ytest, color='red')

    title = ' f(x) = '
    for i in range(len(best_fit)):
        title += f'$({best_fit[i].round(2)} \pm {uncertainties[i].round(2)})x^{i}$ '

        if i != len(best_fit)-1:
            title += '+'
    

    ax1.set_title(title)
    


    plt.show()


'''

User defined variables:

N = Number of samples used in nested sampling
prior_low = lower bounds for each model parameter (uniformly sampled)
prior_high = upper bounds for each model parameter (uniformly sampled)
scale = direct multiplication with each point's likelihood (to avoid underflow/overflow errors)

'''

# Linear function to be fit
def linear_f(x, par):
    y = par[0] + (par[1]*x)
    return(y)

N = 10000
prior_low = [0, 0]
prior_high = [10, 10]
scale=100
data_linear = np.genfromtxt('data_linear.csv', delimiter=',')
run_nested_sampling(data_linear, linear_f, N, prior_low, prior_high, scale)

###################################################################

# Quadratic function to be fit
def quad_f(x, par):
    y = par[0] + (par[1]*x) + (par[2]*(x**2))
    return(y)

N = 10000
prior_low = [0, 0, -5]
prior_high = [10, 10, 5]
scale=100
data_quad = np.genfromtxt('data_quad.csv', delimiter=',')
run_nested_sampling(data_quad, quad_f, N, prior_low, prior_high, scale)

