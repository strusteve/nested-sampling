import numpy as np
import matplotlib.pyplot as plt
import corner


class metropolis_hastings(object):


    def __init__(self, logprob):
        
        self.ext_logprob = logprob


    # Metropolis-hastings MCMC algorithm - returns Markov chain
    def getchain(self, N, guess, stepsizes):

        # Markov Chain
        self.chain = np.zeros((N, len(guess)))
        self.chain[0, :] = guess
        accepted = 0


        # Metropolis-Hastings Algorithm
        for i in range(N-1):

            #Get current parameters & subsequent likelihood
            current_params = self.chain[i, :]
            current_prob = self.ext_logprob(current_params)

            #Propose new parameters (step in parameter space) from gaussian proposal distributions & subsequent likelihood
            new_params = current_params + (stepsizes * np.random.randn(current_params.shape[0]))
            new_prob = self.ext_logprob(new_params)

            #Calculate acceptance ratio (NB uniform priors and symmetric proposal distributions)
            log_prob_ratio = new_prob - current_prob

            # Generate random number from 0-1
            mu = np.random.rand()

            #Accept or reject step (kept in logarithm due to underflow errors)
            if (np.log(mu) < min(np.log(1), log_prob_ratio)):
                self.chain[i+1] = new_params
                accepted += 1
            else:
                self.chain[i+1] = current_params

        # Print acceptance ratio for optimisation of stepsizes
        print('Acceptance rate = ' + str((accepted/N)*100) + '%')

        return(self.chain)