# Summer coding challenge
 
Fit functions to data using:<br /><br /> (1) A Monte-Carlo Markov Chain, specifically the Metropolis-Hastings algorithm<br /> (2) A nested-sampling algorithm (Skilling 2004, Feroz 2008)


### Metropolis-Hastings

Summary: Starts from an initial position in parameter space, then uses proposal distributions to move around in parameter space. Moves are accepted based on an acceptance probability, calculated from posterior and proposal distribution ratios for the old and new steps. Stops after a certain amount of (specified) iterations. 

Input: data with errors on y, generative function, initial guess, stepsizes, number of iterations, likelihood scaling factor

Notes: 
- Uses uniform priors (in practise) and symmetric proposal distributions hence the acceptance ratio is simply a likelihood ratio
- Uses constant stepsizes hence is sensitive to these values

Refinements to be made:
- Outlier model for the likelihood function (introduces three new parameters $P_b, Y_b, V_b$)
- Tune stepsizes via the acceptance ratio automatically

### Nested Sampling

Summary: Starts from N positions in parameter space, replaces the lowest likelihood position with a higher likelihood position sampled from a likelihood-bounded prior. This process iterates, accumulating bayesian evidence at each 'layer' until only a negligible amount is left. Discarded positions represent posterior samples and posterior values for each are calculated via the evidence.

Input: data with errors on y, generative function, number of samples, lower prior bounds, upper prior bounds, likelihood scaling factor

Notes:
- Returns bayesian evidence as well as posterior samples
- In sampling a likelihood-bounded prior, uses uniform priors (explicitly) and symmetric proposal distributions in a Metropolis algorithm
- Uses variable stepsizes in a self-contained Metropolis algorithm

Refinements to be made:
- Outlier model for the likelihood function (introduces three new parameters $P_b, Y_b, V_b$ and a lot of underflow/overflow errors)
- Return evidence error
- Return local evidences
