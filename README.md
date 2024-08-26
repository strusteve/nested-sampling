# Summer coding challenge
 
Fit functions to data using:<br /><br /> (1) A Monte-Carlo Markov Chain, specifically the Metropolis-Hastings algorithm<br /> (2) A nested-sampling algorithm (Skilling 2004, Feroz 2008)


### Metropolis-Hastings

Summary: Starts from an initial position in parameter space, then uses proposal distributions to move around in parameter space. Moves are accepted based on an acceptance probability, calculated from posterior and proposal distribution ratios for the old and new steps. Stops after a certain amount of (specified) iterations. 

Input: likelihood function, initial guess, stepsizes, number of iterations

Notes: 
- Uses uniform priors (in practise) and symmetric proposal distributions hence the acceptance ratio is simply a likelihood ratio.
- Uses constant stepsizes hence is sensitive to these chosen values.

Refinements:
- Tune stepsizes via the acceptance ratio automatically.

### Nested Sampling

Summary: Starts from N positions in parameter space, replaces the lowest likelihood position with a higher likelihood position sampled from a likelihood-bounded prior. This process iterates, accumulating bayesian evidence at each 'layer' until only a negligible amount is left. From the discarded positions posterior samples can be drawn and inferences made.

Input: likelihood function, number of samples, lower prior bounds, upper prior bounds, tolerance

Notes:
- Returns posterior samples as side product of accumulating Bayesian evidence.
- In sampling a likelihood-bounded prior, uses uniform priors (explicitly) and symmetric proposal distributions in a Metropolis algorithm.
- Uses variable stepsizes in a self-contained Metropolis algorithm.

Refinements:
- Improve memory efficiency

