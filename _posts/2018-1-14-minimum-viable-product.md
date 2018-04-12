---
layout: post
title: A minimally viable machine learning service
---

>For the theory-practice iteration to work, the scientist must be, as it were, mentally ambidextrous; fascinated equally on the one hand by possible meanings, theories, and tentative models to be induced from data and the practical reality of the real world, and on the other with the factual implications deducible from tentative theories, models and hypotheses.
    - George E. P. Box

A canonical virtue of science and engineering is simplicity. Between two hypotheses that fit observations equally well, a scientist chooses the simpler one. Why? 

Occam's razor says not to "multiply causes", and its 20th century [Bayesian update](http://mlg.eng.cam.ac.uk/zoubin/papers/05occam/occam.pdf) says that hypotheses that are "hard to vary" are preferred. I borrow the "hard to vary" phrase from David Deutsch, who builds his epistemology on this principle (see [this TED talk](https://www.ted.com/talks/david_deutsch_a_new_way_to_explain_explanation) for a quick exposition). 

My preferred justification for simplicity is this:
>It is clear that, in a world with computable processes, patterns which result from simple processes are relatively likely, while patterns that can only be produced by very complex processes are relatively unlikely. 

I took this quote from the Scholarpedia article on [Solomonoff induction](http://www.scholarpedia.org/article/Algorithmic_probability#Discrete_Universal_A_Priori_Probability), which unifies and formalizes Bayes, Occam and Epicurus (Epicurus said, presumably in ancient Greek, "keep all hypotheses that are consistent with the data").

For years, I didn't fully appreciate the other great benefit of simplicity, which is that simple things are easier to build and maintain. My supervisor instructed me to always cut things down to their essence, to break problems down to their smallest components, as in the very word "analysis". And yet simplifying things is somehow so counter-intuitive that I must constantly remind myself to do it.

## Compositional Structure Search

My PhD project is called "Machine Learning as a Service", or MLaaS for short. My aim is to build a web service where anyone can upload their data, and download an analysis of it, in the form of a report, and perhaps code. Such services already exist - my favorite example being the [Automatic Statistician](https://www.automaticstatistician.com/about/), which brands itself as "An artificial intelligence for data science". Judging by the example analyses, the automatic statistician can build linear models and do time series analysis.

Linked under Research, one finds the paper "Exploiting compositionality to explore a large space of model structures" by Roger Grosse, Ruslan Salakhutdinov, William Freeman and Joshua Tenenbaum, which I have chosen as a starting point for my MLaaS. I decided to familiarize myself with the conveniently available code accompanying the paper, and write the next version of it. The code implements Compositional Structure Search (CSS): a tree-search algorithm that mimics the model building process of a data scientist. 

### Many models are factorizations

A number of machine learning models are matrix factorizations. A simpler version of a factor model that assumes equal noise variance in all variables, probabilistic PCA, factorizes the mean of the observed Gaussian into a product of orthogonal directions of greatest variance in the data and coordinates along these directions so as to place each data point in the latent space. These coordinates are the latent variable. The directions of greaters variance, or principal components, would be considered parameters, not latent variables. The difference is that different parameters from a parameter distribution yield different models, while different latent variables from a latent distribution yield different data points.
What characters live in latent spaces? There are two kinds - continous and discrete. Discrete latent variables are usually objects and events, say "apple" or "seizure" or "walking". Continous latent variables can vary in degree, like size, shape, color and direction, so we might have "300 meters long", "Flat", "Yellow" and "from above". Discrete variables tend to be concepts, continous ones tend to be percepts.


#### Example 
Say you have a dataset of cars, their model name, price, engine specs, size 
and more. 
A continous latent variable model like PCA, might find latent spaces in which 
moving in a direction called "Family - Sports" reveals Fiats on one side and 
Ferraris on the other. Cars in the middle would interpolate between the two, 
maybe like a Ford. 
A discrete latent variable model might find clusters or communities, where 
every member share enough properties to make them distinct from other cars. 
Say all Japanese cars are produced in the same three kinds of factories. 
The discrete latent variable might then "unwittingly" represent those factories.

Combing back to the tree search, a search move consists of growing the tree by building an extension of the model on the current leaf, by applying a composition rule. More technically, when applying a production rule P to a matrix S, sample from the posterior for P’s generative model conditioned
on it evaluating (exactly) to S. Each production rule has its own MCMC sampler. Sampling a finished model would consist of ancestral sampling, where each ancestor has his own sampling method. 

Here comes the exciting part:

>This procedure allows us to reuse computations between different structures.
Most of the computation time is in the initialization
steps. Each of these steps only needs to be run once on the
full matrix, specifically when the first production rule is applied.
Subsequent initialization steps are performed on the
component matrices, which are considerably smaller. This
allows a large number of high level structures to be fit for a
fraction of the cost of fitting them from scratch.

For example, we might extend a low rank factorization by modelling its latent variable matrix as Gaussian mixture. We compose a low rank factor model with a mixture model to get a mixture-prior low rank model. Perhaps we are looking for brain regions in EEG data. 
We could compose the reverse by adding a factor model to each component in the mixture and get a mixture of factor models, a very general density estimator (think balloon-animals, but the segments don't need to connected or have the same size and shape). Those would be level-2 in the tree. 

In general any factorization can be used to extend any of the factors in the current level-n composition. This presents us with a decision: what rule should we use, and where should we apply it?
CSS applies all composition rules to the K models that had the best predictive likelihood on held-out data. Finding the predictive likelihood is challenging in itself, a problem Grosse et al. tackle by expanding out the products in the expression X = U1V1 + · · · + UnVn + E, (1)
where E is an i.i.d. Gaussian “noise” matrix and the Ui’s
are any of the following: (1) a component matrix G, M,
or B, (2) some number of Cs followed by G, (3) a Gaussian
scale mixture.  The held-out row x can therefore be
represented as:
x = VT1 u1 + · · · + VTn un + e. (2)


The predictive likelihood is given by:
p(x|XO) = int p(UO, V |XO)p(u|UO)p(x|u, V ) dUO du dV (3)
where UO is shorthand for (UO1, . . . , UOn) and u is shorthand
for (u1, . . . , un). 

In order to evaluate this integral, we generate samples from the posterior 
p(O, V |X) using the MCMC samplers, and compute the sample average of ppred(x), int p(u|UO)p(x|u, V ) du

Me and my supervisor think that predictive likelihood is only one relevant measure of model quality. An orthogonal measure is robustness: how much do parameters vary between different training sets? 
Consider this figure from Kevin Murphy's book Machine Learning, A Probablistic Perspective:
(figure of ring scatter plot and a mixture of factors fit)
No matter how we draw a sample from the ring, a model that captures the ring structure, would always fit more or less the same ring. But a mixture of factors would yield different orientations, sizes and numbers of the individual factors. The ring model is hard to vary while maintaining the same goodness of fit. The mixture model is easy to vary while maintaining the same goodness of fit.