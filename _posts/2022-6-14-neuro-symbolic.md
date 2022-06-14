---
layout: post
title: 
katex: True
---


Example: DL-based CV vs inverse graphics? DL-regression vs symbolic regression?
Thesis: Scaling deep learning won't produce flexible generalization ability. Symbolic computation is needed.
Supporting arguments:
1. DL can't extrapolate, only interpolate.
	1. Extrapolation matters for generalization.
		1. Show example of NN vs symbolic regression.
	2. Lack of OOD generalization makes a model brittle, and soon outdated.
2. DL can't infer causes, only associations.
	1. We need causality to explain things. Otherwise, we're just predicting.
	2. We need causality to figure out what to do to achieve our goals.
	3. We need causality for imagination and creativity.
3. DL requires a dense sampling of the latent manifold, and a large compute budget.



Take a look at the following objects:
![tufa]({{ site.url }}/images/tufa.png "Various novel objects.")
[[^2]]
<center> Various novel objects, some of them are tufas. Credit [Josh Tenenbaum](https://www.slideserve.com/victoria/ucsd07-powerpoint-ppt-presentation)  </center>
The ones highlighted in red are tufas. Can you spot other tufas among the objects? What would a tufa look like if you viewed it from above? What if the lighting was coming from below, and blue in color? How would you divide a tufa into parts, and how can the parts move, if at all? If you held a tufa by the "roots" and spun around in a circle, what would happen to the other end?
Most likely you can answer all these questions without too much trouble. How do you do that? How 



Deep learning is amazing. Tasks that once required specialized methods, or were impossible to even approach, are now routinely solved with a single method:
1. Gather a large training dataset of input -> output examples
2. Initialize a large neural network (NN), a function composed of layers of differentiable transformations parameterized by weights
3. Minimize a loss function wrt. all the weights using stochastic gradient descent (SGD)

This method gives us superhuman performance at the game of Go, computer vision that 10 years ago would've been unimaginable, speech recognition, machine translation and now, apparently, mastery of natural language. The secret to powerful deep learning is simply scaling up the basic method, meaning
1. train on lots of data
2. have lots of parameters
3. perform lots of SGD iterations (compute)

Scale is often the only difference between not being able to solve a task at all, to achieving superhuman performance. Some researchers, including heavyweights like Richard Sutton, Nando de Freitas and Ilya Sutskever, believe scale will take deep learning all the way to AGI: a system that can learn not just a single narrow task at a time, but any task a human can learn, from perception to board games to engineering to even AI research itself. The scaling hypothesis, as argued by Gwern [here](https://www.gwern.net/Scaling-hypothesis#scaling-hypothesis), "has only looked more and more plausible every year since 2010.".
Others disagree. Most vocally Gary Marcus, but also Francois Chollet and Joshua Tenenbaum, argue that while deep learning certainly is an important advance, general intelligence involves abilities that deep learning alone is incapable of producing, by design. They point out that where deep learning fails, it needs to be complemented by symbolic representations, discrete abstractions or a "language of thought", resulting in a neuro-symbolic apporach.

In this post, I will explain what is lacking in deep learning, why it matters, why scale doesn't help and how neuro-symbolic approaches already have helped and will continue to do so.

To start us off, we need a quick primer of machine learning 101. I will assume some familiarity with basic ML and probability theory, and present a probabilistic perspective on ML. This will later allow us to see what deep learning is missing, and help us bridge the gap between the neural and the symbolic.

Nature generates some data, X, by sampling it from the data distribution p(X). X can be a matrix of numbers representing, say, the height and weight of a set of people, a collection of images, a text corpus, a time series, etc.
In unsupervised learning, we seek to estimate p(X), often by positing additional hidden variables Z, which capture some underlying regularities in X, such as clusters or latent factors. In supervised learning, we think of one of the variables as the output y, and want to estimate p(y|X), i.e. the posterior probability of getting y given that we've observed X. To keep notation clean, let's stick with the unsupervised case for now.

We need some way of representing p(X). We do this by selecting a model, m, which usually is parametrized by weights, w: p(y, X, w | m). The m to the right of the conditioning bar reminds us that we're not accessing raw reality. Rather, we're looking at it through a particular lens, constituted by our choice of hyperparameters and whatever domain knowledge and inductive biases we bring to bear on the problem. This lens reveals only p(X | m), which, if our model is a good representation of reality, is similar to p(X). But where did w go? 
Bayesian inference gives us the posterior of w:
p(w | X, m) = p(X | w, m)p(w | m) / p(X | m)
To make predictions about future data, we're interested in the posterior predictive, p(x | X, m). We get it by *marginalizing* w out, over the posterior:
p(x | X, m) = int p(x | X, w, m)p(w | X, m) dw
We can also marginalize w out over the prior, which gives us the posterior predictive's lesser known little brother, the prior predictive:
p(x | m) = int p(x | w, m)p(w | m) dw
Evaluating the prior predictive at the data we observed, p(X | m)
gives us that rare and coveted quantity, the marginal likelihood, aka model evidence. The model evidence unlocks a second level to Bayes' theorem:
p(m | X) = p(X | m)p(m)/p(X)
Of course, we still can't view p(X) through a truly perfect, universal lens, merely a wider one. So let's consider a set of models, M:
p(m | X, M) = p(X | m, M)p(m | M)/p(X | M)
The difference between the first and second level of Bayes, is that the first level tells us which parameters for the model are made likely by conditioning on the observed data. But the second level tells us which model, each of which may be entirely different in nature, is made likely by conditioning on the data. Bayes' theorem itself works exactly the same on the second level, so there's nothing to see here mathematically. What's interesting is the fact that we have a principled way of assigning probabilities to models. The catch is that the model evidence is notoriously hard to compute - in fact, it's the main difficulty already in the first level, where it appears as the normalizing constant in the denominator. 

Before this probabilistic perspective bears its first fruit, let's briefly relate this probabilistic perspective to deep learning. Not counting Bayesian deep learning, regular DL, with all its successes, doesn't even bother with the first Bayes level, let alone the second. Rather than computing or approximating the posterior over the weights, SGD simply finds the single value of weights that maximize the likelihood p(X | w, m) FOOTNOTE: Most loss functions are log-likelihoods. The derogatory term for this among Bayesians is "point-estimate" - the posterior distribution, with all its spaciousness and complexity, has been reduced to a single, ugly spike of infinite probability density, with a name straight out of Mordor: the Dirac delta. The prior p(w | M) also appears in deep learning, as a regularization term in the loss function. FOOTNOTE: Briefly, a Gaussian prior on the weights, corresponds to weight decay/L2 regularization, when your point-estimate is the mode of the posterior (MAP), found by minimizing - log p(X | w, m)p(w | m)  = - log p(X | w, m) - log p(w | m) = MSE + L2norm(w)

Apart from providing a unifying framework for seemingly disparate techniques and phenomena, a probabilistic perspective on ML formalizes the notions of generalization, bias-variance decomposition and Occam's razor in a single concept.

- A probabilistic perspective of generalization
- Universal approximation theorem - what kinds of functions can NNs represent?
- What do NNs learn in practice?
	- Hierarchical features
	- Latent space interpolation
		- Input space vs latent space
	- Shortcut learning
		- Adversarial attacks
- The generalization spectrum: why extrapolation matters
- Probabilistic programming
	- Bayesian program learning
		- Buying skill with priors
	- Bayesian program synthesis
	- Programmable inference
	- Joshua Tenenbaum's vision
		- Game engine in your head
		- Learning as program synthesis
		- Language of thought
		- Child as hacker
	- Climbing the ladder of causation
- Neuro-symbolic AI
	- has been here all along
		- Convolutions use universal quantifiers
		- AlphaGo and MCTS
	- DreamCoder
		- DreamCoder is AI from a decade into the future
	- ARC challenge
	- Neuro-symbolic reinforcement learning
	- What's still missing:
		- Context
		- Relevance determination: Is consciousness really optional?
- Aligning next generation AI
	- Explainability and fairness need causality
	- Reward maximization vs CIRL