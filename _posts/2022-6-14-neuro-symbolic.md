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
<center> Various novel objects.</center>

[Credit: Josh Tenenbaum](https://www.slideserve.com/victoria/ucsd07-powerpoint-ppt-presentation)

The ones highlighted in red are _tufas_. Can you spot other tufas among the objects? What would a tufa look like viewed from above? What if the lighting was coming from below, and was blue in color? If you held a tufa by the "roots" (I'm sure you know what I mean) and spun it around in a circle, what would happen to the other end? What sound would a tufa make if you dropped it on the floor?

Most likely you can answer all these questions without too much trouble. Perhaps you're not certain exactly what the answers are, but you have a rough idea. Probably you have many more rough ideas about what properties tufas are likely to have.  

In recent debates about building artificial intelligence, two viewpoints dominate. Proponents of the scaling hypothesis argue that deep learning is on track to acquiring all the cognitive abilities that characterize human intelligence, all you need is to scale it up. Others argue that deep learning is lacking in some fundamental ways, which mere scale cannot overcome. According to them, adding symbolic reasoning, causality, and program induction are the next step. 

I think the latter viewpoint is correct. But the best arguments for it haven't been part of the debate. The tufa example above, borrowed from cognitive scientist Joshua Tenenbaum, is meant to showcase some of the abilities that we all possess, that form a good portion of our intelligence but which are not within the scope of deep learning. Present-day deep learning has not produced anything like it. However, the scaling hypothesis is not easily falsified. As large as modern neural networks are, they are still insect-sized. By analogy, wouldn't a much larger network naturally acquire all the abilities that _our_ giant neural network has? I will argue that while the analogy is compelling in some contexts, reasoning from the first principles of machine learning supports the other view: abilities like extrapolation, generalizing out-of-distribution, few-shot-learning and causal reasoning cannot emerge from scaling DL. Indeed, many of the apparent empirical successes of DL are being misinterpreted. 


### Deep learning: a bittersweet lesson

Deep learning is amazing. Tasks that once required specialized methods, or were impossible to even approach, are now routinely solved with a single method:
1. Gather a large training dataset of input -> output examples
2. Initialize a large neural network (NN), a function composed of layers of differentiable transformations parameterized by weights
3. Minimize a loss function wrt. all the weights using stochastic gradient descent (SGD)

This method gives us superhuman performance at the game of Go, computer vision that 10 years ago would've been unimaginable, speech recognition, machine translation and now, apparently, mastery of natural language. The secret to powerful deep learning is simply scaling up the basic method, meaning

1. Train on lots of data
2. Have lots of parameters
3. Perform lots of SGD iterations on lots of GPUs (compute)

Scale is often the only difference between not being able to solve a task at all, to achieving superhuman performance. Some researchers, including heavyweights like Richard Sutton, Nando de Freitas and Ilya Sutskever, believe scale will take deep learning all the way to AGI: a system that can learn not just a single narrow task at a time, but any task a human can learn, from perception to board games to engineering to even AI research itself. The scaling hypothesis, as argued by Gwern [here](https://www.gwern.net/Scaling-hypothesis#scaling-hypothesis), "has only looked more and more plausible every year since 2010". The bitter part of the lesson is that most of the intellectual effort poured into AI has seemingly been for nought. Scale is boring, but practical.

### Machine learing: a probabilistic perspective

To see deep learning from a broader perspective, we need a quick primer of machine learning 101. I will assume some familiarity with basic ML and probability theory, and present a probabilistic perspective on ML. This will later allow us to see what deep learning is missing, and help us bridge the gap between the neural and the symbolic.

Nature generates some data, $$\mathbf{X}$$, by sampling it from the data distribution $$p(\mathbf{X})$$. X can be a matrix of numbers representing, say, the height and weight of people, a collection of images, a text corpus, a time series, etc.
In unsupervised learning, we seek to estimate $$p(\mathbf{X})$$, often by positing additional hidden variables Z, which capture some underlying regularities in X, such as clusters or latent factors. In supervised learning, we think of one of the variables as the output y, and want to estimate p(y|\mathbf{X}), i.e. the posterior probability of getting y given that we've observed X. To keep notation clean, let's stick with the unsupervised case for now.

We need some way of representing $$p(\mathbf{X})$$. We do this by selecting a model, m, which usually is parametrized by weights, w: p(y, \mathbf{X}, w | \mathcal{M}). The $$\mathcal{M}$$ to the right of the conditioning bar reminds us that we're not accessing raw reality. Rather, we're looking at it through a particular lens, constituted by our choice of hyperparameters and whatever domain knowledge and inductive biases we bring to bear on the problem. This lens reveals only p(X | m), which, if our model is a good representation of reality, is similar to $$p(\mathbf{X})$$. But where did w go? 
Bayesian inference gives us the posterior of w:
$$p(w | \mathbf{X}, m) = p(\mathbf{X} | w, m)p(w | m) / p(\mathbf{X} | m)$$
To make predictions about future data, we're interested in the posterior predictive, p(x | X, m). We get it by *marginalizing* w out, over the posterior:
$$p(x | \mathbf{X}, m) = int p(\mathbf{x} | \mathbf{X}, w, m)p(w | \mathbf{X}, m) dw$$
We can also marginalize w out over the prior, which gives us the posterior predictive's lesser known little brother, the prior predictive:
$$p(\mathbf{x} | m) = int p(\mathbf{x} | w, m)p(w | m) dw$$
Evaluating the prior predictive at the data we observed, $$p(\mathbf{X} | m)$$
gives us that rare and coveted quantity, the marginal likelihood, aka model evidence. The model evidence unlocks a second level to Bayes' theorem:
$$p(m | \mathbf{X}) = p(\mathbf{X} | m)p(m)/$$p(\mathbf{X})$$$$
Of course, we still can't view $$p(\mathbf{X})$$ through a truly perfect, universal lens, merely a wider one. So let's consider a set of models, M:
$$p(m | \mathbf{X}, M) = p(\mathbf{X} | m, M)p(m | M)/p(\mathbf{X} | M)$$
The difference between the first and second level of Bayes, is that the first level tells us which parameters for the model are made likely by conditioning on the observed data. But the second level tells us which model, each of which may be entirely different in nature, is made likely by conditioning on the data. Bayes' theorem itself works exactly the same on the second level, so there's nothing to see here mathematically. What's interesting is the fact that we have a principled way of assigning probabilities to models. The catch is that the model evidence is notoriously hard to compute - in fact, it's the main difficulty already in the first level, where it appears as the normalizing constant in the denominator. 

Before this probabilistic perspective bears its first fruit, let's briefly relate this probabilistic perspective to deep learning. Not counting Bayesian deep learning, regular DL, with all its successes, doesn't even bother with the first Bayes level, let alone the second. Rather than computing or approximating the posterior over the weights, SGD simply finds the single value of weights that maximize the likelihood p(X | w, m) [^1] The derogatory term for this among Bayesians is "point-estimate" - the posterior distribution is approximated by a single point. The prior p(w | M) also appears in deep learning, as a regularization term in the loss function.[^2]

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

[^1] Most loss functions are log-likelihoods.
[^2] Briefly, a Gaussian prior on the weights, corresponds to weight decay/L2 regularization, when your point-estimate is the mode of the posterior (MAP), found by minimizing - log p(X | w, m)p(w | m)  = - log p(X | w, m) - log p(w | m) = MSE + L2norm(w)