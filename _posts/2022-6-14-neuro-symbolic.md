---
layout: post
title: 
katex: True
---

Will scaling deep learning produce human-level generality, or do we need a new approach? Can everything be learned from experience, or does learning require innate knowledge and reason? The question is as old as philosophy itself, but as we continue to develop learning systems that affect our lives in profound ways, we need to start answering it. 

Current deep learning has limitations. It lacks data efficiency, generalization ability, extrapolation, robustness to adversarial attacks, compositionality, causality, symbolic reasoning, and several more. Most people agree that these will have to improve if we're to develop AGI. But they disagree about how to do it. 

Proponents of the scaling hypothesis argue that, just as the human brain is a giant neural network trained on years of experience, and just as we've seen ever more impressive and surprising abilities emerge with more data and compute, these limitations will likewise be overcome by adding data and compute.

Perhaps the most common view, is that while deep learning and scale are foundational, we do need to add new concepts, but they will have to be compatible with gradient-based learning.

Proponents of neuro-symbolic AI argue that the above limitations are fundamental to deep learning, no matter how deep it gets. They want to combine deep learning with a radically different approach, symbolic AI. 

On all sides, proponents cite well-known results in support of their views; The universal approximation theorem means that neural networks can represent any function. Bayes' theorem implies that simple models generalize better. No amount of correlation can ever imply causation. 

Each side interprets the abilities and shortcomings of current systems in accordance with their views. Deep learning continues to deliver amazing results, largerly thanks to scale. But it still struggles with full self-driving, medical diagnosis and other long-tailed, billion-dollar problems that we've been working on for a while now. 

I'll present my take on DL's generalization ability. I'll argue that while DL can indeed learn anything given enough data and compute, many problems place harsh constraints on both. Loss functions are minimized by "shortcuts", i.e. solutions that don't generalize beyond the data distribution. But you and I generalize out-of-distribution all the time. In fact, let's try it now:

### Tufa or not tufa.

Take a look at the following objects:
![tufa]({{ site.url }}/images/tufa.png "Various novel objects.")
<center> Various novel objects.</center>

[Credit: Josh Tenenbaum](https://www.slideserve.com/victoria/ucsd07-powerpoint-ppt-presentation)

The ones highlighted in red are _tufas_. Can you spot other tufas among the objects? Can you imagine what a tufa looks like viewed from above? Do you think a tufa is hard and unyielding, or soft and elastic? If you held a tufa by the "roots" and spun it around in a circle, what could happen to the other end? Would a tufa float in water, or sink?

Most likely you can answer all these questions without too much trouble. Perhaps you're not exactly sure of the answers, but you have a rough idea. In general, you can probably make good guesses about various properties of tufas, and what would happen to one in different hypothetical scenarios. 

In recent debates about how to develop artificial intelligence, two viewpoints dominate. Proponents of the scaling hypothesis argue that deep learning is on track to acquire all the cognitive abilities featured by human intelligence, all we need is to scale it up. Opponents argue that deep learning is lacking in some fundamental ways, which mere scale cannot overcome. According to them, adding symbolic reasoning, causality, and program induction are the next step. 

I think the latter viewpoint is correct. But the best arguments for it haven't been part of the debate. The tufa example above, borrowed from cognitive scientist Joshua Tenenbaum, is meant to showcase some of the abilities that we all possess, that form a good portion of our intelligence. These abilities are not within the scope of deep learning. Present-day deep learning certainly hasn't produced anything like it. However, the scaling hypothesis is not easily falsified. As large as modern neural networks are, they are still insect-sized. By analogy, wouldn't a much larger network naturally acquire all the abilities that _our_ giant neural network has? I will argue that while the analogy works in some ways, overall, reasoning from first principles supports the other view: extrapolation, generalizing out-of-distribution, few-shot-learning and causal reasoning cannot emerge from scaling DL. Indeed, many of the apparent empirical successes of DL are being misinterpreted. 


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

### Machine learning: a probabilistic perspective in 5 minutes

To examine deep learning from a broader perspective, we need a quick primer of machine learning. I'll assume some familiarity with probability theory, and present a probabilistic perspective on ML. This will later allow us to see what is missing in deep learning, and help us bridge the gap between the neural and the symbolic.

Nature generates some data, $$\mathbf{X}$$, by sampling it from the data distribution $$p(\mathbf{X})$$. $$\mathbf{X}$$ can be a matrix of numbers representing, say, the height and weight of people, a collection of images, a text corpus, a time series, etc.
In unsupervised learning, we seek to estimate $$p(\mathbf{X})$$, often by positing additional hidden variables $$\mathbf{Z}$$, which capture some underlying regularities in $$\mathbf{X}$$, such as clusters or latent factors. In supervised learning, we think of one of the variables as the output $$\mathbf{y}$$, and want to estimate $$p(\mathbf{y}|\mathbf{X})$$, i.e. the posterior probability of getting $$\mathbf{y}$$ given that we observed $$\mathbf{X}$$. To keep notation clean, let's stick with the unsupervised case for now.

We need some way of representing $$p(\mathbf{X})$$, a model $$\mathcal{M}$$, parametrized by weights, $$\mathbf{w}$$: $$p(\mathbf{X}, \mathbf{w} | \mathcal{M})$$. The "$$\mathcal{M}$$" reminds us that we're looking at the problem through a particular lens, constituted by our choice of hyperparameters and whatever domain knowledge and inductive biases we bring to bear on the problem. This lens can only reveal $$p(\mathbf{X} | \mathcal{M})$$, which is similar to $$p(\mathbf{X})$$ insofar as the model is able to describe the problem well.
Usually in ML, fitting a model means finding a value of $$\mathbf{w}$$ that agrees with the prior and the data in some formal way.
In Bayesian inference, we get an entire probability distribution over $$\mathbf{w}$$, the posterior:

$$
\begin{aligned}
p(\mathbf{w} | \mathbf{X}, \mathcal{M}) = \frac{p(\mathbf{X} | \mathbf{w}, \mathcal{M})p(\mathbf{w} | \mathcal{M})}{p(\mathbf{X} | \mathcal{M})}
\end{aligned}
$$

In a real sense, the posterior gives us not one model, but an entire set of models, with various posterior probabilities. There may not be a single most likely model - the posterior distribution could have many modes. Especially in deep learning, the likelihood, and consequently the posterior, may be high for a wide variety of different weights, corresponding to different functions - that will be important later.

Thus, to make predictions about future data $$\mathbf{x}$$, we want to ensemble all of the models, weighted by their posterior probability. This is done in the posterior predictive, $$p(\mathbf{x} | \mathbf{X}, \mathcal{M})$$. We get it by marginalizing $$\mathbf{w}$$ out, over the posterior:

$$
\begin{aligned}
p(\mathbf{x} | \mathbf{X}, \mathcal{M}) = \int p(\mathbf{x} | \mathbf{w}, \mathcal{M})p(\mathbf{w} | \mathbf{X}, \mathcal{M}) d\mathbf{w}
\end{aligned}
$$

We can also marginalize \mathbf{w} out over the prior, which gives us the posterior predictive's lesser known little brother, the prior predictive:

$$
\begin{aligned}
p(\mathbf{x} | \mathcal{M}) = \int p(\mathbf{x} | \mathbf{w}, \mathcal{M})p(\mathbf{w} | \mathcal{M}) dw
\end{aligned}
$$

Evaluating the prior predictive at the data we observed, $$p(\mathbf{X} | \mathcal{M})$$ gives us that rare and coveted quantity, the marginal likelihood, aka model evidence. The model evidence unlocks a second level to Bayes' theorem:

$$
\begin{aligned}
p(\mathcal{M} | \mathbf{X}) = \frac{p(\mathbf{X} | \mathcal{M})p(\mathcal{M})}{p(\mathbf{X})}
\end{aligned}
$$


Of course, we still can't view $$p(\mathbf{X})$$ through a truly perfect, universal lens, merely a wider one. So let's consider a set of models, $$\mathbb{M}$$:

$$
\begin{aligned}
p(\mathcal{M} | \mathbf{X}, \mathbb{M}) = \frac{p(\mathbf{X} | \mathcal{M}, \mathbb{M})p(\mathcal{M} | \mathbb{M})}{p(\mathbf{X} | \mathbb{M})}
\end{aligned}
$$


The difference between the first and second level of Bayes, is that the first level tells us how much each parameter setting for a model is made likely by conditioning on the observed data. But the second level tells us how much each model, which may be entirely different in nature from the others, is made likely by conditioning on the data. Bayes' theorem itself works exactly the same on the second level, so there's nothing new here mathematically. What's interesting is that we have a principled way of assigning probabilities to models. The catch is that the model evidence is notoriously hard to compute - in fact, it's the main difficulty already in the first level, where it appears as the normalizing constant in the denominator. That difficulty is why Bayesian inference is approximated for all but the simplest models (or models with carefully chosen structure and distributions).

Before this probabilistic perspective bears its first fruit, let's briefly relate this probabilistic perspective to deep learning. Not counting Bayesian deep learning, DL doesn't bother with even the first Bayes level, let alone the second. Rather than computing or approximating the posterior over the weights, SGD simply finds the single value of weights that maximize the likelihood $$p(\mathbf{X} | \mathbf{w}, \mathcal{M})$$ [[^1]] The derogatory term for this among Bayesians is "point-estimate" - the posterior distribution is approximated by a single point. The prior p(w | M) is also present behind the scenes in deep learning, as a regularization term in the loss function.[[^2]]

### Generalization: from hashtables to AGI

### Level 0: No generalization

Caching

### Level 1: Local generalization

Apart from providing a unifying framework for disparate phenomena, a probabilistic perspective on ML sheds light on the notions of generalization, bias-variance decomposition and Occam's razor in a single concept.

Informally, generalization is the ability to use past experience to perform in novel situations. In supervised learning, we usually assume that observed and future data are I.I.D., independent and identically distributed. This allows us to estimate generalization by how well the model performs on a held-out dataset, that is, data which didn't appear in the training set. If the training set and test set are independent and representative samples from the data distribution, we have an unbiased estimate. It is just an estimate - usually we can't _measure_ generalization by testing the model on every conceivable input and output. In unsupervised learning, a probabilistic perspective allows us to estimate generalization by how much probability mass the model assigns to held-out data. Since a distribution obeys conservation of probability mass, unsupervised models can overfit by assigning too much of it to the training data, getting good training loss at the cost of underestimating the probability of future outcomes.

When engineers apply a formal method to real-world problems, the assumptions and idealizations that make the method work often conflict with the nebulous nature of the problem. So it is with the notion of "data distribution". Consider the problem of predicting future temperatures in your city given past measurements. After training on an hours' worth of continuous data, learning that the temperature changes linearly, generalizes very well for a few hours, until it doesn't. Training for 48 hours, learning the day-night cycle generalizes quite well for a few days, until it doesn't. Training for a month, learning the day-night cycle plus a trend generalizes fine for a few months, until it doesn't. Training for a millenium, induction on past experience doesn't prepare the model for anthropogenic climate change. The only constant is change.

https://d2l.ai/chapter_multilayer-perceptrons/environment.html#a-taxonomy-of-learning-problems
We call this phenomenon _non-stationarity_, or _distribution shift_, with special cases _covariate shift_ (when $$p(\mathbf{y} | \mathbf{x})$$ is stationary but $$p(\mathbf{x})$$ isn't), _label shift_ (vice versa). Distribution shift is usually much more subtle than the educational example above, and there is a rich folklore in machine learning with cautionary tales of neglecting to consider distribution shift (Tanks), but we keep doing it. Kaggle winners are usually useless in practice. Most recently, a large swathe of NN-driven medical diagnosis tools have been defeated by models picking up on idiosyncracies of cameras in different hospitals. 

![prob_gen]({{ site.url }}/images/probabilistic_generalization_wilson_izmailov.png "A probabilistic perspective of generalization.")
<center> A probabilistic perspective of generalization.</center>
[Credit: Andrew Gordon Wilson and Pavel Izmailov](https://arxiv.org/pdf/2002.08791.pdf)

It's important to understand that bad generalization ability doesn't imply low skill level, and high skill level doesn't imply good generalization ability. 

	How do NNs learn? Geometric analogy - they unfold a crimpled ball of paper through a chain of geometric transformations.
		Experiment: Initialize an autoencoder with a 2d latent space. 
	Latent space interpolation.
	MLPs and hierarchical features.
		You may ask, if the crux of the issue is to have multiple successive layers of repre-
		sentations, could shallow methods be applied repeatedly to emulate the effects of
		deep learning? In practice, there are fast-diminishing returns to successive applica-
		tions of shallow-learning methods, because the optimal first representation layer in a three-
		layer model isn’t the optimal first layer in a one-layer or two-layer model. What is transforma-
		tive about deep learning is that it allows a model to learn all layers of representation
		jointly, at the same time, rather than in succession (greedily, as it’s called). With joint
		feature learning, whenever the model adjusts one of its internal features, all other fea-
		tures that depend on it automatically adapt to the change, without requiring human
		intervention. Everything is supervised by a single feedback signal: every change in the
		model serves the end goal. This is much more powerful than greedily stacking shallow
		models, because it allows for complex, abstract representations to be learned by
		breaking them down into long series of intermediate spaces (layers); each space is
		only a simple transformation away from the previous one.
	CNNs and translation invariance.
	RNNs and local relationships in sequence.
	Transformers and Hopfield networks.

Distribution shift.
Two models can generalize equally well on one test set, but differently on another (e.g. recognizing a cat by shape of vs texture)
Shortcut learning.
OOD generalization.
https://twitter.com/sirbayes/status/1537177495866327040
https://jacobbuckman.com/2022-06-14-an-actually-good-argument-against-naive-ai-scaling/?utm_source=pocket_mylist

### Do we need anything beyond local generalization?

You can always buy any skill level by injecting priors or data. But that's not intelligence, because intelligence is generalization ability.
However, that's only a good argument if generalization ability is a good definition of intelligence, otherwise it becomes circular.
So the task remains to argue in what generalization gives us that we can't get without it. In short: adaptability.

Some people understand all this and still believe the scaling hypothesis, or at least in the power of interpolative models trained on big data.

#### Why try to extrapolate when you can just buy skills by interpolating more data?
Either because because they believe interpolation is all that is needed:https://www.gwern.net/docs/ai/scaling/2020-hasson.pdf
	"By widening the interpolation zone, the model’s inability to extrapolate becomes less and less of a liability (Feldman,2019; Radhakrishnan et al., 2019)."

	What does "less and less" mean? What is the rate of widening the interpolation zone vs the rate of distribution shift?
	We need more than just to widen the zone. We need a dense sampling of the space. Density is points over volume, and volume scales exponentially in the number of dimensions.

	"The same direct-fit procedures can be expanded to fit arbitrarily complex data structures(Cybenko, 1989; Funahashi, 1989; Hornik et al., 1989; Raghu etal., 2017). The ability of over-parameterized models to robustly fit complex data structures provides unparalleled predictive po-wer within the interpolation zone, making them uniquely suit-able for multidimensional, real-life situations for which no simple, ideal model exists."
		Unparalleled predictive power? Show "Composing graphical models..."
		No simple model exists? The unreasonable effectiveness of mathematics mights as well have been called the unreasonable effectiveness of simple, symbolic representations.
		There is a simple model for computer vision: inverse graphics
		In general, the causal direction is simple: Max Welling's response to the bitter lesson
Or they believe interpolation in high dimensions amounts to extrapolation: https://arxiv.org/abs/2110.09485
	Does a low-dimensional extrapolation problem become a interpolation problem in high dimensions? Sort of like how quantum mechanics is linear, but in space where every particle has a dimension.

Or because they believe NNs are "lazy", and are forced to learn more and more generalizable patterns with larger scale: 
	"So, the larger the model, the better, if there is enough data & compute to push it past the easy convenient sub-models and into the sub-models which express desirable traits like generalizing, factorizing perception into meaningful latent dimensions, meta-learning tasks based on descriptions, learning causal reasoning & logic, and so on. If the ingredients are there, it’s going to happen.": https://www.gwern.net/Scaling-hypothesis#blessings-of-scale
		NNs are still too flexible. 
		The ingredients aren't there - you can represent any function with an NN to any degree of accuracy, but you can't learn any function.
		A larger model trained on larger data learns a different latent manifold, so that interpolation leads to different predictions.
			This is what makes pre-training work.

Experiments:
Test extrapolation as function of scale for MLP on x -> x^2
Test extrapolation as function of scale for symbolic regression on x -> x^2
Test extrapolation as function of scale for Transformer on binary x -> x^2


If we sample a large enough and diverse enough dataset, do we even need OOD? Don't NNs learn more generalizable patterns with larger scale by compressing it? Haven't we seen few-shot learning and meta-learning emerge with scale?

Gwern says yes:
"DL enjoys an unreasonably effective blessing of dimensionality—just simply training a big model on a lot of data induces better properties like meta-learning without even the slightest bit of that architecture being built in; and in general, training on more and harder tasks creates ever more human-like performance, generalization, and robustness."
...and proceeds to cite a bunch of papers where scale improves in-distribution generalization.


*Will everything that's missing from DL in order to generalize more broadly, causality, reasoning, symbols, programs, emerge with scale?*
NNs can represent it.
But they cannot learn it from data.
A "true" model converges at some point. Inverse graphics works on everything.

### Level 1.5: Out-of-distribution generalization

Solving OOD generalization would be a huge boon, economically. Upkeep costs in AI companies are generally much higher than in tradiational software companies, because of this.



### Level 2: Causality and broad generalization

Predictions vs Explanations


### Level 3: Imagination, creativity, analogy and extreme generalization

But if the goal is AGI, OOD generalization is too low a bar. 

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

[^1]: Most loss functions are log-likelihoods.
[^2]: Briefly, a Gaussian prior on the weights, corresponds to weight decay/L2 regularization, when your point-estimate is the mode of the posterior (MAP), found by minimizing - log p(X | w, m)p(w | m)  = - log p(X | w, m) - log p(w | m) = MSE + L2norm(w)