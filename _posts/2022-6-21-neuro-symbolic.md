---
layout: post
title: 
katex: True
---

Will scaling deep learning produce human-level generality, or do we need a new approach? You may have read the exchange between Scott Alexander and Gary Marcus, and felt that there are some good arguments on both sides, some bad ones, but few arguments that go beyond analogy and handwaving - arguments that take what we know about deep learning and intelligence, and look at what that knowledge implies. If you haven't read the exchange, here it is: [SA](https://astralcodexten.substack.com/p/my-bet-ai-size-solves-flubs?s=w), [GM](https://garymarcus.substack.com/p/what-does-it-mean-when-an-ai-fails), [SA](https://astralcodexten.substack.com/p/somewhat-contra-marcus-on-ai-scaling?s=r), [GM](https://garymarcus.substack.com/p/does-ai-really-need-a-paradigm-shift?s=w#footnote-anchor-1). 

I will argue for Marcus' position, but dive a little deeper than he does. I believe that symbolic representations, specifically _programs_, and learning as _program synthesis_, can provide data efficient and flexible generalization, in a way that deep learning can't, no matter how much we scale it. I'll show how _probabilistic programs_ can represent causal models of the world, which deep learning can't do, and why causal models are essential to intelligence. But I'll start by examining the opposing view, that scaling deep learning is sufficient for general intelligence. To that end, I'll quote from [Gwern's thorough essay on the scaling hypothesis](https://www.gwern.net/Scaling-hypothesis).


<div id="toc"></div>


### The scaling hypothesis and the laziness of deep learning

The scaling hypothesis is that

> we can simply train ever larger NNs and ever more sophisticated behavior will emerge naturally as the easiest way to optimize for all the tasks & data

Gwern cites a swathe of papers in support, interpreting them in such a way that the following picture emerges:

> “neural nets are lazy”: sub-models which memorize pieces of the data, or latch onto superficial features, learn quickest and are the easiest to represent internally. If the model & data & compute are not big or varied enough, the optimization, by the end of the cursory training, will have only led to a sub-model which achieves a low loss but missed important pieces of the desired solution. 

> Eventually, after enough examples and enough updates, there may be a phase transition (Viering & Loog 2021), and the simplest ‘arithmetic’ model which accurately predicts the data just _is_ arithmetic.
(...) if there is enough data & compute to push it past the easy convenient sub-models and into the sub-models which express desirable traits like generalizing, factorizing perception into meaningful latent dimensions, meta-learning tasks based on descriptions, learning causal reasoning & logic, and so on

I agree with the premise. Neural nets are indeed "lazy", in that their loss functions are minimized by "shortcuts", solutions that don't generalize beyond the data distribution. A figure from the paper "Shortcut Learning in Deep Neural Networks" by Geirhos et al. illustrates this well:

![shortcuts]({{ site.url }}/images/shortcuts.png "Shortcut learning in NNs.")
[Credit: Geirhos et al.](https://arxiv.org/abs/2004.07780)
<center> Among the set of all possible rules, only some solve the
training data. Among the solutions that solve the training data, only some generalise to an i.i.d. test
set. Among those solutions, shortcuts fail to generalise to different data (o.o.d. test sets), but the
intended solution does generalize.</center>


So, the scaling hypothesis says that at large enough scale, the lazy, shortcut solution is the desired one. 

#### Assumptions underlying the scaling hypothesis

This requires that the desired solution
1. Can be represented by a practically sized NN
2. Has a lower loss than shortcut solutions
3. Can be found by gradient descent

Using the illustration above, we can imagine scaling the model size as the set of rules learnable by ML model #2 expanding, since more parameters means more representable functions. Scaling data and compute corresponds to the set of training and shortcut solutions contracting, since a larger dataset is fit by fewer rules than a small dataset. Visually, the scaling hypothesis requires that (1) the rules learnable by the NN eventually include the orange dot, and that (2 and 3) the blue and beige sets contract around it. 

I will argue that this never actually happens. As long as we use NNs, which are large piecewise-linear functions, as representations, 1. won't hold in general. Roughly speaking, any function can be approximated arbitrarily well as a piecewise linear function - but given a finite amount of pieces, the "arbitrarily well" part goes out the window.
More importantly, as long as the loss we're minimizing is empirical risk, meaning we optimize training set performance, 2. and 3. won't hold. An empirical risk minimizer doesn't care what the NN is doing outside the training sample, so it has zero incentive to find a solution that generalizes everywhere. The minimizer will not allocate any pieces outside the training data where it gains nothing from it.

But there is a potential way around these limitations, which corresponds to a [weaker version of the scaling hypothesis](https://www.gwern.net/docs/ai/scaling/2020-hasson.pdf).
Proponents of this version acknowledge that NNs don't extrapolate, and their generalization ability is confined to an "interpolation zone". But given enough data to cover the relevant parts of the domain, the fact that we're not generalizing out of distribution won't be a problem - the shortcut solution will in practice be indistinguishable from the desired solution. 

The only relevant assumptions then are 

1. There is enough data and compute to solve the problems that general intelligence can solve
2. The data and compute is available within relevant time frames

This weaker version of the scaling hypothesis seems more reasonable, since it doesn't posit that deep learning starts generalizing better at scale, or finds solutions that it can neither represent nor reach through optimization. But viewed from a different perspective, it is far more radical, because it implies all of intelligence, including improvisation, imagination, creativity etc., is simply the ability to recall a similar circumstance from memory. No o.o.d. generalization needed.

I will argue that mundane problems that we solve everyday put far harsher constraints on data, time and compute, than NNs can accommodate. Such problems can only be solved by strong generalization.

Gwern points out some of these assumptions himself:

>Sure, if the model got a low enough loss, it’d have to be intelligent, but how could you prove that would happen in practice? (Training char-RNNs was fun, but they hadn’t exactly revolutionized deep learning.) It might require more text than exists, countless petabytes of data for all of those subtle factors like logical reasoning to represent enough training signal, amidst all the noise and distractors, to train a model. Or maybe your models are too small to do more than absorb the simple surface-level signals, and you would have to scale them 100 orders of magnitude for it to work, because the scaling curves didn’t cooperate. Or maybe your models are fundamentally broken, and stuff like abstraction require an entirely different architecture to work at all, and whatever you do, your current models will saturate at poor performance. Or it’ll train, but it’ll spend all its time trying to improve the surface-level modeling, absorbing more and more literal data and facts without ever ascending to the higher planes of cognition as planned.

but then seems to regard GPT-3 and more recent models as proof by construction that the assumptions hold. I don't.

The problem is not that GPT-3 or other large models don't perform well at task X, or that mistakes A, B and C prove that it's not "really understanding", or that "mere pattern matching" isn't intelligence. These objections are valid, but they're not relevant if merely scaled up versions of essentially the same models start generalizing. For the same reason, we should not be evaluating how well the models perform the tasks they are trained on, but their ability to generalize. Generalization is what general intelligence is for, and where deep learning is making no progress.

What is happening instead, is that as we scale up deep learning models, they get better performance on a test set of statistically identical examples. Contrary to what Gwern wrote, there is no pressure on the models to start generalizing better as we scale up, especially since we're increasing the model size, rather than decreasing it. On the contrary, the exact same, limited generalization ability is much more performant when you have a billion more examples to generalize from. It doesn't matter whether you use the data directly to train on, or indirectly by using a pre-trained model. 

#### What do neural networks learn?

Neural networks are non-linear transformations from one vector space to another. The transformation is performed in steps, layer by layer, from vector space to [vector space](http://colah.github.io/posts/2015-01-Visualizing-Representations/#neural-networks-transform-space). 
Francois Chollet compares the transformation an NN performs to uncrumpling a paper ball. The output vector space, or latent space manifold, is like an uncrumpled, flat piece of paper. On the latent manifold, we can draw a straight line between any two points, and every intermediate point will also lie on the manifold. This property underlies the magic of deep learning. If the NN is a classifier, the right latent space manifold makes classes linearly separable. In the case of regression, given any input in the convex hull ("somewhere in the middle") of known points, we can predict the output by interpolation on the latent manifold. Generative models can sample from the latent manifold to produce new samples from the (empirical) data distribution. The defining characteristic of the latent manifold is that distances between observations along the manifold make the given task easy to solve. 

In the raw pixel space of images, the distance is short if the same pixels have similar colors. But in the subsequent vector spaces learned by a deep NN, the distances will depend on increasingly abstract features. In a layer, the representation from the previous layer is compared to a number of "prototype" features (The comparison is simply a dot product between the representation vector and the prototype - if the vectors align, the dot product is large. Doing that for all the prototypes in the layer is what all those matrix multiplications are for). The next representation is the combined output of these similarity comparisons - the patterns encoded by the previous prototypes have been abstracted away. If two data-points matched the same prototypes to a similar degree, they are similar in that space, and the distance between them is short.

Early layers of convolutional nets learn to [detect edges](https://distill.pub/2017/feature-visualization/), and later layers detect textures, patterns, parts of objects and objects. It may be that if a test image is similar to a training image in "texture space", that's enough to correctly classify it. In fact, it has been observed that NNs rely strongly on textures to classify objects, largely ignoring their shape. That's not a problem, as long as the distribution of test images is the same as the distribution of training images - the textures will always predict the same object that shapes predict. Using textures for classifying objects is not "wrong", in any way that affects the loss function. It's a clever shortcut, albeit with limited applicability.

Here's a toy example of an even simpler shortcut:

![shortcuts]({{ site.url }}/images/starmoon.png "Example of shortcut learning.")
[Credit: Geirhos et al.](https://arxiv.org/abs/2004.07780)
<center> Trained on images of stars and moons (top row), a three-layer MLP correctly classifies new examples from an i.i.d. test set. However, testing it on an o.o.d test set (bottom row) reveals the shortcut: The network has learned to associate object location with a category. </center>


Note that a convolutional net wouldn't take this shortcut. We explicitly prevent this, by building in translation invariance into CNNs; by iterating over patches of the image, we compute the similarity to the same, reusable "prototype" feature. Thus the NN doesn't learn a "vertical edge in the upper right corner" feature, and a "vertical edge in the upper middle" and so on. It learns a single "vertical edge" feature, and the for-loops hardcoded into the convolution operation scan the entire image for it.

The end result is a piecewise function - in the case where the activation function is the ReLU, it's a piecewise linear function.

Conclusion: NNs generalize by [similarity-based abstraction](https://www.youtube.com/watch?v=3Nxe7J07TQY).

> Never attribute to high-level abilities that which can be adequately explained by shortcut learning.

#### NNs find shortcuts because they are "easy to vary"

The reason why NNs find shortcuts has to do with the bias-variance tradeoff. The bias-variance tradeoff roughly says that models that express a large hypothesis space, end up learning widely varying hypotheses, when compared across different training sets. Models that are constrained to a small hypothesis space, will learn the same hypotheses, no matter the training set. Thus if you increase the flexibility of a model by, say, adding parameters, the test error will decrease to a point, and then increase. At first, increasing flexibility allows the model to fit more signal, but at some point the added capacity is used to fit noise.
A better formalization of this is [Bayesian Occam's razor](http://mlg.eng.cam.ac.uk/zoubin/papers/05occam/occam.pdf), where this property follows from Bayesian model comparison, and the Bayesian formalism automatically selects the right complexity level. The most likely model given data is the one that is biased towards producing that data: linear data is best explained by a linear model. Sinusoidal data by a sine wave. In the extreme, a perfect model contains exactly one hypothesis, the one that produces the exact data we observed.

Neural networks sit closer to the other extreme. You can get a neural network to produce pretty much any data, which is why they are so widely applicable. Of course, there has to be more to deep learning than just flexibility, otherwise any sufficiently flexible learning algorithm would perform as well as deep learning does. The answer is that despite being very flexible, neural networks are still [biased](https://arxiv.org/pdf/2002.08791.pdf) towards learning certain structures over others. The most important of these inductive biases is towards hierarchical structure, as explained in the previous section - different complex objects are made of similar simple objects. Another is the smoothness bias - if you change the input just a little, the output also doesn't change much. As mentioned, CNNs are endowed with a powerful bias towards natural images, in which the hierarchical structure bias further disregards the position in the image (and importantly, the image is processed as a 2D matrix rather than a vector). This bias is [so apt that even without training CNNs](https://arxiv.org/abs/1711.10925) on any data, they perform very well on several computer vision tasks.
Transformers are equipped with an ingenious bias towards simple sequence to sequence functions - at each level of the hierarchy, the input is treated as a query sequence, to be compared against stored key sequences, returning a corresponding value sequence. This results in a kind of associative memory, as known from [Hopfield networks](https://arxiv.org/abs/2008.02217).

Inductive biases not only make learning more efficient, they are what allows the model to generalize. This is merely a corollary of the bias-variance tradeoff. Some believe that deep learning is somehow exempt from this, in that highly over-parametrized networks tend to work better, not worse, most clearly shown in the double descent phenomenon. A naive understanding of the bias-variance tradeoff predicts that highly flexible models should perform worse. But double descent is not at all unique to deep learning, and it doesn't contradict the bias-variance tradeoff or Bayesian model comparison framework in any way. What causes the apparent contradiction is not accounting for regularization, ensembling, and marginalizing over the posterior, all of which reduce variance. Whether it's [deep learning, or any model without fixed capacity](https://arxiv.org/pdf/1812.11118.pdf), a point estimate of the model parameters at the level of complexity that is just enough to fit signal and noise, will overfit. Beyond that complexity threshold, models enter an "interpolation zone", where the test performance keeps improving, if the model is regularized properly. If instead of fitting point estimate, one considers the full posterior, or even just an ensemble of point estimates, the mystery falls away - the performance of the ensemble improves monotonically with added complexity. 

The fact that the point estimate has high variance is especially true in deep learning, where the estimate is highly underdetermined by the data (the same dataset can be fit by many, many differently parameter settings). From a Bayesian perspective, the model evidence for large, flexible neural networks, is much lower than for simpler models - if your model can fit everything, it's not a good explanation for anything. 

Neural networks predicting anything while explaining nothing, turns from a philosophical to a practical problem once we consider generalization beyond an i.i.d. test set.


#### Implications of the scaling hypothesis

Here I move away from the specifics of deep learning, and compare it to how learning works in brains. The general argument is that the scaling hypothesis implies brains are X, brains aren't X, therefore the scaling hypothesis is false. The weak point of the argument is that the scaling hypothesis doesn't necessarily imply anything about biological learning. Human intelligence and AI could work entirely differently and yet be equally powerful. But given that deep learning is very simple, the brain is computationally powerful, vast amounts of data is provided through sensory experience, and at least a subset of the brain's representations and learning mechanisms is similar to deep learning, the argument provides evidence against the scaling hypothesis - if the brain doesn't do what deep learning does, and deep learning is simple and works, why doesn't the brain just do deep learning? These arguments also motivate the neuro-symbolic paradigm, which the rest of the blog post delves into.

##### Bigger brains should be smarter
If all you need for general intelligence is a large neural network and lots of data, why aren't animals with human-sized or larger brains, and human-length or longer lives, as general as humans? Why aren't blue whales generally smarter than humans? Why don't non-human animals ever pick up human language[^2], when GPT-3 allegedly can? Why aren't cows much smarter than crows?
As mentioned, these observations doesn't falsify the scaling hypothesis - there could be any number of extra reasons why the straightforward extrapolation doesn't hold. In general, it seems clear that natural intelligence is highly reliant on innate knowledge. It's possible that intelligence that relies on innate knowledge was more likely to evolve, given how evolution tends to exapt existing structures, but that scaling is an equally good, or better approach, if you're starting from scratch. But given that some animals already have large brains, and long lives, and there's at least some selection pressure for higher intelligence, then it would take a very good excuse to explain their lack of general intelligence away under the scaling hypothesis.

Most animals rely on innate knowledge and instincts. They don't seem able to infer causes - squirrels try to bury nuts in concrete, apes over-imitate copied behaviors. The most intelligent animals seem to be able to use and understand symbols, understand causes, and use them to improvise novel behaviors. 

##### Children shouldn't bother with sophisticated learning strategies - just imbibe data and wait for the brain to update

>People learn entirely new systems of interdefined concepts(Carey, 1985; Block, 1987) and do so in a way that is driven by hypotheses and goals rather than blind search (Carey, 2009; Chu et al., 2019), with a sensitivity to what is good in a hypothesis and what is suboptimal or even wrong (Schulz, 2012a; Chu & Schulz, 2020). Such richness is interesting because there are extremely simple algorithms guaranteed to discover optimal hypotheses, e.g. enumerating every possible hypothesis (Gold, 1967; Solomonoff,1964a).  That such a proposal sounds alien as a model of cognition, despite the extreme computational power of the human brain (Gallistel, 2017; Baum, 2004), reflects something not only about the complexity of the world that learners are trying to explain but also their reasons for learning and the sophistication of the machinery they bring to bear in doing so.


### Intelligence is more than pattern matching: Why children are not deep learners

The most popular succinct definition of intelligence is "cross-domain optimization", that is, the ability to achieve goals in a wide range of environments. This definition was formalized by Legg & Hutter, as an expectation of reward over a distribution of environments, weighted by the complexity of the environment. Given a set of environments and goals of varying complexity, an agent is more intelligent if it achieves higher reward on average than another agent, with simple environments counting more than complex ones (failing an easy task is worse than succeeding at a difficult task is good).

Francois Chollet proposes an alternative definition, "skill-acquisition efficiency", which emphasizes generalization, in a way that takes into account both experience, and prior knowledge. Given a set of tasks, an agent is more intelligent if it achieves higher reward using less experience and less built-in knowledge than another agent. 

Neither definition describes what intelligence _is_, in the sense of a tech spec that outlines its design and development process, and nobody is able to write one at present. But developmental psychologists study the only system known to reliably produce general intelligence, the human child. 

The child as scientist

The child as hacker

Social learning

#### Representing knowledge as probabilistic programs

In 2012, Tenenbaum et al. published "Human-level concept learning through probabilistic program induction", in which they present a model of handwritten characters that learned to generate new samples after having seen only a few examples, of a quality indistinguishable from human ones. The success of the model can be attributed to the fact that it encodes the actual handwriting process, down to each individual stroke of the pen. 

![handprog]({{ site.url }}/images/handwrittencharacterprobabilisticprogram.png "Probabilistic program that samples handwritten characters.")
[Credit: Lake et al.](https://www.cs.cmu.edu/~rsalakhu/papers/LakeEtAl2015Science.pdf)

Constructing the model thus required an expert to first understand how the writing process works, figure out what the basic moves were, how they can vary, how they combine, and how to render the character as an image. The expert then needs to be knowledge in both programming and statistics to formulate both the probabilistic program, which represents the writing process in code and the various sources of variation and noise as random variables, and then the inference algorithm, the program which inverts the writing process, from image to strokes - which probably is the hardest part. 

In terms of Chollet's definition of intelligence, we could say this program is just as unintelligent as an equivalent deep learning model would be. While we bought sample efficiency, robustness to distribution shift, and extrapolation ability, we payed for it with an almost complete lack of generality, a large amount of required domain expertise, and an even larger cost in technical expertise. While the model can well be said to "understand" handwriting, as opposed to a deep learning model, it's clear that all the understanding comes directly from the developer.

By writing computations as code, they become data that can be formally manipulated and analyzed (Abelson et al.,
241996). Programming languages thus become programs which take code as input and return
code as output.

It seems that our cognition exhibits the same property. Our thoughts can manipulate and analyze other thoughts. Systems of thought can scrutinize other systems of thought.

Compositionality

#### Learning as probabilistic program synthesis

#### Causality as program editing


[^1]: I don't believe that deep learning models are really understanding, nor that pattern matching is all there is to intelligence, but that's not the point I'm making here.
[^2]: If you think parrots, or dogs/chimps that learn a small vocabulary of sign language should count as counter-examples, why specifically those animals, and not any large-brained animal?