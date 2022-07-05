---
layout: post
title: 
katex: True
---

Will scaling deep learning produce human-level generality, or do we need a new approach? You may have read the exchange between Scott Alexander and Gary Marcus, and felt that there are some good arguments on both sides, some bad ones, but few arguments that go beyond analogy and handwaving. If you haven't read the exchange, here it is: [SA](https://astralcodexten.substack.com/p/my-bet-ai-size-solves-flubs?s=w), [GM](https://garymarcus.substack.com/p/what-does-it-mean-when-an-ai-fails), [SA](https://astralcodexten.substack.com/p/somewhat-contra-marcus-on-ai-scaling?s=r), [GM](https://garymarcus.substack.com/p/does-ai-really-need-a-paradigm-shift?s=w#footnote-anchor-1). 

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
<center> Among the set of all possible rules, only some solve the
training data. Among the solutions that solve the training data, only some generalise to an i.i.d. test
set. Among those solutions, shortcuts fail to generalise to different data (o.o.d. test sets), but the
intended solution does generalize.</center>

[Credit: Geirhos et al.](https://arxiv.org/abs/2004.07780)

So, the scaling hypothesis says that at large enough scale, the lazy, shortcut solution is the desired one. 

#### Assumptions underlying the scaling hypothesis

This requires that the desired solution
1. Can be represented by a practically sized NN
2. Has a lower loss than shortcut solutions
3. Can be found by gradient descent

Using the illustration above, we can imagine scaling the model size as the set of rules learnable by ML model #2 expanding, since more parameters means more representable functions. Scaling data and compute corresponds to the set of training and shortcut solutions contracting, since a larger dataset is fit by fewer rules than a small dataset. Visually, the scaling hypothesis requires that (1) the rules learnable by the NN eventually include the orange dot, and that (2 and 3) the blue and beige sets contract around it. 

I will argue that this never actually happens. As long as we use NNs, which are large piecewise-linear functions, as representations, 1. won't hold in general. Roughly speaking, any function can be approximated arbitrarily well as a piecewise linear function - but given a finite amount of pieces, the "arbitrarily well" part goes out the window.
More importantly, as long as the loss we're minimizing is empirical risk, meaning we optimize training set performance, 2. and 3. won't hold. An empirical risk minimizer doesn't care what the NN is doing outside the training sample, so it has zero incentive to find a solution that generalizes everywhere. The minimizer will not allocate any pieces outside the training data where it gains nothing from it.

Even if all three assumptions held, the stronger hypothesis "Scaling NNs is sufficient for general intelligence" further requires that

1. There is enough data and compute to solve the problems that general intelligence can solve
2. The data and compute is available within relevant time frames

I will argue that mundane problems that we solve everyday put far harsher constraints on data, time and compute, than NNs can accommodate.

Gwern points out some of these assumptions himself:

>Sure, if the model got a low enough loss, it’d have to be intelligent, but how could you prove that would happen in practice? (Training char-RNNs was fun, but they hadn’t exactly revolutionized deep learning.) It might require more text than exists, countless petabytes of data for all of those subtle factors like logical reasoning to represent enough training signal, amidst all the noise and distractors, to train a model. Or maybe your models are too small to do more than absorb the simple surface-level signals, and you would have to scale them 100 orders of magnitude for it to work, because the scaling curves didn’t cooperate. Or maybe your models are fundamentally broken, and stuff like abstraction require an entirely different architecture to work at all, and whatever you do, your current models will saturate at poor performance. Or it’ll train, but it’ll spend all its time trying to improve the surface-level modeling, absorbing more and more literal data and facts without ever ascending to the higher planes of cognition as planned.

but then seems to regard GPT-3 and more recent models as proof by construction that the assumptions hold. I don't.

Critiquing these models is a tricky business. It's very easy to get strawmanned, or bogged down in tangential philosophical debates, so I want be clear:

1. Large language and image models are incredible, surprising and useful achievements
2. Their performance is no indication that we're making progress on AGI

It's difficult to convincingly argue for the latter point, because we all share an intuition that, if a technology works when we test it, then it is sound. Criticizing a large language model for making silly mistakes given obscure prompts, which they get right in the next version, brings to mind the "God of the gaps" rhetoric of creationists. 

But I'm not arguing that large-scale deep learning won't ever be able to do task X, or that mistakes A, B and C prove that it's not "really understanding", or that "mere pattern matching" isn't intelligence [^1]. 
If we want to know whether large-scale DL is progressing towards AGI, we should not be evaluating how well the models perform the tasks they are trained on, but their ability to generalize. Generalization is what general intelligence is for, and where deep learning is making no progress.

What is happening instead, is that as we scale up deep learning models, they get better performance on a test set of statistically identical examples. Contrary to what Gwern wrote, there is no pressure on the models to start generalizing better as we scale up, especially since we're increasing the model size, rather than decreasing it. On the contrary, the exact same, limited generalization ability is much more performant when you have a billion more examples to generalize from. It doesn't matter whether you use the data directly to train on, or indirectly by using a pre-trained model. 

#### What do neural networks learn?

Neural networks are non-linear transformations from one vector space to another. The transformation is performed in steps, layer by layer, from vector space to [vector space](http://colah.github.io/posts/2015-01-Visualizing-Representations/#neural-networks-transform-space). 
Francois Chollet compares the transformation an NN performs to uncrumpling a paper ball. The output vector space, or latent space manifold, is like an uncrumpled, flat piece of paper. On the latent manifold, we can draw a straight line between any two points on the manifold, and every point on the line will also lie on the manifold. If the NN is a classifier, the right latent space manifold allows it use such a line to discriminate between classes. In the case of regression, given any input in the convex hull ("somewhere in the middle") of known points, we can predict the output by interpolation on the latent manifold. Generative models can sample from the latent manifold to produce new samples from the (empirical) data distribution. The defining characteristic of the latent manifold is a meaningful notion of distance (or similarity). 

Distances between observations change in subsequent vector spaces. In the raw pixel space of images, the distance is short if the same pixels have the same colors. But in the subsequent vector spaces learned by a deep NN, the distances will depend on increasingly abstract features. In a convolutional layer, the representation of an image from the previous layer is compared to a number of "prototypes". If two images match the same prototypes to the same degree, they are similar in that space. Early layers of convolutional nets learn to [detect edges](https://distill.pub/2017/feature-visualization/), and later layers detect textures, patterns, parts of objects and objects. It may be that if a test image is similar to a training image in "texture space", that's enough to correctly classify it. In fact, it has been observed a number of times that NNs rely strongly on textures to classify objects, largely ignoring their shape. That's not a problem, as long as the distribution of test images is the same as the distribution of training images - the textures will always predict the same object that shapes predict. 

![shortcuts]({{ site.url }}/images/starmoon.png "Example of shortcut learning.")
<center> Trained on images of stars and moons (top row), a three-layer MLP correctly classifies new examples from an i.i.d. test set. However, testing it on an o.o.d test set (bottom row) reveals the shortcut: The network has learned to associate object location with a category. </center>

[Credit: Geirhos et al.](https://arxiv.org/abs/2004.07780)

Note that a convolutional net wouldn't take this shortcut. We explicitly prevent this, by building in translation invariance into CNNs; by iterating over patches of the image, we compute the similarity to the same, reusable "prototype" feature. Thus the NN doesn't learn a "vertical edge in the upper right corner" feature, and a "vertical edge in the upper middle" and so on. It learns a single "vertical edge" feature, and the for-loops hardcoded into the convolution operation scan the entire image for it.

The end result is a piecewise function - in the case where the activation function is the ReLU, it's a piecewise linear function.

Conclusion: NNs generalize by [similarity-based abstraction](https://www.youtube.com/watch?v=3Nxe7J07TQY).

#### Never attribute to high-level abilities that which can be adequately explained by shortcut learning.



#### Implications of the scaling hypothesis

##### Bigger brains should be smarter
If all you need for general intelligence is a large neural network and lots of data, why aren't animals with human-sized or larger brains, and human-length or longer lives, as general as humans? Why aren't blue whales generally smarter than humans? Why don't non-human animals ever pick up human language[^2], when GPT-3 allegedly can?
This observation doesn't falsify the scaling hypothesis - there could be any number of extra reasons why the straightforward extrapolation doesn't hold. In general, it seems clear that natural intelligence is highly reliant on innate knowledge. It's possible that intelligence that relies on innate knowledge was more likely to evolve, given how evolution tends to exapt existing structures, but that scaling is an equally good, or better approach, if you're starting from scratch. But given that some animals already have large brains, and long lives, and there's at least some selection pressure for higher intelligence, then it would take a very good excuse to explain their lack of general intelligence away under the scaling hypothesis.

Most animals rely on innate knowledge and instincts. They don't seem able to infer causes - squirrels try to bury nuts in concrete, apes over-imitate copied behaviors. The most intelligent animals seem to be able to use and understand symbols, understand causes, and use them to improvise novel behaviors. 

### Intelligence is more than pattern matching: Why children are not deep learners

The child as scientist

The child as hacker

Social learning

#### Representing knowledge as probabilistic programs

Compositionality

#### Learning as probabilistic program synthesis

#### Causality as program editing


[^1]: I don't believe that deep learning models are really understanding, nor that pattern matching is all there is to intelligence, but that's not the point I'm making here.
[^2]: If you think parrots, or dogs/chimps that learn a small vocabulary of sign language should count as counter-examples, why specifically those animals, and not any large-brained animal?