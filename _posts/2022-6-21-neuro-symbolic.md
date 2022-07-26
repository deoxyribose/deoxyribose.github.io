---
layout: post
title: 
katex: True
---

Will scaling deep learning produce human-level generality, or do we need a new approach? You may have read the exchange between Scott Alexander and Gary Marcus, and felt that there are some good arguments on both sides, some bad ones, but few arguments that go beyond analogy and handwaving - arguments that would take what we know about deep learning and intelligence, and look at what that knowledge implies. If you haven't read the exchange, here it is: [SA](https://astralcodexten.substack.com/p/my-bet-ai-size-solves-flubs?s=w), [GM](https://garymarcus.substack.com/p/what-does-it-mean-when-an-ai-fails), [SA](https://astralcodexten.substack.com/p/somewhat-contra-marcus-on-ai-scaling?s=r), [GM](https://garymarcus.substack.com/p/does-ai-really-need-a-paradigm-shift?s=w#footnote-anchor-1). 

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

There is a potential way around these limitations, which corresponds to a [weaker version of the scaling hypothesis](https://www.gwern.net/docs/ai/scaling/2020-hasson.pdf).
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

To keep things straight, let's distinguish between three versions of the scaling hypothesis:

1. Deep learning will never learn to extrapolate, but with enough data, parameters and compute, interpolation is enough.
2. Deep learning will learn to extrapolate, as long as there always are large amounts of data and compute available, which there is.
3. Deep learning will learn to extrapolate, and the more data and compute has been used, the less additional data and compute is needed, i.e. generalization ability increases with scale.

The first versions require less of neural networks themselves, but require the world to be more easily learnable. The latter versions don't require the world to be easily learnable, but they require neural networks to become better at scale. Without any math, let's look at what NNs do, and see if we can shed some light on these ideas.

#### What do neural networks learn?

Neural networks are non-linear transformations from one vector space to another. The transformation is performed in steps, layer by layer, from vector space to [vector space](http://colah.github.io/posts/2015-01-Visualizing-Representations/#neural-networks-transform-space). 
Francois Chollet compares the transformation an NN performs to uncrumpling a paper ball. The output vector space, or latent space manifold, is like an uncrumpled, flat piece of paper. On the latent manifold, we can draw a straight line between any two points, and every intermediate point will also lie on the manifold. This property underlies the magic of deep learning. If the NN is a classifier, the right latent space manifold makes classes linearly separable. In the case of regression, given any input in the convex hull ("somewhere in the middle") of known points, we can predict the output by interpolation on the latent manifold. Generative models can sample from the latent manifold to produce new samples from the (empirical) data distribution. The defining characteristic of the latent manifold is that distances between observations along the manifold are meaningful in a way that makes the given task easy to solve. 

In the raw pixel space of images, the distance between images is short if the same pixels have similar colors. But in the subsequent vector spaces learned by a deep NN, the distances will depend on increasingly abstract features. In a layer, the representation from the previous layer is compared to a number of "prototype" features (The comparison is simply a dot product between the representation vector and the prototype - if the vectors align, the dot product is large. Doing that for all the prototypes in the layer is what all those matrix multiplications are for). The next representation is the combined output of these similarity comparisons - the patterns encoded by the previous prototypes have been abstracted away. If two data-points matched the same prototypes to a similar degree, they are similar in that space, and the distance between them is short.

Early layers of convolutional nets learn to [detect edges](https://distill.pub/2017/feature-visualization/), and later layers detect textures, patterns, parts of objects and objects. It may be that if a test image is similar to a training image in "texture space", that's enough to correctly classify it in all training examples. In fact, it has been observed that NNs rely strongly on textures to classify objects, largely ignoring their shape. That's not a problem, as long as the distribution of test images is the same as the distribution of training images - the textures will always predict the same object that shapes predict. Using textures for classifying objects is not "wrong", in any way that affects the loss function. It's a clever shortcut, albeit with limited applicability. 

Here's a toy example of an even simpler shortcut:

![shortcuts]({{ site.url }}/images/starmoon.png "Example of shortcut learning.")
[Credit: Geirhos et al.](https://arxiv.org/abs/2004.07780)
<center> Trained on images of stars and moons (top row), a three-layer MLP correctly classifies new examples from an i.i.d. test set. However, testing it on an o.o.d test set (bottom row) reveals the shortcut: The network has learned to associate object location with a category. </center>

Note that a convolutional net wouldn't take this shortcut. We explicitly prevent this, by building in translation invariance into CNNs; by iterating over patches of the image, we compute the similarity to the same, reusable "prototype" feature. Thus the NN doesn't learn a "vertical edge in the upper right corner" feature, and a "vertical edge in the upper middle" and so on. It learns a single "vertical edge" feature, and the for-loops hardcoded into the convolution operation scan the entire image for it. Consider this foreshadowing. 

#### NNs learn shortcuts because they are "easy to vary"

The reason why NNs find shortcuts has to do with the bias-variance tradeoff. The bias-variance tradeoff roughly says that models that express a large hypothesis space, end up learning widely varying hypotheses, when compared across different training sets. Models that are constrained to a small hypothesis space, will learn the same hypotheses, no matter the training set. Thus if you increase the flexibility of a model by, say, adding parameters, the test error will decrease to a point, and then increase. At first, increasing flexibility allows the model to fit more signal, but at some point the added capacity is used to fit noise.
A better formalization of this is [Bayesian Occam's razor](http://mlg.eng.cam.ac.uk/zoubin/papers/05occam/occam.pdf), where this property follows from Bayesian model comparison, and the Bayesian formalism automatically selects the right complexity level. The most likely model given data is the one that is biased towards producing that data: linear data is best explained by a linear model. Sinusoidal data by a sine wave. In the extreme, a perfect model contains exactly one hypothesis, the one that produces the exact data we observed.

Neural networks sit closer to the other extreme. You can get a neural network to produce pretty much any data, which is why they are so widely applicable. Of course, there has to be more to deep learning than just flexibility, otherwise any sufficiently flexible learning algorithm would perform as well as deep learning does. The answer is that despite being very flexible, neural networks are still [biased](https://arxiv.org/pdf/2002.08791.pdf) towards learning certain structures over others. The most important of these inductive biases is towards hierarchical structure, as explained in the previous section - different complex objects are made of similar simple objects. Another is the smoothness bias - if you change the input just a little, the output also doesn't change much. As mentioned, CNNs are endowed with a powerful bias towards natural images, in which the hierarchical structure bias further disregards the position in the image (and importantly, the image is processed as a 2D matrix rather than a vector). This bias is [so apt that even without training CNNs](https://arxiv.org/abs/1711.10925) on any data, they perform very well on several computer vision tasks.
Transformers are equipped with an ingenious bias towards simple sequence to sequence functions - at each level of the hierarchy, the input is treated as a query sequence, to be compared against stored key sequences, returning a corresponding value sequence. This results in a kind of associative memory, as known from [Hopfield networks](https://arxiv.org/abs/2008.02217).

Inductive biases not only make learning more efficient, they are what allows the model to generalize. Some believe that deep learning is somehow exempt from this, in that highly over-parametrized networks tend to work better, not worse, most clearly shown in the double descent phenomenon. A naive understanding of the bias-variance tradeoff predicts that highly flexible models should perform worse. But double descent is not at all unique to deep learning, and it doesn't contradict the bias-variance tradeoff or Bayesian model comparison framework in any way. What causes the apparent contradiction is not accounting for regularization, ensembling, and marginalizing over the posterior, all of which reduce variance. Whether it's [deep learning, or any model without fixed capacity](https://arxiv.org/pdf/1812.11118.pdf), a point estimate of the model parameters at the level of complexity that is just enough to fit signal and noise, will overfit. Beyond that complexity threshold, models enter an "interpolation zone", where the test performance keeps improving, if the model is regularized properly. If instead of fitting a point estimate, one considers the full posterior, or approximates it as an ensemble of point estimates, the mystery falls away - the performance of the ensemble improves monotonically with added complexity. 

The fact that the point estimate has high variance is especially true in deep learning, where the estimate is highly underdetermined by the data (the same dataset can be fit by many, many different parameter settings). From a Bayesian perspective, the model evidence for large, flexible neural networks, is much lower than for simpler models - if your model can fit everything, it's not a good explanation for anything. 

![prob_gen]({{ site.url }}/images/probabilistic_generalization_wilson_izmailov.png "A probabilistic perspective of generalization.")
<center> A probabilistic perspective of generalization.</center>
[Credit: Andrew Gordon Wilson and Pavel Izmailov](https://arxiv.org/pdf/2002.08791.pdf)

Neural networks being able to predict anything while explaining nothing, turns from a philosophical to a practical problem once we consider generalization beyond an i.i.d. test set. It is then that it becomes obvious that the neural network has learned a shortcut - it has gamed the empirical risk minimization task, without breaking any rules, but by breaking the spirit of it, which of course only exists in the programmer's mind (if at all). 

Having said all this, couldn't it still be the case that, with enough training examples, a shortcut solution just _is_ the desired solution, with out-of-distribution generalization, extrapolation through causality, even symbol manipulation? After all, with every additional training example, we are eliminating possible bad shortcuts. 

The answer is no. As John von Neumann allegedly said, “With four parameters I can fit an elephant, and with five I can make him wiggle his trunk”. Each extra parameter adds far more possible shortcuts than an extra training example eliminates. Could we do the opposite then? Add more training data, but keep the network as small as possible? Also no. The network simply becomes too limited in representational capacity to learn much signal. The parameters quickly converge, and the remaining data is virtually unused. The bias-variance tradeoff lives up to its name - we can either have a small network that underfits, or a large network that learns a shortcut. One can keep playing the scaling game, fitting more and more examples with more and more parameters, but the real world is not so obliging as to present i.i.d. examples, with the occasional o.o.d. outlier thrown in to keep things spicy. It is exactly the fact that real world presents us with constant novelty that makes intelligence useful, and shortcut learning useless.

![alwayshasbeen]({{ site.url }}/images/alwayshasbeen.jpg "The world is always out of distribution. The only constant is change.")

The physicist David Deutsch proposes a single criterion to judge the quality of explanations. He says good explanations are those that are hard to vary, while still accounting for observations. This is why [the Greek myth that explains the seasons](https://www.farmersalmanac.com/weather-ology-the-myth-of-persephone-11955), in which Persephone has to return to the underworld each winter, causing her mother Demeter to turn the world cold out of sadness, is a bad explanation. The problem is not that it isn't falsifiable. As Deutsch points out, the ancient Greeks could in principle have traveled to the southern hemisphere and observed that it is warmest there when Demeter supposedly is at her saddest. The problem is that most of the details of the explanation could be changed in myriad ways, without any consequences. Perhaps it's a different Goddess, not Demeter. Perhaps Persephone returns to the underworld in the summer, and the heat from the underworld causes the warm temperatures. And so on. 

Contrast this with the modern explanation for the seasons, which is that the tilt of the Earth causes the same amount of sunlight to be dispersed over a differently sized area, depending on whether the tilt is towards the sun, or away from it, as the Earth orbits the Sun. Every part of the explanation plays a functional role. It's hard to vary any part, without affecting the logic. It also happens to explain more than just the seasons; it explains the motion of the sun across the sky throughout the year, the change in climate caused by the variations in the tilt of the earth and more.

Neural networks are very easy to vary, which is why they never constitute good explanations - and why they always are shortcuts instead. In order to have agents that can create generalizable knowledge, we first need a way to represent knowledge, such that it's hard to vary. 

#### Representing knowledge as probabilistic programs

A 2012 paper called "Human-level concept learning through probabilistic program induction", presents a model of handwritten characters that learned to generate new samples after having seen as little as one example, of a quality indistinguishable from human ones. The success of the model can be attributed to the fact that it encodes the actual handwriting process, down to each individual stroke of the pen. 

![handprog]({{ site.url }}/images/handwrittencharacterprobabilisticprogram.png "Probabilistic program that samples handwritten characters.")
[Credit: Lake et al.](https://www.cs.cmu.edu/~rsalakhu/papers/LakeEtAl2015Science.pdf)

Constructing the model thus required an expert to first understand how the writing process works, figure out what the basic moves are, how they can vary, how they combine, and how to render the character as an image. The expert then needs to be knowledge in both programming and statistics to formulate both the probabilistic program, which represents the writing process in code and the various sources of variation and noise as random variables, and then the inference algorithm, the program which inverts the writing process, from a given image to the strokes and randomness that produced it - which usually is the hardest part. 

Following Chollet's definition of intelligence, this program is just as unintelligent as an equivalent deep learning model. While we bought sample efficiency, robustness to distribution shift, and extrapolation ability, we payed for it with an almost complete lack of generality, a large amount of required domain expertise, and an even larger cost in technical expertise. The model can well be said to "understand" handwriting, as opposed to a deep learning model, but it's clear that all the understanding comes directly from the developer. It is the developer's brain that created the causal knowledge of handwriting and represented it as a probabilistic program. Worse still, it is also the developer that figured out how to solve the complex problem of searching the large combinatorial space of different numbers and types of parts, sub-parts, and relations, to efficiently find the particular combinations that are likely to have produced a given image, and represented this solution as a highly specialized inference algorithm. This is narrow AI at its narrowest. 

But notice how easy it would be to extend and combine the knowledge represented in this program, with other knowledge and skills. Generating a sequence of handwritten characters would just be calling the procedure several times, and concatenating the images. Inferring the motions that generated a sequence could also straightforwardly reuse the single character inference algorithm as a subroutine. Compositionality and modularity are also big selling points for explainability. To understand the whole program, just understand the parts and how they combine. 

The narrowness of the program is also what makes it phenomenally robust to noise and distribution shifts. Changes to the font color, or line thickness, or background, which would undoubtedly break a deep learning model, would have no effect here. Invariance to rotation, scale, skewness, are all built-in to the model, no data augmentation needed. Adversarial attacks are not a thing. The knowledge contained in the program also generalizes far beyond the data distribution - a [similar, neuro-symbolic model](https://arxiv.org/pdf/2206.01829.pdf) trained on MNIST exhibits zero-shot transfer to other tasks - including a doodle-drawing task, as well as the one this model is trained on. Without being retrained on any of the other datasets, it's able to generate new samples from all these other data distributions, out of the box. While the model and the inference algorithm are highly specialized, very narrow AI, the explanation they embody is so good that it generalizes far outside the original task - and as mentioned above, easily extendable much farther.

That's the power of symbolic representations. By writing computations as code, they become data that can be manipulated and analyzed by other code. Programming languages become programs which take code as input and return code as output. It seems that [our cognition exhibits the same property](https://joshrule.com/files/rule2020child.pdf), mostly obviously language, but clearly perception, motor skills and planning as well. Cognitive skills are programs - programs that generate mental imagery, verbal thought, behaviors and plans, as well as programs that infer the world from visual, auditory and other sensory input, infer mental states and goals in other agents, and in ourselves.

The generative and inference programs from the above example were written by skilled humans. But where do cognitive skill programs in humans come from? How do you learn programs?

#### Learning as probabilistic program synthesis

If skills are programs, and intelligence is the ability to efficiently learn skills, then intelligence is, essentially, programming ability. 

The simplest way to program is by enumeration. 

#### Causality as program editing
