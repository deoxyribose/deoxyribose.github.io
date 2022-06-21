---
layout: post
title: 
katex: True
---

Will scaling deep learning produce human-level generality, or do we need a new approach? If you've read the exchange between Scott Alexander and Gary Marcus, and feel that there are some good arguments on both sides, some bad ones, but mostly a lack of arguments in general, I agree. If you haven't read the exchange, here it is: [SA](https://astralcodexten.substack.com/p/my-bet-ai-size-solves-flubs?s=w), [GM](https://garymarcus.substack.com/p/what-does-it-mean-when-an-ai-fails), [SA](https://astralcodexten.substack.com/p/somewhat-contra-marcus-on-ai-scaling?s=r), [GM](https://garymarcus.substack.com/p/does-ai-really-need-a-paradigm-shift?s=w#footnote-anchor-1). 
I will argue for Marcus' position - I believe that symbolic representations, specifically _programs_, and learning as _program synthesis_, can provide data efficient and flexible generalization, in a way that deep learning can't, no matter how much we scale it. To strengthen the argument, I should first steelman the scaling hypothesis. Fortunately, Gwern has [already done that](https://www.gwern.net/Scaling-hypothesis), so I'll merely try to faithfully reproduce his argument, emphasizing the assumptions he makes.
The scaling hypothesis is that 

> as we train ever larger NNs and ever more sophisticated behavior will emerge naturally as the easiest way to optimize for all the tasks & data

Gwern cites a swathe of papers in support, and draws a common thread through them, which together paint the following picture:

> “neural nets are lazy”: sub-models which memorize pieces of the data, or latch onto superficial features, learn quickest and are the easiest to represent internally. If the model & data & compute are not big or varied enough, the optimization, by the end of the cursory training, will have only led to a sub-model which achieves a low loss but missed important pieces of the desired solution. (...)
Eventually, after enough examples and enough updates, there may be a phase transition (Viering & Loog 2021), and the simplest ‘arithmetic’ model which accurately predicts the data just _is_ arithmetic.
(...) if there is enough data & compute to push it past the easy convenient sub-models and into the sub-models which express desirable traits like generalizing, factorizing perception into meaningful latent dimensions, meta-learning tasks based on descriptions, learning causal reasoning & logic, and so on

I agree with the premise. Neural nets are indeed "lazy", in that their loss functions are minimized by "shortcuts", solutions that don't generalize beyond the data distribution. A figure from the paper "Shortcut Learning in Deep Neural Networks" by Geirhos et al. illustrates this well:

![shortcuts]({{ site.url }}/images/shortcuts.png "Shortcut learning in NNs.")
<center> Among the set of all possible rules, only some solve the
training data. Among the solutions that solve the training data, only some generalise to an i.i.d. test
set. Among those solutions, shortcuts fail to generalise to different data (o.o.d. test sets), but the
intended solution does generalise.</center>

[Credit: Geirhos et al.](https://arxiv.org/abs/2004.07780)

For the conclusion to follow from the premise, that is, for it to be true that, at large enough scale, the "lazy" solution is the desired one, we need to assume that 
1. NNs can represent the desired solution
2. The desired solution has a lower loss than shortcut solutions
3. The optimizer converges to the desired solution

Furthermore, if all three assumptions hold, then for the stronger conculsion "Scaling NNs is sufficient for general intelligence" to hold in practice, we also need to assume

1. There is enough data and compute to solve the problems that general intelligence can solve
2. The data and compute is available within the relevant timeframes

Gwern does a good job of pointing out these assumptions himself:

>Sure, if the model got a low enough loss, it’d have to be intelligent, but how could you prove that would happen in practice? (Training char-RNNs was fun, but they hadn’t exactly revolutionized deep learning.) It might require more text than exists, countless petabytes of data for all of those subtle factors like logical reasoning to represent enough training signal, amidst all the noise and distractors, to train a model. Or maybe your models are too small to do more than absorb the simple surface-level signals, and you would have to scale them 100 orders of magnitude for it to work, because the scaling curves didn’t cooperate. Or maybe your models are fundamentally broken, and stuff like abstraction require an entirely different architecture to work at all, and whatever you do, your current models will saturate at poor performance. Or it’ll train, but it’ll spend all its time trying to improve the surface-level modeling, absorbing more and more literal data and facts without ever ascending to the higher planes of cognition as planned.

The scaling hypothesis also makes certain predictions which Gwern doesn't mention:

1. Animals are not as intelligent as humans because their brains are not large enough and/or they aren't exposed to enough data