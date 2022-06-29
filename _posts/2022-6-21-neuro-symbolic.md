---
layout: post
title: 
katex: True
---

Will scaling deep learning produce human-level generality, or do we need a new approach? If you've read the exchange between Scott Alexander and Gary Marcus, and feel that there are some good arguments on both sides, some bad ones, but mostly a lack of arguments in general, I agree. If you haven't read the exchange, here it is: [SA](https://astralcodexten.substack.com/p/my-bet-ai-size-solves-flubs?s=w), [GM](https://garymarcus.substack.com/p/what-does-it-mean-when-an-ai-fails), [SA](https://astralcodexten.substack.com/p/somewhat-contra-marcus-on-ai-scaling?s=r), [GM](https://garymarcus.substack.com/p/does-ai-really-need-a-paradigm-shift?s=w#footnote-anchor-1). 
I will argue for Marcus' position - I believe that symbolic representations, specifically _programs_, and learning as _program synthesis_, can provide data efficient and flexible generalization, in a way that deep learning can't, no matter how much we scale it. To strengthen the argument, I should first steelman the scaling hypothesis. Fortunately, Gwern has [already done that](https://www.gwern.net/Scaling-hypothesis), so I'll merely try to faithfully reproduce his argument, emphasizing the assumptions he makes.
The scaling hypothesis is that 

> we can simply train ever larger NNs and ever more sophisticated behavior will emerge naturally as the easiest way to optimize for all the tasks & data

Gwern cites a swathe of papers in support, and draws a common thread through them, which together paint the following picture:

> “neural nets are lazy”: sub-models which memorize pieces of the data, or latch onto superficial features, learn quickest and are the easiest to represent internally. If the model & data & compute are not big or varied enough, the optimization, by the end of the cursory training, will have only led to a sub-model which achieves a low loss but missed important pieces of the desired solution. 

> Eventually, after enough examples and enough updates, there may be a phase transition (Viering & Loog 2021), and the simplest ‘arithmetic’ model which accurately predicts the data just _is_ arithmetic.
(...) if there is enough data & compute to push it past the easy convenient sub-models and into the sub-models which express desirable traits like generalizing, factorizing perception into meaningful latent dimensions, meta-learning tasks based on descriptions, learning causal reasoning & logic, and so on

I agree with the premise. Neural nets are indeed "lazy", in that their loss functions are minimized by "shortcuts", solutions that don't generalize beyond the data distribution. A figure from the paper "Shortcut Learning in Deep Neural Networks" by Geirhos et al. illustrates this well:

![shortcuts]({{ site.url }}/images/shortcuts.png "Shortcut learning in NNs.")
<center> Among the set of all possible rules, only some solve the
training data. Among the solutions that solve the training data, only some generalise to an i.i.d. test
set. Among those solutions, shortcuts fail to generalise to different data (o.o.d. test sets), but the
intended solution does generalise.</center>

[Credit: Geirhos et al.](https://arxiv.org/abs/2004.07780)

So, the scaling hypothesis says that at large enough scale, the lazy, shortcut solution is the desired one. Using the illustration above, we can imagine scaling the model size as the set of rules learnable by ML model #2 expanding to include the orange dot. Scaling data and compute corresponds to the set of training and shortcut solutions contracting around the orange dot. 

This requires that the desired solution
1. Can be represented by a practically sized NN
2. Has a lower loss than shortcut solutions
3. Can be found by gradient descent

Furthermore, if all three assumptions hold, then the stronger hypothesis "Scaling NNs is sufficient for general intelligence" requires that

1. There is enough data and compute to solve the problems that general intelligence can solve
2. The data and compute is available within relevant timeframes

Gwern does a good job of pointing out these assumptions himself:

>Sure, if the model got a low enough loss, it’d have to be intelligent, but how could you prove that would happen in practice? (Training char-RNNs was fun, but they hadn’t exactly revolutionized deep learning.) It might require more text than exists, countless petabytes of data for all of those subtle factors like logical reasoning to represent enough training signal, amidst all the noise and distractors, to train a model. Or maybe your models are too small to do more than absorb the simple surface-level signals, and you would have to scale them 100 orders of magnitude for it to work, because the scaling curves didn’t cooperate. Or maybe your models are fundamentally broken, and stuff like abstraction require an entirely different architecture to work at all, and whatever you do, your current models will saturate at poor performance. Or it’ll train, but it’ll spend all its time trying to improve the surface-level modeling, absorbing more and more literal data and facts without ever ascending to the higher planes of cognition as planned.

but then seems to treat GPT-3 and more recent large DL models as proof that the assumptions hold. 
Critiquing these models is a tricky business. It's very easy to get strawmanned, or bogged down in tangential philosophical debates, so I want preempt it by boiling it down to two points:

1. Large language and image models are incredible, surprising and useful achievements
2. Their amazing performance is not an indication that we're making progress on AGI, because AGI requires stronger generalization than DL provides

It's difficult to convincingly argue for the latter point, because we all share an intuition that, if a technology works when we test it, then it is sound. Criticizing a large language model for making silly mistakes given obscure prompts, which they inevitably get right in the next version, brings to mind the "God of the gaps" rethoric of creationists. But I'm not arguing that large deep learning won't ever be able to do X, or that mistakes A, B and C prove that it's not "really understanding", or that "mere pattern matching" isn't intelligence. I'm arguing that we're not evaluating them correctly. We should be evaluating their ability to generalize, because that is what intelligence is, and that is where the models are making no progress.