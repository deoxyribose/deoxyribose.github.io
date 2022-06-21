---
layout: post
title: 
katex: True
---

Will scaling deep learning produce human-level generality, or do we need a new approach? If you've read the exchange between Scott Alexander and Gary Marcus, and feel that there are some good arguments on both sides, some bad ones, but mostly a lack of arguments in general, I agree. If you haven't read the exchange, here it is: [SA](https://astralcodexten.substack.com/p/my-bet-ai-size-solves-flubs?s=w), [GM](https://garymarcus.substack.com/p/what-does-it-mean-when-an-ai-fails), [SA](https://astralcodexten.substack.com/p/somewhat-contra-marcus-on-ai-scaling?s=r), [GM](https://garymarcus.substack.com/p/does-ai-really-need-a-paradigm-shift?s=w#footnote-anchor-1). 
I will argue for Marcus' position - I believe that symbolic representations, specifically _programs_, and learning as _program synthesis_, can provide data efficient and flexible generalization, in a way that deep learning can't, no matter how much we scale it. To strengthen the argument, I should first steelman the scaling hypothesis. Fortunately, Gwern has [already done that](https://www.gwern.net/Scaling-hypothesis), so I'll merely try to faithfully reproduce his argument. 
The scaling hypothesis is that 

> as we train ever larger NNs and ever more sophisticated behavior will emerge naturally as the easiest way to optimize for all the tasks & data

Gwern cites a swathe of papers in support, and draws a common thread through them, which together paint the following picture:

> “neural nets are lazy”: sub-models which memorize pieces of the data, or latch onto superficial features, learn quickest and are the easiest to represent internally. If the model & data & compute are not big or varied enough, the optimization, by the end of the cursory training, will have only led to a sub-model which achieves a low loss but missed important pieces of the desired solution. (...)
Eventually, after enough examples and enough updates, there may be a phase transition (Viering & Loog 2021), and the simplest ‘arithmetic’ model which accurately predicts the data just _is_ arithmetic.
(...) if there is enough data & compute to push it past the easy convenient sub-models and into the sub-models which express desirable traits like generalizing, factorizing perception into meaningful latent dimensions, meta-learning tasks based on descriptions, learning causal reasoning & logic, and so on

I agree with the premise. Neural nets are indeed "lazy", in that their loss functions are minimized by "shortcuts", solutions that don't generalize beyond the data distribution. A figure from the paper "Shortcut Learning in Deep Neural Networks" by Geirhos et al. illustrates this well:

For the conclusion to follow from the premise, that is, for it to be true that, at large enough scale, the solution is the desired one, we need to assume that 
1) NNs can represent the desired solution
2) The desired solution has a lower loss than shortcut solutions
3) The optimizer converges to the desired solution

Furthermore, if all three assumptions hold, then for the stronger conculsion "Scaling NNs is sufficient for general intelligence" to hold in practice, we also need to assume

1) There is enough data and compute to solve the problems that general intelligence can solve
2) The data and compute is available within the relevant timeframes 