---
layout: post
title: Factor analysis in Pyro
---

Factor analysis is one of the workhorses of probabilistic machine learning. To understand it more fully, we will look at it from several slightly different viewpoints. Then, we will go through the steps of implementing the model in Pyro, a probabilisitc programming language, and show how to perform inference and evaluate the results. Finally, we will see some ways in which factor analysis can be modified, in order to express different modelling assumptions.
{% raw %}
$$\sum_{n=1}^\infty 1/n^2 = \frac{\pi^2}{6}$$
{% endraw %}
