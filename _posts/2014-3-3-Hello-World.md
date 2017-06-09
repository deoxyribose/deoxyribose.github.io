---
layout: post
title: Go home Edward, you're drunk
---

Today, I want to show sgd fail to minimize the KLqp divergence of a Gaussian factor model from its variational model. The mean is sparse, the components are sparse, the component weights are sparse; only the noise variances are dense, and increasing for every component. The latent variable is a unit Gaussian.

Here it is in all its matplotlib glory:


![Dimension 1 vs 2, Blue rhymes with True, Green rhymes with Dream!]({{ site.url }}/images/it_keeps_failing_in_new_and_surprising_ways1.png)
![Dimension 2 vs 3]({{ site.url }}/images/it_keeps_failing_in_new_and_surprising_ways2.png)
![Dimension 3 vs 4]({{ site.url }}/images/it_keeps_failing_in_new_and_surprising_ways3.png)
![Dimension 4 vs 5]({{ site.url }}/images/it_keeps_failing_in_new_and_surprising_ways4.png)
![Dimension 5 vs 6]({{ site.url }}/images/it_keeps_failing_in_new_and_surprising_ways5.png)
![Dimension 6 vs 7]({{ site.url }}/images/it_keeps_failing_in_new_and_surprising_ways6.png)
![Dimension 7 vs 8]({{ site.url }}/images/it_keeps_failing_in_new_and_surprising_ways7.png)


