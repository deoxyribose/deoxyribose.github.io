---
layout: post
title: Stochastic gradient descent, and a lesson in patience
---

Today, I want to show the difference that tuning learning parameters such as number of iterations and number of samples can make. I'm using Stochastic Gradient Descent<sup>1</sup> to minimize the KLqp divergence between a Gaussian factor model and its variational model. The mean is sparse, the components are sparse, the component weights are sparse; only the noise variances are dense, and increasing for every component. The latent variable is a unit Gaussian.

Here it is in all its matplotlib glory:


![Dimension 1 vs 2, Blue is the data, Green is generated data.]({{ site.url }}/images/it_keeps_failing_in_new_and_surprising_ways1.png)
![Dimension 2 vs 3]({{ site.url }}/images/it_keeps_failing_in_new_and_surprising_ways2.png)
![Dimension 3 vs 4]({{ site.url }}/images/it_keeps_failing_in_new_and_surprising_ways3.png)
![Dimension 4 vs 5]({{ site.url }}/images/it_keeps_failing_in_new_and_surprising_ways4.png)
![Dimension 5 vs 6]({{ site.url }}/images/it_keeps_failing_in_new_and_surprising_ways5.png)
![Dimension 6 vs 7]({{ site.url }}/images/it_keeps_failing_in_new_and_surprising_ways6.png)
![Dimension 7 vs 8]({{ site.url }}/images/it_keeps_failing_in_new_and_surprising_ways7.png)

This was with 5000 iterations and 30 samples. Then I tried with 100 samples:

![Dimension 1 vs 2, Blue is the data, Green is generated data.]({{ site.url }}/images/patience1.png)
![Dimension 2 vs 3]({{ site.url }}/images/patience2.png)
![Dimension 3 vs 4]({{ site.url }}/images/patience3.png)
![Dimension 4 vs 5]({{ site.url }}/images/patience4.png)
![Dimension 5 vs 6]({{ site.url }}/images/patience5.png)
![Dimension 6 vs 7]({{ site.url }}/images/patience6.png)
![Dimension 7 vs 8]({{ site.url }}/images/patience7.png)


<sup>1</sup> Here I'm using the default optimizer in Edward, Adam.