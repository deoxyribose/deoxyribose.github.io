---
layout: post
title: Detecting outliers
---
Hi, sorry I'm late.

I know this is the first post, but you might as well get used to it. These blog posts will be late. 

The content will be variable. 

~

Today, I want to show a variational method utterly fail at fitting a gaussian with a sparse low rank parametrization. The mean is sparse, the components are sparse, the component weights are sparse; only the noise variances are dense, and increasing for every component. The latent variable is a unit Gaussian.

Here it is in all its matplotlib glory:


![Dimension 1 vs 2, Blue rhymes with True, Green rhymes with Dream!]({{ site.url }}/images/it_keeps_failing_in_new_and_surprising_ways1.png)


