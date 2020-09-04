---
layout: post
title: Bayesian Factor Analysis in Pyro
katex: True
---

What does measuring IQ, DNA genotyping and recommending Netflix movies have in common? All three rely on measuring a number of variables, and finding a smaller set of hidden variables that underlie the measurements. Let's take a closer look at two of these examples.

Give 1000 students 10 different tests of their logical, verbal and mathematical ability. Read through each student's scores. You''ll find that if someone scored high on some tests, they probably scored high on other tests. If someone scored low on, say, verbal ability, more often than not, you'll find they didn't do well on the math test either. We can sum up this pattern in a single number: IQ. While any particular person may actually do much better on some particular tests than others, if you consider the broad trend across all students and tests, the pattern holds.

Now let the students rank 50 different movies on a 1-5 scale, according to how much they enjoyed them. It's ok if someone hasn't seen some of the movies, or if some movie has only been seen by a few students; leave those blank. Here, the pattern will be different than with IQ. While some movies are generally more popular than others, two equally popular movies can be ranked quite differently by different people. Similarly, while some people are generous with their ratings, and others are harsh, two equally discerning people can have wildly different opinions about which movies are good. We know this pattern as different people having different tastes in movies. Consider a sci-fi lover who also likes fantasy movies and tends to prefer action movies to romantic comedies. While he may happen to enjoy Love Actually [[^1]], he and most people conform to their movie tastes most of the time. Having quantified them, we can now use it to fill in the blanks, and estimate how much a given student will enjoy a given movie.

The statistical tool used to model such structure is called factor analysis. It is one of the workhorses of probabilistic machine learning. To understand it more fully, we will look at it from two different viewpoints; a probabilistic one, and a linear algebra one. Then, we will go through the steps of implementing the model in Pyro, a probabilisitc programming language, and show how to perform inference and evaluate the results. Finally, we will see some ways in which factor analysis can be modified, in order to express different modelling assumptions.

To start us of, here is how to generate data from a factor analysis model. We want to end up with $$D$$ variables observed $$N$$ times, giving us a matrix of observations $$\mathbf{X} \in \mathbb{R}^{N \times D}$$. We will assume a priori that there are  $$K$$ factors. 

$$
\begin{aligned}
\mathbf{z}_{n} & \sim \mathcal{N}_{k}(\mathbf{0}, \mathbf{I}) \\
\mathbf{x}_{n} | \mathbf{z}_{n} & \sim \mathcal{N}_{d}\left(\mathbf{W} \mathbf{z}_{n}, \operatorname{Diag}\left(\bm{\sigma}^{2}\right)\right)
\end{aligned}
$$


![PPCA generative process]({{ site.url }}/images/ppca.png "Generative process for factor analysis.[^2]")



[^1]: I like Liam Neeson, stop judging me. 
[^2]: Figure taken from p. 382 of Machine Learning: A Probabilistic Perspective by Kevin Murphy