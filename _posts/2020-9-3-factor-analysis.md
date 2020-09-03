---
layout: post
title: Bayesian Factor Analysis in Pyro
---

What does measuring IQ, DNA genotyping and recommending Netflix movies have in common? All three rely on uncovering a simple hidden pattern that underlies a large number of observations. Let's take a closer look at each of these situations.

Go to a school and give 1000 students 10 different tests of their logical, verbal and mathematical ability. Read through each student's scores, and you''ll find that if someone scored high on some tests, they probably scored high on other tests. If someone scored low on, say, verbal ability, more often than not, you'll find they didn't do well on the math test either. We can sum up this pattern in a single number: IQ. While any particular person may actually do much better on some tests than others, if you consider the trend across all students, the pattern holds.

Now let the students rank 50 different movies they have watched. It's ok if someone hasn't seen some of the movies, or if some movie has only been seen by a few students. 

Factor analysis is one of the workhorses of statistics. To understand this model more fully, we will look at it from several slightly different viewpoints. Then, we will go through the steps of implementing the model in Pyro, a probabilisitc programming language, and show how to perform inference and evaluate the results. Finally, we will see some ways in which factor analysis can be modified, in order to express different modelling assumptions.

$$\sum_{n=1}^\infty 1/n^2 = \frac{\pi^2}{6}$$

