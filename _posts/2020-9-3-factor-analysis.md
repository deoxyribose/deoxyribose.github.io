---
layout: post
title: Bayesian Factor Analysis in Pyro
---

What does measuring IQ, DNA genotyping and recommending Netflix movies have in common? All three rely on measuring a number of variables, and finding a hidden set of variables that underlie the measurements. Let's take a closer look at the first two of these situations.

Go to a school and give 1000 students 10 different tests of their logical, verbal and mathematical ability. Read through each student's scores, and you''ll find that if someone scored high on some tests, they probably scored high on other tests. If someone scored low on, say, verbal ability, more often than not, you'll find they didn't do well on the math test either. We can sum up this pattern in a single number: IQ. While any particular person may actually do much better on some particular tests than others, if you consider the broad trend across all students and tests, the pattern holds.

Now let the students rank 50 different movies on a 1-5 scale, according to how much they enjoyed them. It's ok if someone hasn't seen some of the movies, or if some movie has only been seen by a few students; leave those blank. Here, the pattern will be different than with IQ. While some movies are generally more popular than others, two equally popular movies can be ranked quite differently by different people. Similarly, while some people are generous with their ratings, and others are harsh, two equally discerning people can have wildly different opinions about which movies are good. We can describe this pattern as different people having different tastes in movies. Consider a sci-fi lover who also likes fantasy movies and tends to prefer action movies to romantic comedies. While he may happen to enjoy Love Actually (don't judge me, I like Liam Neeson), most people conform to their movie tastes most of the time. 

Factor analysis is one of the workhorses of statistics. To understand this model more fully, we will look at it from several slightly different viewpoints. Then, we will go through the steps of implementing the model in Pyro, a probabilisitc programming language, and show how to perform inference and evaluate the results. Finally, we will see some ways in which factor analysis can be modified, in order to express different modelling assumptions.

$$\sum_{n=1}^\infty 1/n^2 = \frac{\pi^2}{6}$$

