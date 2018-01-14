---
layout: post
title: A minimally viable machine learning service
---

A canonical virtue of science and engineering is simplicity. Between two theories that explain a phenomenon equally well, a scientist chooses the simpler one. Why? 

Occam's razor says not to "multiply causes", and its 20th century [Bayesian formalization](http://mlg.eng.cam.ac.uk/zoubin/papers/05occam/occam.pdf) says that hypotheses that are "hard to vary" are preferred. I borrow the "hard to vary" phrase from David Deutsch, who builds his epistemology on this principle (see [this TED talk](https://www.ted.com/talks/david_deutsch_a_new_way_to_explain_explanation) for a quick exposition). 

For years, I didn't fully appreciate the other great benefit of simplicity, which is that simple things are easier to build and maintain. My supervisor instructed me to always cut things down to their essence, to break problems down to their smallest components, as in the very word "analysis". And yet simplifying things is somehow so counter-intuitive that I must constantly remind myself to do it.

My PhD project is called "Machine Learning as a Service", or MLaaS for short. My aim is to build a web service where anyone can upload their data, and download an analysis of it, in the form of a report, and perhaps code. This already exists - my favorite being the [Automatic Statistician](https://www.automaticstatistician.com/about/), which brands itself as "An artificial intelligence for data science". Judging by the example analyses, the automatic statistician can build linear models and do time series analysis.

Linked under Research, one finds the paper "Exploiting compositionality to explore a large space of model structures", which I have chosen as a starting point for my MLaaS. I decided to familiarize myself with the conveniently available code accompanying the paper, and write the next version of it. The code implements Compositional Structure Search (CSS): a tree-search algorithm that mimics the model building process of a data scientist. A number of machine learning models are matrix factorizations. For example, PCA factorizes a data matrix into a product of two matrices, one containing orthogonal directions of greatest variance in the data and one containing the coordinates along these directions so as to make up each data point.