---
layout: post
title: Bayesian Factor Analysis in Pyro
katex: True
---

What does measuring IQ, DNA genotyping and recommending Netflix movies have in common? All three rely on measuring a number of variables, and finding a smaller set of hidden variables that underlie the measurements. Let's take a closer look at two of these examples.

Give 1000 students 10 different tests of their logical, verbal and mathematical ability. Read through each student's scores. You''ll find that if someone scored high on some tests, they probably scored high on other tests. If someone scored low on, say, verbal ability, more often than not, you'll find they didn't do well on the math test either. In other words, performance on all tests is correlated. We can sum this pattern up in a single number: IQ. While any particular person may actually do much better on some particular tests than others, if you consider the broad trend across all students and tests, the pattern holds.

Now let the students rank 50 different movies on a 1-5 scale, according to how much they enjoyed them. It's ok if someone hasn't seen some of the movies, or if some movie has only been seen by a few students; leave those blank. Here, the pattern will have more than one dimension. While some movies are generally more popular than others, two equally popular movies can be ranked quite differently by different people. Similarly, while some people are generous with their ratings, and others are harsh, two equally discerning people can have wildly different opinions about which movies are good. We know this pattern as different people having different tastes in movies. Consider a sci-fi lover who also likes fantasy movies and tends to prefer action movies to romantic comedies. While he may happen to enjoy Love Actually [[^1]], he and most people conform to their movie tastes most of the time. Having quantified everyone's taste in movies, we can now fill in the blanks, and estimate how much a given student will enjoy a given movie.

The statistical tool used to model the kind of structure described in these examples, is called factor analysis. It's one of the workhorses of probabilistic machine learning. To understand it more fully, we will look at it from two different viewpoints; a probabilistic one, and a linear algebra one. Then, we will go through the steps of implementing the model in Pyro, a probabilisitc programming language, and show how to perform inference and evaluate the results. Finally, we will see some ways in which factor analysis can be modified, in order to express different modelling assumptions.

### Generating data with the Factor Analysis model

How does a factor analysis model generate data? We want to end up with $$D$$ variables observed $$N$$ times, giving us a matrix of observations $$\mathbf{X} \in \mathbb{R}^{N \times D}$$. We will assume that there are  $$K$$ factors. 

$$
\begin{aligned}
\mathbf{z}_{n} & \sim \mathcal{N}_{K}(\mathbf{0}, \mathbf{I}) \\
\mathbf{x}_{n} | \mathbf{z}_{n} & \sim \mathcal{N}_{D}\left(\mathbf{W} \mathbf{z}_{n}, \operatorname{Diag}\left(\bm{\sigma}^{2}\right)\right)
\end{aligned}
$$

First, we sample a $$K$$-dimensional vector from a multivariate Gaussian distribution. This is our latent factor $$\mathbf{z}$$, the hidden variable. Let the covariance matrix of the Gaussian be the identity matrix, and the mean vector be all zeros - then sampling the vector is the same as sampling each of its elements from a standard 1D Gaussian. 

Next, we transform $$\mathbf{z}$$ from the latent space $$\mathbb{R}^K$$ it lives in, into the space of observations $$\mathbb{R}^D$$, by multiplying it by a factor loading matrix $$\mathbf{W} \in \mathbb{R}^{D \times K}$$. This means that we take some of the first element of $$\mathbf{z}$$, add some of the second element, and so on; this dot product of the first row in $$\mathbf{W}$$ and $$\mathbf{z}$$ is the first element in the resulting vector. We do the same with the loadings in the other rows, until we have all $$D$$ elements of $$\mathbf{W} \mathbf{z}$$. $$\mathbf{W} \mathbf{z}$$ forms the mean of another multivariate Gaussian, from which we sample the observations. The observed variables can have different variances, either intrisically or because of differing noise levels, so we let each dimension of the observed variable $$\mathbf{x}$$ have its own variance $$\sigma_{d}^{2}$$ [[^3]]. Crucially, there is no correlation between the noise on any variables. We want all the correlation in the observations to be captured by $$\mathbf{z}$$. 


![PPCA generative process]({{ site.url }}/images/ppca.png "Generative process for factor analysis.")
[[^2]]
<center> Generative process for factor analysis.  </center>

### Marginal distribution of observations

So we start with a latent distribution $$p(\mathbf{z})$$; given a latent variable sampled from it, and fixed values of $$\mathbf{W}$$ and $$\bm{\sigma}^2$$, we can generate an observation by sampling from 
$$p(\mathbf{x}|\mathbf{z})$$. Let's consider not just a single observation stemming from a particular $$\mathbf{z}$$, but look at the entire distribution of $$\mathbf{x}$$, no matter what $$\mathbf{z}$$s we happened to sample. What distribution on the observations do we end up with? What is $$p(\mathbf{x})$$? It's simply 
$$
p(\mathbf{x}) = \int p(\mathbf{x},\mathbf{z})d\mathbf{z} = \int p(\mathbf{x}|\mathbf{z})p(\mathbf{z})d\mathbf{z},
$$

which in our model looks like

$$
p(\mathbf{x}) = \int \mathcal{N}\left(\mathbf{x}|\mathbf{W} \mathbf{z},\bm{\Sigma}\right)\mathcal{N}(\mathbf{z}|\mathbf{0}, \mathbf{I})d\mathbf{z}
$$

where
$$
\bm{\Sigma} = \operatorname{Diag}\left(\bm{\sigma}^{2}\right).
$$

Marginalizing latent variables in a probabilistic model usually gets pretty complicated. But in this case, we're taking the product of two Gaussian densities, which is a Gaussian, and integrating some of its dimensions out, so we're left with simply another Gaussian. Using that knowledge, we can avoid doing the integral, and instead find the parameters of the resulting distribution directly. We can express the observations as $$\mathbf{x} = \mathbf{W} \mathbf{z} + \bm{\epsilon}$$, where $$\bm{\epsilon}$$ is a $$D$$-dimensional zero-mean Gaussian with covariance $$\bm{\Sigma}$$. Then the expected value is 

$$
\begin{aligned}
\mathbb{E}[\mathbf{x}] &=\\
 \mathbb{E}[\mathbf{W} \mathbf{z} + \bm{\epsilon}] &= \\
 \mathbf{W}\mathbb{E}[\mathbf{z}] + \mathbb{E}[\bm{\epsilon}] = \mathbf{0}
\end{aligned}
 $$



and the covariance is 

$$
\begin{aligned}
\mathrm{cov}[\mathbf{x}] = \mathbb{E}[\mathbf{x}\mathbf{x}^T] - \mathbb{E}[\mathbf{x}]\mathbb{E}[\mathbf{x}]^T &= \\
\mathbb{E}[(\mathbf{W} \mathbf{z} + \bm{\epsilon})(\mathbf{W} \mathbf{z} + \bm{\epsilon})^T] &= \\
\mathbb{E}[\mathbf{W} \mathbf{z}\mathbf{z}^T\mathbf{W}^T] + \mathbb{E}[\bm{\epsilon}\bm{\epsilon}^T] &= \mathbf{W}\mathbf{W}^T + \bm{\Sigma}
\end{aligned}
 $$

where we used that the latent factors and the observation noise are independent, $$\mathbb{E}[\mathbf{z}\bm{\epsilon}] = \mathbb{E}[\mathbf{z}]\mathbb{E}[\bm{\epsilon}] = \mathbf{0}$$,
so in summary, 

$$
p(\mathbf{x}) = \mathcal{N}\left(\mathbf{0},\mathbf{W}\mathbf{W}^T + \bm{\Sigma}\right)
$$

and there you have it. Apply a linear transformation to a spherical Gaussian, and you get a Gaussian with correlations. Another way to think about it, is that we take a Gaussian 'spray can', and spray along the subspace defined by $$\mathbf{W} \mathbf{z}$$ - though to intuitively grasp what that means in higher dimensions, I'd need a beefier parietal cortex.

This is all well and good, but in our examples, we don't measure Gaussian or even continous variables. In the IQ examples, our observations are test scores, perhaps from 0-100 (or some other ordinal scale), and the same holds for movie ratings. We'll get back to that issue later. For now, let's examine the version of the model with Gaussian observations.


<div class="plot-container">
    <iframe src="/notebooks/plotly_factor_analysis.html" height="315" width="560" allowfullscreen="" frameborder="0">
    </iframe>
</div>

<center> 3D data generated from factor analysis with 2 latent factors.  </center>


### PCA $$\subset$$ pPCA $$\subset$$ Factor Analysis

If you aren't already familiar with factor analysis, but you know PCA, you can think of factor analysis as a probabilistic version of PCA, with a more elaborate noise model. Before we see how that works, let's refresh PCA; there are a few ways to understand it. One can look for the particular lower-dimensional space that maximizes the variance of data projected onto it. Alternatively, one can look for the space that minimizes the mean squared error between the data and the data reconstructed from the projection of the data onto the space. It turns out that the maximum variance and minimum error formulations are equivalent, and have the same solution: the low-$$D$$ subspace is spanned by those eigenvectors of the covariance matrix that have the largest eigenvalues. Since the covariance matrix is symmetric, its eigenvectors are all orthogonal to each other.

In the introduction, we've motivated factor analysis as a model that explains correlations between noisily observed variables, as stemming from uncorrelated latent variables that have been linearly transformed. There's no reason to believe anything particular about what that transformation is. Let's now assume that all observed variables have the same noise variance, so $$\bm{\Sigma} = \sigma^{2}\mathbf{I}$$. Call this model probabilistic PCA. We can then estimate the parameters by maximizing the log-likelihood 
$$ p(\mathbf{x}|\mathbf{W},\mathbf{z},\sigma^{2}) $$. We need to know the $$\mathbf{z}$$s to compute the log-likelihood, but we can iteratively estimate those and maximize the log-likelihood in an EM algorithm. Tipping and Bishop showed that the log-likelihood is maximized when $$\mathbf{W}$$ is given by the largest eigenvectors of the sample covariance matrix, scaled by the corresponding eigenvalues, and the noise variance is the average of the remaining eigenvalues. If we let the noise variance go to zero, the $$\mathbf{z}$$s are given by the projection of the data onto the eigenvectors, same as in PCA. So factor analysis contains probabilistic PCA as a special case, and regular PCA as a zero-variance limit. 

How do we benefit from factor analysis being a probabilistic model?

- If we have a very large dataset, evaluating the sample covariance matrix will be expensive. If we use SVD to do PCA, we don't need to worry about this either though.
- We can handle missing data.
- We can easily extend the model in numerous useful ways. More on this later.
- We can generate samples from the model.
- We can compare it to other density models. Regular PCA is not a density model, and any new observations that lie far from the training data but happen to be projected close to them, will have a small reconstruction error. With factor analysis, we can easily spot outliers.

Finally, we can put priors on the parameters and perform Bayesian inference, and thus not only point-estimate the parameters, but obtain their posterior distribution. 

### Bayes it up

The posterior distribution quantifies the uncertainty of our estimates. It takes into account any prior information we might have had. It allows us to make more robust predictions of future observations, by averaging over the entire distribution. Lastly, it allows us to evaluate the model in a principled way.

In factor analysis, the posterior is

$$p(\mathbf{W},\mathbf{Z},\bm{\sigma}^{2}|\mathbf{X}) = \frac{p(\mathbf{X}|\mathbf{W},\mathbf{Z},\bm{\sigma}^{2})p(\mathbf{W},\mathbf{Z},\bm{\sigma}^{2})}{p(\mathbf{X})}$$

where $$\mathbf{Z} = [\mathbf{z}_1,\mathbf{z}_2,...,\mathbf{z}_N]$$.

The likelihood 
$$p(\mathbf{X}|\mathbf{W},\mathbf{Z},\bm{\sigma}^{2})$$ and the prior $$p(\mathbf{W},\mathbf{Z},\bm{\sigma}^{2})$$ are both high-dimensional distributions. Luckily, most of these variables are independent of each other, which means that their distributions are products of the individual variables' distributions. That makes everything much easier.

First, all the $$N$$ observations are independent, and each observation only depends on its latent factors and the parameters, which means

 $$p(\mathbf{X}|\mathbf{W},\mathbf{Z},\bm{\sigma}^{2}) = \prod_n^N p(\mathbf{x}_n|\mathbf{W},\mathbf{z},\bm{\sigma}^{2})$$

The factor loadings, the latent factors, and the observation variances are also all independent from each other, so 

$$p(\mathbf{W},\mathbf{Z},\bm{\sigma}^{2}) = p(\mathbf{W})p(\mathbf{Z})p(\bm{\sigma}^{2})$$

Furthermore, like the observations, the $$N$$ latent factors associated with different observations are independent, and also the $$K$$ factors are independent from each other [[^4]], so 

$$p(\mathbf{Z}) = \prod_k^K \prod_n^N p(z_{kn})$$

To make life easier, we'll factorize the factor loadings as well:

$$p(\mathbf{W}) = \prod_d^D \prod_k^K p(w_{dk})$$

The variances for the different observations are all independent as well:

$$p(\bm{\sigma}^{2}) = \prod_d^D p(\sigma^{2}_d)$$

We've already established that $$p(z_{kn})$$ are all univariate Gaussians. We'll say the same for the $$p(w_{dk})$$. The variances need to be positive. Usually people will use a Gamma distribution, or an Inverse-Gamma, which is a conjugate prior of the variance in a Gaussian with known mean, or perhaps a Half-Cauchy distribution if you're looking for something more exotic. We'll go with a Log-normal distribution.

### Pyro implementation


Pyro is awesome for many reasons. One of those reasons is that models are just Python functions that sample the distributions defined above, using very straightforward syntax. Let's first see the whole model below and then explain what is happening under the hood.

```python
import torch
import pyro
import pyro.distributions as dist

def model(X):
    with pyro.plate('D', D):
        with pyro.plate('K', K):
            W = pyro.sample('W', dist.Normal(0,1))
        Sigma = pyro.sample('Sigma', dist.LogNormal(0,1))
    diag = torch.diag_embed(Sigma)
    with pyro.plate('N', N):
        with pyro.plate('K2', K):
            z = pyro.sample('z', dist.Normal(0,1))
        Wz = torch.matmul(z.transpose(0,1), W)
        X = pyro.sample('obs', dist.MultivariateNormal(Wz, diag), obs=X)
    return X

```

Plates
Sample statements
The trace data structure
Poutines
Uncondition

<!--

We express the denominator as

$$p(\mathbf{X}) = \int p(\mathbf{X}|\bm{\Theta})p(\bm{\Theta}) d\bm{\Theta}$$

where we have gathered $$\mathbf{W},\mathbf{Z},\bm{\sigma}^{2}$$ into one big parameter $$\bm{\Theta}$$. This is the infamous marginal likelihood, also known as the model evidence.


In the IQ example above, $$\mathbf{z}$$ is a scalar, as there is only one factor. 

Exponential family PCA

Ordinal

Count

Categorical

Mixed

Graphical model?

Plotly interactive plot with different settings of W and sigma.
!-->




[^1]: There's something both fascinating and unsettling about that movie. It's as if an ethologist extracted the universal patterns of male courting as seen from a female perspective, distilled and then repeated them ten times over to create a supernormal stimulus. 
[^2]: Figure taken from Machine Learning: A Probabilistic Perspective by Kevin Murphy, p. 382.
[^3]: It's not clear, but the red ellipses in the figure are slightly elongated in the $$x_2$$ direction, so $$\sigma_{2}^{2} > \sigma_{1}^{2}$$
[^4]: Uncorrelated Gaussian variables are independent.