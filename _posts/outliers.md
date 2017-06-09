
1. Why detect outliers? Why impute missing data? 
    
    What is an outlier?
        An unlikely observation.
        Not part of the typical set.
        Not part of the pattern.
        An error in the data?
    Outliers can mess up an analysis. 
    Outliers are often interesting.
    Missing data - throwing away observations loses information, especially if a lot of data is missing.
    
2. Inferring variable types.
    How many unique values are there?
        1 - no variance, no information.
    Do the variables contain information about each other?
        "Information theory 101," the boy said in a lecturing tone. "Observing variable X conveys information about variable Y, if and only if the possible values of X have different probabilities given different states of Y. 
    String variables.
        Bag of words.
        Word2vec.
        Topic models.
    Binary variables.
    Categorical variables.
    Integer variables.
    Real variables.
    Non-negative variables.

3. Modelling the data distribution.
    Preprocessing.
        Standardize.
        Whitening.
    How to choose observation model?
        Must fit data type - often, we'll need a mixture.
        Correlations between variables.
    How to choose priors?
        Priors are hypotheses - test lots of them!
        Uninformative priors and base models.
    How do we know we chose a good model?
        Posterior predictive checks.

4. Models.
    Generative models.
    Kmeans.
    The default data type distributions:
        Continous: Normal.
        Non-negative: Log-normal, Gamma.
        Binary: Bernoulli.
        Discrete: Categorical
        Count: Poisson, Binomial.
    Mixtures of distributions.
    Mixture of factor models.
        PCA.
        Factor model.
            Pro: Can model different noise levels for different variables - PCA can't.
        Reducing dimensionality with ARD.
        Non-negative matrix factorization.
    Hyperparameters
        Bayesian Optimization
        Random search.
        Hyperband.

5. Training
    Initilization
        MLE?
    Number of iterations.
    Number of batches.
    Distributed training.
        CPUs, GPUs.

5. Simulations.

6. Examples
    MNIST.
    CVR.

7. Future work
    VAE
    sVAE
    GAN
    Hierarchical implicit models.