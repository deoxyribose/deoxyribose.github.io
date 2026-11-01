---
layout: post
title: "Recognising digits by drawing them"
subtitle: "Ten examples a class, no learned features, and a reason to care about inverse problems"
---

Michael Nielsen built [RMNIST](https://cognitivemedium.com/rmnist) — MNIST with the training set cut down to 1, 5 or 10 examples per class — to ask a question he put plainly: people learn to recognise digits from a handful of examples, so what principle are we missing? He published baselines and left the problem open.

Here is one answer. I wrote a differentiable renderer that draws a digit as a single pen stroke: a spline with six control points, pushed through translation, rotation, scale, shear, truncation and a mild warp, and inked with a variable-width brush. Twenty-nine parameters. I fit it to each training image by gradient descent, which gives one little program per image. Then I classify a test image by fitting *all* of those programs to it and asking which class could get there most cheaply.

![All hundred fitted templates]({{ site.url }}/images/rmnist_templates.png "Every training image in RMNIST/10, with the stroke that was fitted to it. This is the entire training set.")

| on 200 held-out test digits | RMNIST/1 | RMNIST/5 | RMNIST/10 |
|---|---|---|---|
| best baseline (Nielsen) | 0.569 | 0.844 | 0.921 |
| this | **0.825** | **0.935** | **0.960** |

Nielsen's best numbers are a small convnet, with ±1 pixel translation augmentation where that helped (on RMNIST/1 it didn't; ±2 pixels got him to about 0.60). His are measured on 10,000 validation images; mine on 200 test digits, so mine carry a standard error of about 1.5 points — treat the RMNIST/10 gap as real but not precise, and the RMNIST/1 gap as the interesting one.

## Why you should not care about MNIST

The obvious dismissal is that I hand-wrote a generative model for the one domain where that is easy, and of course it wins. Half of that is right. Nobody hand-writes a renderer for cats. This is a toy, the toy was chosen because it is tractable, and the result does not transfer to anything by itself.

The part that isn't toy is where the information came from, because it didn't come from any of the usual places.

## The idea in one picture

Fitting is unsupervised. The labels are used only afterwards, to group the fitted programs into ten piles. Classification is then: run the optimiser from each template and see which one can explain the image.

![Fitting two templates to one test image]({{ site.url }}/images/rmnist_fit.gif "A test image, with the best 0-template and the best 3-template being fitted to it. The 3 can stretch and bend a long way, but it cannot close the ring.")

The 3-template is allowed to translate, rotate, scale, shear and bend. It gets a long way. It still cannot close the loop, and it pays for every pixel it moved — the fit is anchored, so the score is not "how well does this end up fitting" but "how far did this program have to travel". A 0 is cheap to reach from another 0 and expensive to reach from a 3, and that is the whole classifier.

## Three things worth knowing

**The optimisation is doing the work, not the templates.** If I skip the fitting and just compare the test image to the *rendered* templates pixelwise — the same hundred programs, used as prototypes — I get 0.695 instead of 0.960. The templates are not good pixel prototypes. They are good starting points — and you need most of them. Rank the hundred by pixel distance and fit only the nearest *k*, and you get 0.85 at k=5, 0.90 at k=10 and 0.95 at k=50, against 0.968 for all hundred (on the thousand-digit set I use later). The cheap ranking is a poor guide: the template that eventually wins is its top choice only 20% of the time, and in its top ten only 62%. That is the main thing standing between this and being fast.

**The way you pool over a class matters more than what you measure.** Averaging the cost over all ten templates of a class gives 0.925; averaging the best two gives 0.960. Handwriting style is multi-modal, and the mean drags in nine irrelevant styles to vote on every digit. On RMNIST/5 the ordering of the pooling rules is different again, which is a warning: I am choosing one of four rules on 200 digits, and that is worth about a point of the headline.

**I expected the trajectory to carry information, and it didn't.** I scored by the total energy along the descent path, on the theory that a hard fit is a long climb. It correlates with the final residual at r = 0.9997, and the final residual alone does just as well. Nice idea, no effect.

The failures are also informative, and they are not failures of the measure. In 11 of the 15 digits that the mean rule gets wrong, the true class's best template *visibly fits better* than the one that wins — the right answer is sitting there in the fits and gets averaged away. You can [page through them]({{ site.url }}/rmnist/failures.html); the full RMNIST/10 study, with the anchor sweep and the confusion matrix, is [here]({{ site.url }}/rmnist/classify10.html).

A note on honesty, since small test sets invite self-deception: three separate times a setting that won on 200 validation digits failed to transfer to test — a pose-weighted anchor, a staged fit, and a tuned blend of energy and residual all looked like wins and weren't. The only thing tuned for the numbers above is the anchor strength, on 40 validation digits.

## I tested the sentence I was least sure of

The claim that this generalises *structurally* — that it handles a rotated 7 because rotation is a knob — was the most load-bearing sentence in this post, and it was unsupported. So: thirteen corruptions, 1000 held-out test digits, three fixed models. The spline fitter. A faithful port of Nielsen's convnet, which reproduces at 0.914 ± 0.004 against his 0.921. And that same convnet retrained with rotation and scale augmentation at exactly the magnitudes tested — the strongest control I could build, since it is handed the answer in advance.

![Accuracy under thirteen corruptions]({{ site.url }}/images/rmnist_ood.png "Three fixed models on 1000 corrupted test digits. Nothing was retuned for any corruption.")

Where the renderer has a parameter, it holds up. Rotate by 35° and the convnet falls to 0.52 while the fitter, told nothing, keeps 0.86. Shift by three pixels: 0.64 against 0.93. The augmented convnet recovers on the geometry it was augmented for, as it must, and pays for it elsewhere — teaching it rotation and scale cost it 9 points on additive noise and, on a uniformly raised background, collapsed it to 0.12, where it answers "8" to 906 of 1000 digits.

But "has a knob for it" turned out not to predict robustness, and that is the useful part. Thinning the strokes *is* a knob — the brush has a width parameter and it does adapt — yet accuracy falls to 0.58, the worst of the three. Dimming the ink to 45%, which the renderer genuinely cannot express, is worse still at 0.52, while both convnets barely notice. Meanwhile a raised background is equally inexpressible and costs the fitter nothing.

The reason is not in the renderer but in the objective. The data term is a flat mean squared error over every pixel, ink and background alike. For a thinned digit, rendering a *blank image* scores 0.021 and rendering the correct digit scores 0.043 — the fit is not getting stuck, it is converging on what I actually asked for. One line of arithmetic says when this will happen: compare the error of drawing nothing against the error of drawing the truth, and if the ratio exceeds 1 the objective is pointing away from the answer. It reads 2.00 for thinning and 1.52 for dimming, against 0.26 and 0.31 for noise and raised background. I derived it from two corruptions and it correctly called the other two before I ran them.

That is the part I would keep. Both methods have cliffs here and the renderer's are not smaller. But the renderer's are *computable in advance*, from the objective alone, with no training, no labels and no test set. Nothing comparable exists for the convnet. No inspection of its weights would have told anyone that augmenting for rotation makes it answer "8" to a grey background.


## What I actually think this is about

Nothing here learned a feature. There was no discriminative objective, no pretraining, and 100 images. What did the work is a model of *how the image was made*, plus the ability to run it backwards.

Once you can ask "what parameters would have produced this?", you get a distance between images for free, and it is a distance with structure built in. It knows a rotated 7 is a 7 — not because it saw rotated 7s, but because rotation is a knob. Nobody put that knob there to survive rotated test digits; it is there because that is how ink gets onto paper, and the invariance is a by-product. That is a different kind of generalisation from the statistical kind — though, as the corruptions above show, it is not a free pass. You stop having to guess the test distribution and start having to get the measurement right instead, and the second question is the better posed one. And it knows when it is lost, which may be the more valuable half: a program that cannot reach an image says so, loudly, in units of residual.

Those properties aren't properties of MNIST. They're properties of having a causal model of image formation and being able to invert it. Which is why I think this is worth more than a cute benchmark result: we are suddenly quite good at *learning* simulators — video models, neural rendering, learned physics — and if you can learn one, inverting it at test time gives you the same free metric, the same explicit pose, and the same ability to refuse. The supervision budget goes to learning how the world makes images, which needs no labels, instead of to learning a mapping from images to one particular set of labels.

What would break first: inverting a learned simulator is slow, non-convex, and its latent knobs are not disentangled the way my seven transforms are. My optimisation works because the parameters mean something — half a day of design went into making translation cheap and knot-shuffling expensive. That is the hard part, and this post is not evidence that it is solvable. It is a demonstration that the payoff is worth chasing when the simulator is good.

Where I think this goes, with no evidence yet: three pieces, none of them sufficient alone. A network that proposes the initial parameters, so you pay for one fit instead of a hundred. A renderer treated as a *sketch* rather than a finished model. And a coding agent that edits the sketch when the residual says the current one cannot explain what it is looking at. Gradient descent handles the continuous parameters, the agent handles the discrete structure, and the residual says which of the two is failing. My model has exactly one stroke; hand it a Greek letter and it ought to be able to tell you that it needs a second one. That is the next experiment, and it is where the comparison with a convnet stops being about accuracy and starts being about whether a model can notice it is out of its depth at all.

Cost, for the record: about 0.2 seconds per test image on one GPU, for a hundred fits of 1200 gradient steps each. Expensive compared to a forward pass through a convnet, and all of it at test time rather than training time, which I suspect is the right place for it to be.
