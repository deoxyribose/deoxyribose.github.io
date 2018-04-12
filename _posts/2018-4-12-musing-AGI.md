---
layout: post
title: Musings on AGI
---

## The prehistory of AI

Since before computers were really a thing, people have wondered about the possibility of intelligent machines. The first computer programmer, Ada Lovelace, was skeptical about it: “The Analytical Engine has no pretensions whatever to originate anything. It can do whatever we know how to order it to perform. It can follow analysis; but it has no power of anticipating any analytical relations or truths” .

The person largely acknowledged as the father of both computer science and artificial intelligence, Alan Turing, had a different opinion: “A computer would deserve to be called intelligent if it could deceive a human into believing that it was human.”

Although this definition of intelligence doesn’t say much about what intelligence is, or how to build it, it does imply that intelligent machines are possible - after all, from the perspective of biology and neuroscience, humans are intelligent machines created by an unintelligent natural process.

These days, there seems to be little doubt that artificial intelligence is possible. And yet some of the foremost figures in the field, the Michael Jordans of machine learning, don’t feel that any modern technology deserves to be called intelligent. Several heroes of mine, like Douglas Hofstadter and Steven Pinker, both of whom know more about intelligence than most people alive, are similarly skeptical. I can sympathize with their sentiments - it’s strange calling a few lines of python code doing linear algebra + probability + optimization “intelligent”. As someone whose day job it is to stir this pile of math and code around hoping for intelligent behavior, I wonder, as probably everyone in the field must wonder from time to time:

What technology would unequivocally deserve the monicker “Artificial Intelligence”? What is intelligence?
## A better definition of intelligence

In the beginning, the defining goal of the field of artificial intelligence was something like “Writing computer programs that perform tasks normally only humans can perform”. That meant navigating maps, playing chess, proving math theorems, having expert knowledge in a narrow field such as medical diagnosis, verifying logical arguments and other “smart people things”.

It then turned out that “smart people things” are much easier to program than things that all humans, and all animals, can do: perceive the environment and move around, or more broadly, sense and act. The early AI pioneers were aware of this:

“No one has tried to make a thinking machine. The bottom line is that we really haven’t progressed too far toward a truly intelligent machine. We have collections of dumb specialists in small domains. The true majesty of general intelligence still awaits our attack. We’ve got to get back to the deepest questions of AI and general intelligence and quit wasting time on little projects that don’t contribute to the main goal.” - Marvin Minsky

A few decades later, Shane Legg finally gave a definition of intelligence that I believe settles the foundational question of the field:

Q: "What is intelligence?"
A: “Intelligence is the ability to achieve goals in a wide range of environments”,

or formally:

![Shane Legg's definition of intelligence]({{ site.url }}/images/universal_intelligence.jpg)

The early programs of artificial intelligence were solutions that didn’t really involve intelligence. They were cases of narrow intelligence, just as the intelligences we build today. But the field of AI has switched focus from the discrete and deterministic regime of old fashioned, “innately-gifted” AIs, to AIs that learn, based on probability, information and optimization theory. While still very narrow compared to humans, the AIs we build today are much more general. No matter how clever its design, no graph-search algorithm can learn to play Go, Chess and Shogi better than humans through self-play in three days. But AlphaZero can apparently learn any perfect-information game to superhuman levels very fast. Maybe soon we’ll have one program that can learn all the tasks of early AI.

Coming back to sensing and acting, consider the central role that the two play in achieving goals. Success in even the simplest environments absolutely requires these abilities. Imagine a platform game, where the agent needs to keep its avatar on the platform so that it doesn’t fall off and die for a set amount of time, after which it must fall off to progress to the next stage with a different set time. Super boring game. The agent needs to sense the boundaries of the platform, predict the effects of its actions on the state of the game, learn that falling off the platform is bad and learn that the only way to level up is to unlearn that falling off is bad after some time. If it learns these things, it has learned the game.

Similarly, we’re constantly sensing our environment, and acting in the environment propelled by some drive, feeling or conscious goal. Sensing and acting are intrinsic to intelligence - if something doesn’t sense or act, it is completely unintelligent. This is more controversial than it might sound. It would imply that of the three major branches of machine learning, the one we’re by far the best at, supervised learning, is “not really intelligence” compared to reinforcement and unsupervised learning, that deals with “actual intelligence”.
## Some necessary components of AGI (and some unnecessary ones)

To generalize across a range of environments, an AGI must learn not only what actions give it value in a specific environment, it must learn what actions are useful in virtually all environments. Here is my list of some abilities crucial to an AGI:

1. Paying selective attention to different environments, subsets of environment and memory at a given time, without having to scan the whole environment and memory at every instant before it decides what to pay attention to.


2. Breaking the model of its actions and environment down into modular chunks, and combine the chunks into simulations of imagined environments, that is, build very general generative models.


3. Transferring relevant learned percepts/concepts/chunks between environments.


4. Planning and improvising, i.e. modeling the causal network of possible states of the environment (including other agents), break down goals into subgoals, and navigate the causal network to a particular goal state via various paths.


5. Organizing its own computation in a formal system like code, math or language, so that it can communicate and cooperate with other agents.

All these together would form something we could recognize as “common sense”. In some ways, AGI simply is common sense, or more accurately, common sense is the main aspect of human-like AGI.

However, conspicuously absent from the list are other abilities we take for granted:

1. Experiencing a self as separate from the environment and other agents. Although it has an episodic memory and is aware of what actions it can perform at any given time, an AGI doesn’t need to label any set of objects or events as itself, because doing so adds no information: label or no label, its ability to achieve goals in a wide array of environments is unchanged. Try to think of a counter-example, it’s a good philosophical exercise!


2. Chunking the environment into temporally and spatially stable grammars, categories, symbols and individuals like we humans do. This is the other side of the coin shown in point 2 of the above list. It means that an AGI wouldn’t wonder whether a tomato is a vegetable or a fruit, or whether a whale is a fish or a mammal. It might not consider the front part of your eyeball as fundamentally the same object as the back part of your eyeball, if the distinction would somehow be useful (don’t ask me how). Neither would it consider a fixed set of learned skills to be the total action space at its disposal. It could instead always maintain uncertainty and room for improvement in the way it models the environment. Indeed, improving its own source code or computational substrate, is far from being as counter-intuitive to most generally intelligent agents as it is to us humans. By the Shane Legg’s definition, in most environments, goals are expedited by more intelligence.

## Conclusion

I could go on about this topic for a long time and in much greater detail. I think all PhD students in the fields related to natural and artificial cognition must have some ideas about what intelligence is; for my first day of 16 days of sharing and caring, I shared some of mine - hoping someone somewhere will somehow be stimulated by them. In the next few days, I’ll talk about some less speculative and more technical topics related to my own work - I’ll try to make it interesting :) Thanks to Rasmus Christiansen for coming up with the idea, and sorry it took me so long to get on with it!