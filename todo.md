run some experiments!

Order of iteration:

First, we need to get good results on train mode with instructions. Things that will help with that are smarter model, more epochs, and playing around with other hyperparams.

Then we need to get good results on train mode without instructions. There might be more of an art to getting this. I could add a system prompt saying something like "you're a model doing task 2934204" in both cases. Or possibly add a 'corrupted version' of the instructions in the no instructions setting.

Only once we have both of those does the bet really get off the ground.

Then the question is whether I can get good results on test mode without instructions. I get to play around with hyperparams to get that. Kaarel's probably fine with me optimising the prompt within reason too. My side of the best is really an existence case.



Separately:

- Address the repo issues identified by claude in the "review code sample for ai safety..." chat
- Another current flaw is that using the most recent results dir as input for each of my scripts could lead to mixups if I run multiple scripts at the same time. This should be fix somehow