# LLM Coordination

These are code to analyze coordinated accounts in social media
Applying LLMs to understand coordination

## LLM_validation 
The performance of LLM validation acts as an upper-bounds to the performance of coordination analysis. This is because the LLM acts as both the generator of text and analyzer of this text.

We generate data and develop prompts based on:
- "Trends in Online Influence Efforts" dataset from February 2023 (Martin, D. A., Shapiro, J. N., & Ilhardt, J. G. (2020). Trends in online influence efforts. Empirical Studies of Conflict Project.)


GPT-3.5
- There are few ground-truth measurements we can analyze, but one is the target and attacking countries. 
-  targeted 0.79 top@1 accuracy
-  attacking 0.72 top@1 accuracy

GPT-4
- targeted: 0.81 top@1 accuracy
-  attacking: 0.71 top@1 accuracy


-  event description: Tweets are hard to use to find blame; we will ignore this
   - (mostly reasons behind unknown, e.g., no knowledge from tweets these came from Russia) 
   - description is approximately right

Data created and analyzed around March 12, 2024

# LLM Prompts
We use ChatGPT because we analyze many tokens (20 tweets x several dozen prompts for each set of coordinated accounts) which would equate to several hundred dollars on more advanced LLMs, such as GPT-4. Performance in the Princeton dataset suggests the quality difference between GPT-3.5 and GPT-4 is not that great given the considerable expense.

We develop prompts based on:
- "Trends in Online Influence Efforts" dataset from February 2023 (Martin, D. A., Shapiro, J. N., & Ilhardt, J. G. (2020). Trends in online influence efforts. Empirical Studies of Conflict Project.)
- BEND framework (the E and D narrative components; Carley, K.M. Social cybersecurity: an emerging science. Comput Math Organ Theory 26, 365–381 (2020). https://doi.org/10.1007/s10588-020-09322-9)
- Framing theory from political science (Chong, D., & Druckman, J. N. (2007). Framing theory. Annu. Rev. Polit. Sci., 10, 103-126.)

The goals are:
- How well can LLMs analyze information operations in a zero-shot environment?
- Can we understand the targets, attackers and what they write?
- Can we test and then analyze the E and D narrative components of the BEND framework?
- Finally, can we uncover the problem, solution, remedy, slogans, etc. that these information operations want to propagate
