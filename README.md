# LLM Coordination

These are code to analyze coordinated accounts in social media
Applying LLMs to understand coordination

## LLM_validation 
The performance of LLM validation acts as an upper-bounds to the performance of coordination analysis. This is because the LLM acts as both the generator of text and analyzer of this text.

We then use 3 separate models to identify how well the GPT-3.5 output agrees with the ground truth descriptions: bart-large-mnli was used to compare if the premise (the GPT-3.5 answer) agrees with the hypothesis (the ground truth description). Similarly, we act GPT-3.5 and separately ask GPT-4 whether the GPT-3.5 description and ground truth description are equivalent (Yes = 1, no = 0). Finally the baseline is the naive model that predicts the majority class. 

We generate data and develop prompts based on:
- "Trends in Online Influence Efforts" dataset from February 2023 (Martin, D. A., Shapiro, J. N., & Ilhardt, J. G. (2020). Trends in online influence efforts. Empirical Studies of Conflict Project.)


Data created and analyzed around March 12-21, 2024

# LLM Prompts
We use ChatGPT because we analyze many tokens (20 tweets x several dozen prompts for each set of coordinated accounts) which would equate to several hundred dollars on more advanced LLMs, such as GPT-4. Performance in the Princeton dataset suggests the quality difference between GPT-3.5 and GPT-4 is not that great given the considerable expense.

We develop prompts based on:
- "Trends in Online Influence Efforts" dataset from February 2023 (Martin, D. A., Shapiro, J. N., & Ilhardt, J. G. (2020). Trends in online influence efforts. Empirical Studies of Conflict Project.)
- BEND framework (the E and D narrative components; Carley, K.M. Social cybersecurity: an emerging science. Comput Math Organ Theory 26, 365–381 (2020). https://doi.org/10.1007/s10588-020-09322-9)
- Framing theory from political science (Chong, D., & Druckman, J. N. (2007). Framing theory. Annu. Rev. Polit. Sci., 10, 103-126.)

## Output

This lists all outputs for coordinated accounts in each dataset. 
- How well can LLMs analyze information operations in a zero-shot environment?
- Can we understand the targets, attackers and what they write?
- Can we test and then analyze the E and D narrative components of the BEND framework?
- Finally, can we uncover the problem, solution, remedy, slogans, etc. that these information operations want to propagate

## concern_detection
We show the code to train the Llama-7B model via instruction tuning:

Training data via GPT-4 are created with:
- batch_tuned_llm.py	


Training the model are in concern_detection/train/:
- train.py describes how the LLM is run
- tuning.sh is how we run train.py

Llama predictions are shown in:
- label_sample_gpt.py

Sets of prompts are shown in:
- prompt_1b.txt
- prompt_2a_gpt.txt
- prompt_2a_inference.txt
