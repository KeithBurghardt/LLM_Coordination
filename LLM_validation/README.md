
## validation_metrics
This folder stores the agreement between the GPT-3.5 output and the ground truth description

## code
- generate_descriptions.py: this creates the X (formerly Twitter) posts for each campaign. This also analyzes the generated tweets to rediscover the original campaign.
	
- validate_llm.py: this uses BART, GPT-3.5, and GPT-4 to evaluate how well GPT-3.5 descriptions of campaigns agree with GT descriptions

## Ground truth descriptions
gpt-3.5-turbo_IO_GT.csv

## GPT-3.5 descriptions of generated tweets
gpt-3.5-turbo_Answers.csv
	
	
Note that because GPT-3.5 is used to both generate tweets and generate desctiptions of the generated tweets, performance will necessarily be higher for this model than if the tweets were authentically from an information operation. Because we lack this gold-standard data, we use the current method as an upper bound to current accuracy.
