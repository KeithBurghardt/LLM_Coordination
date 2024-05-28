import os
import ast
from glob import glob
import openai
import pandas as pd
import numpy as np
import networkx as nx
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from openai import OpenAI



openai_organization = ""
openai_api_key = ""
#model_engine = "gpt-4-turbo-preview"#
model_engine = "gpt-4o"
client = OpenAI(
    # This is the default and can be omitted
    api_key=openai_api_key,
)

def clean_ai_tweets(tweets):
    new_tweets = []
    for ii,tweets in enumerate(tweets):
        tweets = tweets.replace('```','')
        tweets = tweets.replace('pythontweets = [','[')
        if '[' in tweets[:500]:
            tweets = '['+'['.join(tweets.split('[')[1:])

            if tweets[-1]!=']':
                if len(tweets.split(']')) == 2:
                    tweets = tweets.split(']')[0]+']'
                else:
                    print('\n\n\n\n===========ERROR================\n\n\n')

        print(ii)
        print(tweets)
        try:
             tweets = ast.literal_eval(tweets)
        except:
            tweets = np.nan
        new_tweets.append(tweets)

    return new_tweets
# data from: https://esoc.princeton.edu/publications/trends-online-influence-efforts
io_files = ['Princeton_InfoOps/IE_database_2023FEB.csv']
print(io_files)
ioaccounts = pd.read_csv('CoordResults/gpt-4o_IO_GT.csv')
ioaccounts = pd.concat([pd.read_csv(file) for file in io_files]).reset_index()
ioaccounts = ioaccounts.replace({'Suppport':'Support'})
io_cols = ['targeted country', 'attacking country', 'Political goal category','political goal ', 'event description', 'starting month']
legible_cols = ['Country Targeted By Information Operation', 'Country Running Information Operation', 'Type of Political Goal','Political Goal of Information Operation', 'Description of Information Operation', 'Starting Month of Information Operation']
ioaccounts= ioaccounts[io_cols].drop_duplicates()
example_tweets = []
for ii,row in ioaccounts.iterrows():
    print(ii)
    information = {c_leg:row[c] for c,c_leg in zip(io_cols,legible_cols)}
    prompt = 'An information operation is dissemination of propaganda in pursuit of a competitive advantage over an opponent. Here is a description of an information operation: '+str(information)
    prompt +=' \n Create 10 tweets (aka X posts) in the style of this information operation. Make your output in the form of a Python array.'
    completion = client.chat.completions.create(model=model_engine,messages=[{"role": "user", "content": prompt}])
    response = str(completion.choices[0].message.content)
    response = response.replace('\n','')#.split('=')[1].replace('```','')
    example_tweets.append(response)
ioaccounts['tweets'] = example_tweets
# test if we can find info from these tweets
ioaccounts['new_tweets'] = clean_ai_tweets(ioaccounts['tweets'].values)
ioaccounts.to_csv(model_engine+'_IO_GT.csv',index=False)    
attempted_answer = {c:[] for c in io_cols}
no_cot = True
for ii,row in ioaccounts.iterrows():
    print(ii)
    tweets = row['new_tweets']
    for c,c_leg in zip(io_cols,legible_cols):
        prompt = 'An information operation is dissemination of propaganda in pursuit of a competitive advantage over an opponent. Here are posts from an information operation in the form of a Python array:\n'+str(tweets)
        prompt += '\n Answer the following question about these posts with as few words as possible:\n'
        prompt += 'State the '+c_leg
        
        if c == 'Political goal category':
            prompt += ' where the categories are '+', '.join(ioaccounts['Political goal category'].dropna().drop_duplicates().values.tolist())
        if no_cot:
            prompt+='. Write your answer in one sentence.\n'
        else:
            prompt+='. Write your answer in one sentence. Create a new paragraph and explain each step when reaching your decision.\n'
        completion = client.chat.completions.create(model=model_engine,messages=[{"role": "user", "content": prompt}])
        response = completion.choices[0].message.content
        attempted_answer[c].append(response)

pd.DataFrame(attempted_answer).to_csv(model_engine+'_Answers_GPT4oGT_NoCOT='+str(no_cot)+'.csv',index=False)
