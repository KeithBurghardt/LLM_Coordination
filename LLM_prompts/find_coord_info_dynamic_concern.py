import os,time
from datetime import datetime
import ast
from glob import glob
import openai
import pandas as pd
import numpy as np
import networkx as nx
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from openai import OpenAI

#openai_api_key = ""
#model_engine = "gpt-4-turbo-preview"#
model_engine = "gpt-3.5-turbo"
client = OpenAI(
    # This is the default and can be omitted
    api_key=openai_api_key,
)

fallacies = pd.read_csv('fallacies.csv',lineterminator='\n')
io_cols = ['targeted country', 'attacking country', 'Political goal category','political goal ', 'event description', 'starting month']
legible_cols = ['Country Targeted By The Information Operation', 'Country Running The Information Operation', 'Type of Political Goal','Political Goal of The Information Operation', 'Description of the Information Operation', 'Starting Month of the Information Operation']
bend_prompts = ['The below posts are from a coordinated campaign. Explain which, if any, posts use an Engage tactic (bring up a related but relevant topic)',
'The below posts are from a coordinated campaign. Explain which, if any, posts use an Explain tactic (provide details on or elaborate a topic)',
'The below posts are from a coordinated campaign. Explain which, if any, posts use an Excite tactic (elicit a positive emotion such as joy or excitement)',
'The below posts are from a coordinated campaign. Explain which, if any, posts use an Enhance tactic (encourage the topic-group to continue with the topic)',
'The below posts are from a coordinated campaign. Explain which, if any, posts use a Dismiss tactic (explain why the topic is not important)',
'The below posts are from a coordinated campaign. Explain which, if any, posts use a Distort tactic (alter the main message of the topic)',
'The below posts are from a coordinated campaign. Explain which, if any, posts use a Dismay tactic  (elicit a negative emotion such as sadness or anger)',
'The below posts are from a coordinated campaign. Explain which, if any, posts use a Distract tactic (discuss a completely different irrelevant topic). Show examples of these posts and explain why these use the tactic.']


#'What is the frame signature'
frame_prompts = ['State the major themes from the following posts:',
'Here is a list of intentional fallacies: '+', '.join(fallacies['fallacy'].values.tolist())+'\n Create a list of which, if any, of these fallacies exist in the following posts and explain why this fallacies exists. Finally, provide an example of a post associated with each fallacy:',
'List the cultural cues or in-group language is being used in the following posts, and explain what is the culture or in-group implied by these posts:',
'List the language motifs  mentioned in the following posts:']
io_prompts = []
for c,c_leg in zip(io_cols,legible_cols):
    prompt = 'The below posts are from an information operation. State the '+c_leg.lower()+' for the posts.'
    prompt += ' Provide your answers with as few words as possible. \n'
    io_prompts.append(prompt)


# concern (maybe hazard, emotion)
# dynamic
# - Challenge: (2022,10,7), (2022,10,27)
# - Phase 1a: (2017,4,27), (2017,5,7)
# - Phase 1b: datetime(2022,4,10),datetime(2022,4,24)
# - Phase 2a: datetime(2023,4,11), datetime(2023,4,28)
# Compare against Princeton datasets - can we match IOs in Phases to Priceton data? What about via Twitter data?
# Split data by concerns (or topics if challenge)
# Use LLM to say "if concern is terrorism, what are they saying, etc."

phase1a_file = 'AllCombinedTwitterData+text_new.csv'
phase1a_G_file = 'hashtag_coord_phase1b=False.edgelist'
phase1b_file = 'Phase1B_all_twitter_data.csv'
phase1b_G_file = 'hashtag_coord_phase1b=True.edgelist'
phase2a_file = 'sampled_twitter_en_tl_global_0805.jsonl'
phase2a_G_file = 'sampled_twitter_en_tl_global_0805_Twitter_hashtag_min_hash=3.edgelist'
challenge_file = 'challenge_problem_two_21NOV.jsonl'
challenge_G_file = 'challenge_problem_two_21NOV_Twitter_hashtag.edgelist'
tweet_sample_size = 20
coord_size = 10
for coord_files,G_file in [[phase1b_file,phase1b_G_file]]:#[[phase1b_file,phase1b_G_file],[phase1a_file,phase1a_G_file],[phase2a_file,phase2a_G_file],[challenge_file,challenge_G_file]]:
    data_type = 'phase1a'
    concerns = pd.DataFrame({'empty':[]})
    if str(coord_files) == str(phase2a_file):
        data_type = 'phase2a'
        concerns = pd.read_json('concern_phase2a_twitter_hashtag.json')
    if str(coord_files) == str(phase1b_file):
        data_type = 'phase1b'
        concerns = pd.concat([pd.read_json('phase1b_twitter_t1.json'),pd.read_json('phase1b_twitter_t2.json')])
    if str(coord_files) == str(challenge_file):
        data_type = 'challenge'
        concerns = pd.DataFrame({'empty':[]})

    G_hashtag = nx.read_edgelist(G_file)
    # we only need to know users, not similarity
    coord_users = set(list(G_hashtag.nodes()))
    if '.csv' in coord_files:
        coord_data = pd.read_csv(coord_files)
    elif '.json' in coord_files:
        coord_data = pd.read_json(coord_files,lines=True)
        coord_data = coord_data.loc[['twitterData' in l.keys() for l in coord_data['mediaTypeAttributes'].values],]
        if 'twitterAuthorScreenname' not in coord_data.columns:
            usernames =[l['twitterData']['twitterAuthorScreenname'] for l in coord_data['mediaTypeAttributes'].values]
            coord_data['twitterAuthorScreenname'] = usernames
            engagement = [l['twitterData']['engagementType'] for l in coord_data['mediaTypeAttributes'].values]
            coord_data['engagementType'] = engagement
    else: 
        print('CANNOT PARSE ',coord_files)
        break
    date = [datetime.fromtimestamp(t/1000) if str(t) != 'nan' else np.nan  for t in coord_data['timePublished'].values]
    coord_data['date'] = date

    is_coord = [u in coord_users for u in coord_data['twitterAuthorScreenname'].values]
    coord_data = coord_data.loc[is_coord,]
    if len(concerns) >0:
        coord_data = pd.merge(coord_data,concerns[['id','concern_labels']],on='id')
        concern_cols = []
        for concerns in coord_data['concern_labels'].drop_duplicates().values:
            concern_cols += concerns
        concern_cols = list(set(concern_cols))
        for c in concern_cols:
            coord_data[c] = [int(c in concerns) for concerns in coord_data['concern_labels'].values]

    elif data_type != 'challenge': # concern already in file
        coord_data.columns = [c.replace('_x','') for c in coord_data.columns]
        concern_cols = [c for c in coord_data.columns if 'concern' in c.lower() and '_y' not in c]
    else:
        concern_cols = ['all']
    if data_type == 'phase1a':
        # - Phase 1a: (2017,4,27), (2017,5,7)
        times = [datetime(2017,1,1),datetime(2017,4,27),datetime(2017,5,7),datetime(2023,12,31)]
    if data_type == 'phase1b':
        # - Phase 1b: datetime(2022,4,10),datetime(2022,4,24)
        times = [datetime(2017,1,1),datetime(2022,4,10),datetime(2022,4,24),datetime(2023,12,31)]
    if data_type == 'phase2a':
        # - Phase 2a: datetime(2023,4,11), datetime(2023,4,28)
        times = [datetime(2017,1,1), datetime(2023,4,11),datetime(2023,4,28),datetime(2023,12,31)]
    if data_type == 'challenge':    
        # - Challenge: (2023,10,7), (2023,10,27)
        times = [datetime(2022,1,1),datetime(2023,10,7),datetime(2023,10,27),datetime(2023,12,31)]
    for add_concern in [False]:#[True,False]:
        if add_concern == True and concern_cols == ['all']: continue
        for concern in concern_cols:
            concern_str = concern.replace('_',' ').replace('-',' ')
            if data_type == 'phase1a':
                 concern_str = concern_str.split('3')[0]
            print(concern_str)
            concern_coord_data = coord_data.copy(deep=True)
            if concern != 'all':
                concern_coord_data = coord_data.loc[coord_data[concern]>0.5,]
            for r1,r2 in zip(times[:-1],times[1:]):
                for tweet_type in ['all']:#['rt_only','non_rt','all']:
                    outfile = model_engine+'_coord_data_'+data_type+'_'+tweet_type+'_'+str(r1)+'-'+str(r2)+'_concern='+concern+'_explicitly-stated='+str(add_concern)+'.csv'
                    if concern_str != concern:
                        outfile = model_engine+'_coord_data_'+data_type+'_'+tweet_type+'_'+str(r1)+'-'+str(r2)+'_concern='+concern+'_explicitly-stated='+str(add_concern)+'_updatedconcern.csv'
                    if os.path.exists(outfile):
                        if len(pd.read_csv(outfile).dropna()) > 0: continue
                    coord_info_data = {p:[] for p in io_prompts+frame_prompts+bend_prompts}
                    coord_info_data['size']=[]
                    coord_info_data['tweets']=[]
                    coord_info_data['problem'] = []
                    coord_info_data['cause'] = []
                    coord_info_data['remedy'] = []
                    coord_info_data['metaphore'] = []
                    coord_info_data['catchphrase'] = []
                    coord_info_data['slogan'] = []
                    coord_info_data['motif'] = []
                    coord_info_data['frame_examples'] = []

                    sG = nx.connected_components(G_hashtag)
                    for s in sG:
                        print(len(s))
                        if len(s) < coord_size: continue
                        coord_info_data['size'].append(len(s))
                        set_s = set(s)
                        is_component = [u in set_s for u in concern_coord_data['twitterAuthorScreenname'].values]                
                        component_coord_data = concern_coord_data.loc[is_component,]
                        if len(component_coord_data.loc[(component_coord_data['date']>r1) & (component_coord_data['date']<=r2),]) > tweet_sample_size:
                            tweets = component_coord_data.loc[(component_coord_data['date']>r1) & (component_coord_data['date']<=r2)].sample(tweet_sample_size)['contentText'].values.tolist()
                        else:
                            tweets = component_coord_data['contentText'].values.tolist()
                        if tweet_type == 'rt_only':
                            tweets = component_coord_data.loc[(component_coord_data['engagementType']=='retweet') & (component_coord_data['date']>r1) & (component_coord_data['date']<=r2),]['contentText'].values.tolist()
                            if len(tweets) > tweet_sample_size:
                                tweets = component_coord_data.loc[(component_coord_data['engagementType']=='retweet')& (component_coord_data['date']>r1) & (component_coord_data['date']<=r2),].sample(tweet_sample_size)['contentText'].values.tolist()
                        elif tweet_type =='non_rt':
                            tweets = component_coord_data.loc[(component_coord_data['engagementType']!='retweet') & (component_coord_data['date']>r1) & (component_coord_data['date']<=r2),]['contentText'].values.tolist()
                            if len(tweets) > tweet_sample_size:
                                tweets = component_coord_data.loc[(component_coord_data['engagementType']!='retweet') & (component_coord_data['date']>r1) & (component_coord_data['date']<=r2),].sample(tweet_sample_size)['contentText'].values.tolist() 

                        coord_info_data['tweets'].append(tweets)
                        tweets = '\n'.join(tweets)
                        attempted_answer = {c:[] for c in io_cols}
                        for prompt in io_prompts+frame_prompts+bend_prompts:
                            full_prompt =prompt+'\n'+tweets
                            if add_concern and concern != 'all':
                                full_prompt+='\n Keep in mind that these tweets are about the topic: '+concern_str+'.'
                            try:
                                completion = client.chat.completions.create(model=model_engine,messages=[{"role": "user", "content": full_prompt}])
                                response = completion.choices[0].message.content
                            except:
                                response = ''
                            coord_info_data[prompt].append(response)
                        problem_prompt = 'State the main problem that the following posts are mentioning explicitly or alluding to:'
                        problem_prompt += '\n'+tweets
                        if add_concern and concern != 'all':
                            problem_prompt+='\n Keep in mind that these tweets are about the topic: '+concern_str+'.'

                        try:
                            completion = client.chat.completions.create(model=model_engine,messages=[{"role": "user", "content": problem_prompt}])
                            problem = completion.choices[0].message.content
                        except:
                            problem = ''
                        cause_prompt = 'The problem stated in these tweets is the following: '+problem+' State the cause of the problem that the following posts are mentioning explicitly or alluding to:'
                        cause_prompt += '\n'+tweets
                        if add_concern and concern != 'all':
                            cause_prompt+='\n Keep in mind that these tweets are about the topic: '+concern_str+'.'

                        try:
                            completion = client.chat.completions.create(model=model_engine,messages=[{"role": "user", "content": cause_prompt}])
                            cause = completion.choices[0].message.content
                        except:
                            cause = ''
                        rem_prompt = 'The problem stated in these tweets is the following: '+problem+' The cause of the problem is: '+cause+' State the remedy of the problem that the following posts are mentioning explicitly or alluding to:'
                        rem_prompt += '\n'+tweets
                        if add_concern and concern != 'all':
                            rem_prompt+='\n Keep in mind that these tweets are about the topic: '+concern_str+'.'

                        try:
                            completion = client.chat.completions.create(model=model_engine,messages=[{"role": "user", "content": rem_prompt}])
                            remedy = completion.choices[0].message.content
                        except:
                            remedy = ''

                        coord_info_data['problem'].append(problem)
                        coord_info_data['cause'].append(cause)
                        coord_info_data['remedy'].append(remedy)
                        additional_questions = ['The problem stated in these tweets is the following: '+problem+' The cause of the problem is: '+cause+'The remedy of the problem is: '+remedy+' State the metaphores explicity mentioned related to the problem, cause, and remedy stated in the following posts:',
                                    'The problem stated in these tweets is the following: '+problem+' The cause of the problem is: '+cause+'The remedy of the problem is: '+remedy+' State the catchphrases explicity mentioned related to the problem, cause, and remedy stated in the following posts:',
                                    'The problem stated in these tweets is the following: '+problem+' The cause of the problem is: '+cause+'The remedy of the problem is: '+remedy+' State the slogans explicity mentioned related to the problem, cause, and remedy stated in the following posts:',
                                    'The problem stated in these tweets is the following: '+problem+' The cause of the problem is: '+cause+'The remedy of the problem is: '+remedy+' State the text motifs explicity mentioned related to the problem, cause, and remedy stated in the following posts:',
                                    'The problem stated in these tweets is the following: '+problem+' The cause of the problem is: '+cause+'The remedy of the problem is: '+remedy+' Given the following text, list the representative posts describing each problem, solution, and remedy:']             
                        for prompt in additional_questions:
                            prompt += '\n'+tweets
                            if add_concern and concern != 'all':
                                prompt+='\n Keep in mind that these tweets are about the topic: '+concern_str+'.'

                            try:
                                completion = client.chat.completions.create(model=model_engine,messages=[{"role": "user", "content": prompt}])
                                response = completion.choices[0].message.content
                            except:
                                response = ''
                            if 'State the metaphore' in prompt:
                                coord_info_data['metaphore'].append(response)
                            elif 'State the catchphrases' in prompt:
                                coord_info_data['catchphrase'].append(response)
                            elif 'State the slogans' in prompt:
                                coord_info_data['slogan'].append(response)
                            elif 'State the text motifs' in prompt:
                                coord_info_data['motif'].append(response)
                            elif 'list the representative posts' in prompt:
                                coord_info_data['frame_examples'].append(response)
                        for key in coord_info_data.keys():
                            print([key,len(coord_info_data[key])])
                    pd.DataFrame(coord_info_data).to_csv(outfile,index=False)

