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

model_engine = "gpt-4o"
client = OpenAI(
    # This is the default and can be omitted
    api_key=openai_api_key,
)

fallacies = pd.read_csv('fallacies.csv',lineterminator='\n')
io_cols = ['targeted country', 'attacking country', 'Political goal category','political goal ', 'event description']
legible_cols = ['Country Targeted By The Information Operation', 'Country Running The Information Operation', 'Type of Political Goal','Political Goal of The Information Operation', 'Description of the Information Operation', 'Starting Month of the Information Operation']
bend_prompts = ['State which, if any, posts use an '+c for c in ['Engage tactic (bring up a related but relevant topic)','Explain tactic (provide details on or elaborate a topic)',' Excite tactic (elicit a positive emotion such as joy or excitement)','Enhance tactic (encourage the topic-group to continue with the topic)']]
bend_prompts += ['State which, if any, posts use a '+c for c in ['Dismiss tactic (explain why the topic is not important)','Distort tactic (alter the main message of the topic)','Dismay tactic  (elicit a negative emotion such as sadness or anger)','Distract tactic (discuss a completely different irrelevant topic)']]

disarm_cols = ['bait influencers', 'contain misinformation', 'contain some truth', 'distort truth', 'contain content unrelated to the main topic', 'have authors that claim to be experts', 'use a meme or joke', 'promote contests or prizes', 'direct users to alternative platforms', 'are harassing', 'call for a boycott or "cancel"', 'harass people based on identities', 'threaten to dox', 'doxes someone', 'promote a crowdfunding campaign', 'sell merchandise', 'encourage attendance at an event', 'encourage physical violence']
disarm_prompts = ['State which, if any, posts '+c for c in disarm_cols]


#'What is the frame signature'
frame_prompts = ['State the major themes from the posts',
'Here is a list of intentional fallacies: '+', '.join(fallacies['fallacy'].values.tolist())+'\n Create a list of which, if any, of these fallacies exist in the posts',
'List the cultural cues or in-group language is being used in the posts',
'List the language motifs  mentioned in the posts']
io_prompts = []
ioaccounts = pd.read_csv('CoordResults/gpt-4o_IO_GT.csv')
for c,c_leg in zip(io_cols,legible_cols):
    
    prompt = 'State the '+c_leg

    if c == 'Political goal category':
        prompt += ' where the categories are '+', '.join(ioaccounts['Political goal category'].dropna().drop_duplicates().values.tolist())
    #prompt += '. Write your answer in one sentence. Create a new paragraph and explain each step when reaching your decision.\n'
    

    #prompt = 'The below posts are from an information operation. State the '+c_leg.lower()+' for the posts.'    
    #if c == 'Political goal category':
    #    prompt += ' The categories are '+', '.join(ioaccounts['Political goal category'].dropna().drop_duplicates().values.tolist())
    #prompt += ' Provide your answers with as few words as possible. \n'
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

phase1a_file = 'INCASdatasets/master_p1a_usc_ta1/AllCombinedTwitterData+text_new.csv'#['INCASdatasets/master_p1a_usc_ta1/master_timeslice_two_remainder_actor_filtered_usc_ta1.json','INCASdatasets/master_p1a_usc_ta1/master_both_timeslice_one_remainder_actor_filterd_usc_ta1.json']
phase1a_G_file = 'INCASdatasets/master_p1a_usc_ta1/hashtag_coord_phase1b=False.edgelist'
phase1b_file = 'INCASdatasets/Phase1B/Phase1B_all_twitter_data.csv'
phase1b_G_file = 'INCASdatasets/Phase1B/hashtag_coord_phase1b=True.edgelist'
phase2a_file = 'INCASdatasets/Phase2A/sampled_twitter_en_tl_global_0805.jsonl'
phase2a_G_file = 'INCASdatasets/Phase2A/sampled_twitter_en_tl_global_0805_Twitter_hashtag_min_hash=3.edgelist'
challenge_file = 'INCASdatasets/Challenge/challenge_problem_two_21NOV.jsonl'
challenge_G_file = 'INCASdatasets/Challenge/challenge_problem_two_21NOV_Twitter_hashtag.edgelist'
#phase2b_file = 'incas2a_somer_pred_coord_tweets.csv'
phase2a_newfile = 'INCASdatasets/Phase2A/incas2a_somer_pred_coord_tweets.csv'
for hashtag_only in [False,True]:
    print('Add stance prompt: tell me the number of tweets with stance X')
    for coord_files,G_file in [[phase2a_newfile,'']]:#[[phase1a_file,phase1a_G_file],[phase1b_file,phase1b_G_file],[phase2a_file,phase2a_G_file],[challenge_file,challenge_G_file]]:
        data_type = 'phase1a'
        if str(coord_files) == str(phase1b_file):
            data_type = 'phase1b'
        elif str(coord_files) == str(phase2a_file):
            data_type = 'phase2a'
        elif str(coord_files) == str(challenge_file):
            data_type = 'challenge'
        elif str(coord_files) == str(phase2a_newfile):
            data_type = 'phase2a_new_fiona'
            if hashtag_only:
                data_type = 'phase2a_new_hashtags'


        if '.csv' in coord_files:
            coord_data = pd.read_csv(coord_files)
            if hashtag_only:
                coord_data = coord_data.loc[coord_data['keith_coord_label']==1,]
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
        if coord_files != phase2a_newfile:
            G_hashtag = nx.read_edgelist(G_file)
            # we only need to know users, not similarity
            coord_users = set(list(G_hashtag.nodes()))

            is_coord = [u in coord_users for u in coord_data['twitterAuthorScreenname'].values]
            coord_data = coord_data.loc[is_coord,]


        for tweet_type in ['all']:#['rt_only','non_rt','all']:
            coord_info_data = {p:[] for p in io_prompts+frame_prompts+bend_prompts+disarm_prompts}
            coord_info_data['tweets']=[]
            coord_info_data['problem'] = []
            coord_info_data['cause'] = []
            coord_info_data['remedy'] = []
            coord_info_data['metaphore'] = []
            coord_info_data['catchphrase'] = []
            coord_info_data['slogan'] = []
            coord_info_data['motif'] = []
            #coord_info_data['frame_examples'] = []
            if coord_files != phase2a_newfile:
                sG = nx.connected_components(G_hashtag)
                for s in sG:
                    print(len(s))
                    if len(s) < 4: continue
                    set_s = set(s)
                    is_component = [u in set_s for u in coord_data['twitterAuthorScreenname'].values]                
                    component_coord_data = coord_data.loc[is_component,]

                    tweets = component_coord_data.sample(100)['contentText'].values.tolist()
                    if tweet_type == 'rt_only':
                        tweets = component_coord_data.loc[coord_data['engagementType']=='retweet',].sample(100)['contentText'].values.tolist()
                    elif tweet_type =='non_rt':
                        tweets = component_coord_data.loc[coord_data['engagementType']!='retweet',].sample(100)['contentText'].values.tolist()
                    coord_info_data['tweets'].append(tweets)
                    #tweets = '\n'.join(tweets)
                    attempted_answer = {c:[] for c in io_cols}
                    for prompt in io_prompts+frame_prompts+bend_prompts+disarm_prompts:
                        #full_prompt =prompt+'\n'+tweets
                        full_prompt = 'An information operation is defined as dissemination of propaganda in pursuit of a competitive advantage over an opponent. Here are social media posts from an information operation in the form of a Python array:\n'+str(tweets)+prompt
                        full_prompt +='.\nWrite your answer in one sentence. Create a new paragraph and explain each step when reaching your decision.\n'
                        completion = client.chat.completions.create(model=model_engine,messages=[{"role": "user", "content": full_prompt}])
                        response = completion.choices[0].message.content
                        coord_info_data[prompt].append(response)
                    problem_prompt = 'An information operation is defined as dissemination of propaganda in pursuit of a competitive advantage over an opponent. Here are social media posts from an information operation in the form of a Python array:\n'+str(tweets)
                    problem_prompt += '\nState the main problem that the posts are mentioning explicitly or alluding to'
                    problem_prompt +='.\nWrite your answer in one sentence. Create a new paragraph and explain each step when reaching your decision.\n'
                    completion = client.chat.completions.create(model=model_engine,messages=[{"role": "user", "content": problem_prompt}])
                    problem = completion.choices[0].message.content
                    cause_prompt = 'An information operation is defined as dissemination of propaganda in pursuit of a competitive advantage over an opponent. Here are social media posts from an information operation in the form of a Python array:\n'+str(tweets)
                    cause_prompt += 'The problem stated in these posts is the following: '+problem+' State the cause of the problem that the posts are mentioning explicitly or alluding to'
                    cause_prompt +='.\nWrite your answer in one sentence. Create a new paragraph and explain each step when reaching your decision.\n'
                    
                    completion = client.chat.completions.create(model=model_engine,messages=[{"role": "user", "content": cause_prompt}])
                    cause = completion.choices[0].message.content
                    rem_prompt = 'An information operation is defined as dissemination of propaganda in pursuit of a competitive advantage over an opponent. Here are social media posts from an information operation in the form of a Python array:\n'+str(tweets)
                    
                    rem_prompt += 'The problem stated in these tweets is the following: '+problem+' The cause of the problem is: '+cause+' State the remedy of the problem that the following posts are mentioning explicitly or alluding to:'
                    rem_prompt +='.\nWrite your answer in one sentence. Create a new paragraph and explain each step when reaching your decision.\n'

                    completion = client.chat.completions.create(model=model_engine,messages=[{"role": "user", "content": rem_prompt}])
                    remedy = completion.choices[0].message.content
                    coord_info_data['problem'].append(problem)
                    coord_info_data['cause'].append(cause)
                    coord_info_data['remedy'].append(remedy)
                    pre_prompt = 'An information operation is defined as dissemination of propaganda in pursuit of a competitive advantage over an opponent. Here are social media posts from an information operation in the form of a Python array:\n'+str(tweets)
                    pre_prompt += 'The problem stated in these tweets is the following: '+problem+' The cause of the problem is: '+cause+'The remedy of the problem is: '+remedy
                    additional_questions = [pre_prompt+'. State the metaphores explicity mentioned related to the problem, cause, and remedy stated in the posts',
                                            pre_prompt+'. State the catchphrases explicity mentioned related to the problem, cause, and remedy stated in the posts',
                                            pre_prompt+'. State the slogans explicity mentioned related to the problem, cause, and remedy stated in the posts',
                                            pre_prompt+'. State the text motifs explicity mentioned related to the problem, cause, and remedy stated in the posts',
                                            pre_prompt+'. List the representative posts describing each problem, solution, and remedy']
                    for prompt in additional_questions:
                        prompt += '\n'+tweets
                        completion = client.chat.completions.create(model=model_engine,messages=[{"role": "user", "content": prompt}])
                        response = completion.choices[0].message.content
                        if 'State the metaphore' in prompt:
                            coord_info_data['metaphore'].append(response)
                        elif 'State the catchphrases' in prompt:
                            coord_info_data['catchphrase'].append(response)
                        elif 'State the slogans' in prompt:
                            coord_info_data['slogan'].append(response)
                        elif 'State the text motifs' in prompt:
                            coord_info_data['motif'].append(response)
                        #elif 'list the representative posts' in prompt:
                        #    coord_info_data['frame_examples'].append(response)
                    for key in coord_info_data.keys():
                        print([key,len(coord_info_data[key])])


            else:
                concerns = [c for c in coord_data.columns if 'Concern/' in c]
                coord_info_data['concerns']=[]
                for c in concerns:
                    coord_info_data['concerns'].append(c)
                    component_coord_data = coord_data.loc[coord_data[c]==1,]
                    try:
                        tweets = component_coord_data.sample(100)['contentText'].values.tolist()
                    except: # if we have <100 tweets
                        tweets = component_coord_data['contentText'].values.tolist()
                    if len(tweets)>100:
                        print('ERROR: TOO MANY TWEETS')
                        break
                    if tweet_type == 'rt_only':
                        tweets = component_coord_data.loc[coord_data['engagementType']=='retweet',].sample(100)['contentText'].values.tolist()
                    elif tweet_type =='non_rt':
                        tweets = component_coord_data.loc[coord_data['engagementType']!='retweet',].sample(100)['contentText'].values.tolist()
                    coord_info_data['tweets'].append(tweets)
                    attempted_answer = {c:[] for c in io_cols}
                    for prompt in io_prompts+frame_prompts+bend_prompts+disarm_prompts:
                        #full_prompt =prompt+'\n'+tweets
                        full_prompt = 'An information operation is defined as dissemination of propaganda in pursuit of a competitive advantage over an opponent. Here are social media posts from an information operation in the form of a Python array:\n'+str(tweets)+'\n'
                        full_prompt +=prompt
                        full_prompt +='.\nWrite your answer in one sentence. Create a new paragraph and explain each step when reaching your decision.\n'
                        #print(full_prompt+'\n==============\n\n')
                        completion = client.chat.completions.create(model=model_engine,messages=[{"role": "user", "content": full_prompt}])
                        response = completion.choices[0].message.content
                        #print(response)
                        coord_info_data[prompt].append(response)
                    problem_prompt = 'An information operation is defined as dissemination of propaganda in pursuit of a competitive advantage over an opponent. Here are social media posts from an information operation in the form of a Python array:\n'+str(tweets)+'\n'
                    problem_prompt += 'State the main problem that the posts are mentioning explicitly or alluding to'
                    problem_prompt +='.\nWrite your answer in one sentence. Create a new paragraph and explain each step when reaching your decision.\n'
                    #print(problem_prompt+'\n=============\n\n')
                    completion = client.chat.completions.create(model=model_engine,messages=[{"role": "user", "content": problem_prompt}])
                    problem = completion.choices[0].message.content
                    cause_prompt = 'An information operation is defined as dissemination of propaganda in pursuit of a competitive advantage over an opponent. Here are social media posts from an information operation in the form of a Python array:\n'+str(tweets)+'\n'
                    cause_prompt += 'The problem stated in these posts is the following: '+problem+' State the cause of the problem that the posts are mentioning explicitly or alluding to'
                    cause_prompt +='.\nWrite your answer in one sentence. Create a new paragraph and explain each step when reaching your decision.\n'
                    #print(cause_prompt+'\n=============\n\n')
                    completion = client.chat.completions.create(model=model_engine,messages=[{"role": "user", "content": cause_prompt}])
                    cause = completion.choices[0].message.content
                    rem_prompt = 'An information operation is defined as dissemination of propaganda in pursuit of a competitive advantage over an opponent. Here are social media posts from an information operation in the form of a Python array:\n'+str(tweets)+'\n'
                    
                    rem_prompt += 'The problem stated in these tweets is the following: '+problem+' The cause of the problem is: '+cause+' State the remedy of the problem that the posts are mentioning explicitly or alluding to'
                    rem_prompt +='.\nWrite your answer in one sentence. Create a new paragraph and explain each step when reaching your decision.\n'
                    #print(rem_prompt+'\n=============\n\n')
                    completion = client.chat.completions.create(model=model_engine,messages=[{"role": "user", "content": rem_prompt}])
                    remedy = completion.choices[0].message.content
                    coord_info_data['problem'].append(problem)
                    coord_info_data['cause'].append(cause)
                    coord_info_data['remedy'].append(remedy)
                    pre_prompt = 'An information operation is defined as dissemination of propaganda in pursuit of a competitive advantage over an opponent. Here are social media posts from an information operation in the form of a Python array:\n'+str(tweets)+'\n'
                    pre_prompt += 'The problem stated in these tweets is the following: '+problem+' The cause of the problem is: '+cause+'The remedy of the problem is: '+remedy
                    additional_questions = [pre_prompt+'. State the metaphores explicity mentioned related to the problem, cause, and remedy stated in the posts',
                                            pre_prompt+'. State the catchphrases explicity mentioned related to the problem, cause, and remedy stated in the posts',
                                            pre_prompt+'. State the slogans explicity mentioned related to the problem, cause, and remedy stated in the posts',
                                            pre_prompt+'. State the text motifs explicity mentioned related to the problem, cause, and remedy stated in the posts',
                                            pre_prompt+'. List the representative posts describing each problem, solution, and remedy']
                    for prompt in additional_questions:
                        full_prompt =prompt+'.\nWrite your answer in one sentence. Create a new paragraph and explain each step when reaching your decision.\n'
                        print(full_prompt+'\n=============\n\n')
                        completion = client.chat.completions.create(model=model_engine,messages=[{"role": "user", "content": prompt}])
                        response = completion.choices[0].message.content
                        if 'State the metaphore' in prompt:
                            coord_info_data['metaphore'].append(response)
                        elif 'State the catchphrases' in prompt:
                            coord_info_data['catchphrase'].append(response)
                        elif 'State the slogans' in prompt:
                            coord_info_data['slogan'].append(response)
                        elif 'State the text motifs' in prompt:
                            coord_info_data['motif'].append(response)
                        #elif 'list the representative posts' in prompt:
                        #    coord_info_data['frame_examples'].append(response)
                    for key in coord_info_data.keys():
                        print([key,len(coord_info_data[key])])
                    
            pd.DataFrame(coord_info_data).to_csv(model_engine+'_coord_data_'+data_type+'_'+tweet_type+'_hashtags_only='+str(hashtag_only)+'.csv',index=False)
