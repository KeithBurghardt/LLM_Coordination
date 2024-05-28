import country_converter as coco
import os,time
from glob import glob
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from openai import OpenAI
from transformers import pipeline

openai_api_key = ""


#test_model_engine = "gpt-3.5-turbo"
test_model_engine = "gpt-4o"
# we use these three model engines (BART is converted to bart-large-mnli)
nocot=True
for model_engine in ["bart","gpt-4","gpt-3.5-turbo","gpt-4o"]:
    if model_engine != "bart":
        client = OpenAI(
        # This is the default and can be omitted
        api_key=openai_api_key,
        )

    if model_engine == "bart":
        classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

    gt = pd.read_csv('CoordResults/gpt-4o_IO_GT.csv')
    answers = pd.read_csv('CoordResults/'+test_model_engine+'_Answers_GPT4oGT_NoCOT='+str(nocot)+'.csv')

    for c in gt.columns:
        if 'country' not in c: continue
        standard_names = coco.convert(names=gt[c].values, to='name_short')
        gt[c] = standard_names
        if test_model_engine == "gpt-4o":
           answers[c] = [a.split('\n')[0] for a in answers[c].values]

        standard_names = coco.convert(names=answers[c].values, to='name_short')
        answers[c] = standard_names
        print(answers[c].values[:3])
    cols = ['targeted country', 'attacking country', 'Political goal category','political goal ', 'event description']
    # check agreement 3 ways: BART, GPT-3.5, GPT-4

    checks = {c:[] for c in cols}
    for c in cols:
        print(c)
        if model_engine == "bart":
            for llm_ans,gt_ans in zip(answers[c].values,gt[c].values):
                sequence_to_classify = str(llm_ans)
                candidate_labels = [str(gt_ans)]
                conf = classifier(sequence_to_classify, candidate_labels)['scores'][0]
                checks[c].append(conf)
        else:
            prompt = 'Below are two statements. Write "yes" if the statements are equivalent and "no" otherwise'
            for llm_ans,gt_ans in zip(answers[c].values,gt[c].values):
                full_prompt = prompt + '\n'+str(llm_ans)+'\n\n'+str(gt_ans)
                completion = client.chat.completions.create(model=model_engine,messages=[{"role": "user", "content": full_prompt}])
                response = completion.choices[0].message.content
                checks[c].append(response)
    pd.DataFrame(checks).to_csv('CoordResults/'+test_model_engine+'_'+model_engine+'_llm_validation_GPT4oGT_NoCOT='+str(nocot)+'.csv',index=False)



