def get_single_prompt(args, prompt_type, sample):
    def get_prompt_form(question, passage):
        sequential_prompt_dict = {
            'rag': f"Answer the following question based on the given passages with one or few words. Provide your evidence between two ## symbols at the end of your response, either the passage id or your internal knowledge. For example, provide \"Answer: apple.\nEvidence: ## Passage-0, Passage-1 ##.\" if you are referring to Passage-0 and Passage-1 to obtain the answer \"apple\". If there is no information in the passages, explain the answer by yourself.\nQuestion: {question}\nPassages: {passage}",

            'cf_use': f"Assume that your answer is wrong due to your improper use of the retrieved passages. Please read the given passages carefully to regenerate the answer with one or few words.",

            # 'cf_quality': f"Assume that your answer is wrong because the quality of your referred passages is poor. Please re-select the passages  to regenerate the answer with one or few words, followed by the passage id as evidence.", #  and your referred passage id as evidence

            'cf_quality': f"Assume that your answer is wrong because the quality of your referred passages is poor. Please re-select the passages, to regenerate the answer with one or few words and your referred passage id as evidence.", #  

            'fusion_use': f"Your answer is likely to be wrong because of the improper use of retrieval passages, please choose to keep or discard this output. Generate $$ keep $$ if you choose to keep this answer, otherwise, generate $$ discard $$.",

            'fusion_quality': f"Your answer is likely to be wrong because of the poor quality of retrieval passages, please choose to keep or discard this output. Generate $$ keep $$ if you choose to keep this answer, otherwise, generate $$ discard $$.",


            'fusion_probability': f"Provide the probability that your regenerated answer is correct. Give ONLY the probability, no other words or explanation.\n\nFor example:\n\nProbability: <the probability between 0.0 and 1.0 that your specific answer is correct, without any extra commentary whatsoever; just the probability!>\n",


            'priori': f'Given the following information: \n{passage}\nCan you answer the following question based on the given information or your internal knowledge, if yes, you should give a short answer with one or few words, if no, you should answer \"Unknown\".\nQuestion: {question}',

            'ioe': f'If you are very confident about your answer, maintain your answer. Otherwise, update your answer.', #  Your final answer should be put between two ## symbols, like ## ANSWER ##, at the end of your response.

            'ioe_decision': f'You give two different answers in previous responses. Check the problem and your answers again, and give the best answer.', #  Your final answer should be put between two ## symbols, like ## ANSWER ##, at the end of your response.

            'calibration': f'Provide the probability that your answer is correct. Give ONLY the probability, no other words or explanation.\n\nFor example:\n\nProbability: <the probability between 0.0 and 1.0 that your specific answer is correct, without any extra commentary whatsoever; just the probability!>\n',
        }
        
        return sequential_prompt_dict[prompt_type]


    question = sample['question']

    passage = None
    if prompt_type in ['rag', 'priori']:
        passage_lst = []
        for i in range(args.num_passage):
            passage_lst.append(("Passage-%d" % i) + sample[args.ra_type][i])
        passage = '\n'.join(passage_lst)

    if "fusion" in prompt_type:
        tail=''
    else:
        tail = '\nAnswer: '

    prompt = get_prompt_form(question, passage) + tail
    
    return prompt