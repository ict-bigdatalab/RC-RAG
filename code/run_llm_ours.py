import os
from tqdm import tqdm
import argparse
from llm_ours import get_init_message, get_new_message, get_llm_result
from prompt import get_single_prompt
from utils import (load_source, save_result, 
                   deal_fusion, deal_prediction_with_evidence, deal_fusion_probability)
from our_method import annotation, decision, count_fold
import copy

# from vllm import LLM, SamplingParams

ra_dict = {
    'sparse': 'sparse_ctxs',
    'dense': 'dense_ctxs',
    'gold': 'gold_ctxs',
}


def get_args():

    parser = argparse.ArgumentParser()
    # parser.add_argument('--do_annotate', action="store_true", help="Whether or not to annotate label.")
    # parser.add_argument('--do_inference', action="store_true", help="Whether or not to prompting generate.")
    # parser.add_argument('--do_eval', action="store_true", help="Whether or not to eval.")
    
    # parser.add_argument('--data_start_index', type=int, help="Load data from this index.", required=True)
    # parser.add_argument('--data_end_index', type=int, help="Load data until this index.", required=True)

    # parser.add_argument("--data_name", type=str, help="The name of the dataset.", required=True)
    # parser.add_argument('--model_type', type=str, choices=['chatgpt', 'vllm_mistral', 'gomall_mistral', 'gomall_llama3'], required=True)
    # parser.add_argument("--prompt_trick", type=str, default=None, help="The trick for prompting generation.")

    # parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for LLM.")
    # parser.add_argument('--max_tokens', type=int, default=256, help="Number of generated tokens.")
    # parser.add_argument("--num_passage", type=int, default=3, help="Number of retrieved passages.")
    # parser.add_argument('--ra', type=str, help="Type of retriever.", required=True)

    # parser.add_argument("--num_iteration", type=int, help="Number of iteration.", required=True)
    # parser.add_argument('--re_fusion', action="store_true", help="Change fusion_strategy.")
    # parser.add_argument("--fusion_strategy", type=str, default="prompt", help="The type of fusion strategy.")

    # parser.add_argument('--annotated_file', type=str, required=True)
    # parser.add_argument('--source', type=str, required=True)
    # parser.add_argument('--res_file', type=str)
    # parser.add_argument('--output_dir', default='output', type=str)
    # parser.add_argument('--outfile', default='ours', type=str)
    args = parser.parse_args()

    args.data_start_index=0
    args.data_end_index=7785
    
    args.do_annotate=False # don't modify ###
    args.do_inference=False
    args.do_eval=True
    
    args.data_name='tq'
    args.model_type='chatgpt'
    args.prompt_trick=None #"ioe"

    args.temperature=0.0
    args.max_tokens=256
    args.num_passage=3
    args.ra="dense"
    
    args.num_iteration=1
    args.re_fusion=True #
    args.fusion_strategy = "probability" #
    # "prompt"

    args.annotated_file='output/tq_chatgpt_None_dense/ioe_inference.json'
    args.source=f'data/source/{args.data_name}.json'
    args.res_file='output/tq_chatgpt_None_dense/fuse_prompt_iter_3_1_inference.json' #
    args.output_dir=f'output/{args.data_name}_{args.model_type}_{args.prompt_trick}_{args.ra}'
    args.outfile=f'fuse_{args.fusion_strategy}_iter_{args.num_iteration}'

    args.ra_type=ra_dict[args.ra]


    return args

def rag_generation_annotation(args, res, sample, vllm_bags):

    ####################
    single_prompt = get_single_prompt(args, 'rag', sample)
    messages = get_init_message(single_prompt)
    llm_result = get_llm_result(args, messages, vllm_bags)
    res['rag']['reject'], res['rag']['answer'], res['rag']['evidence'] = deal_prediction_with_evidence(llm_result)
    
    for ans in sample['reference']:
        if " or " in ans:
            new_ans = ans.split(" or ")
            sample['reference'].extend(new_ans)
    sample['reference']=list(set(sample['reference']))

    annotation(res, sample)

    return messages


def ours_generation_decision_fusion(args, res, sample, vllm_bags, messages):
    messages1 = copy.deepcopy(messages)
    # ####################
    for i in range(args.num_iteration):
        single_prompt = get_single_prompt(args, 'cf_use', sample)
        if i == 0:
            messages.extend(get_new_message(res['rag']['answer'], single_prompt))
        else:
            messages.extend(get_new_message(res['cf_use']['answer'], single_prompt))
        llm_result = get_llm_result(args, messages, vllm_bags)

        res['cf_use']['reject'], res['cf_use']['answer'], res['cf_use']['evidence'] = deal_prediction_with_evidence(llm_result)
    
        decision(res, 'cf_use')

        if res['cf_use']['pred_decision'] == "discard":
            break
    if res['cf_use']['pred_decision'] == "continue":
        res['cf_use']['pred_decision'] = "keep"

    # # ####################
    for i in range(args.num_iteration):
        single_prompt = get_single_prompt(args, 'cf_quality', sample)
        if i == 0:
            messages1.extend(get_new_message(res['rag']['answer'], single_prompt))
        else:
            messages1.extend(get_new_message(res['cf_quality']['answer'], single_prompt))
        llm_result = get_llm_result(args, messages1, vllm_bags)

        res['cf_quality']['reject'], res['cf_quality']['answer'], res['cf_quality']['evidence'] = deal_prediction_with_evidence(llm_result)

        decision(res, 'cf_quality')

        if res['cf_quality']['pred_decision'] == "discard":
            break
    if res['cf_quality']['pred_decision'] == "continue":
        res['cf_quality']['pred_decision'] = "keep"
    
    fusion(args, res, sample, vllm_bags, messages, messages1)


def fusion(args, res, sample, vllm_bags, messages, messages1):
    if res['cf_use']['pred_decision'] == res['cf_quality']['pred_decision']:
        res['pred_decision'] = res['cf_use']['pred_decision']
        return

    if args.fusion_strategy == "safety":
        if res['cf_use']['pred_decision'] == "discard" or res['cf_quality']['pred_decision'] == "discard":
            res['pred_decision'] = "discard"
        else:
            res['pred_decision'] = "keep"
    elif args.fusion_strategy == "prompt":

        if res['cf_use']['pred_decision'] == "discard":
            single_prompt = get_single_prompt(args, 'fusion_use', sample)
            messages.extend(get_new_message(res['cf_use']['answer'], single_prompt))
            llm_result = get_llm_result(args, messages, vllm_bags)

        else:
            single_prompt = get_single_prompt(args, 'fusion_quality', sample)
            messages1.extend(get_new_message(res['cf_quality']['answer'], single_prompt))
            llm_result = get_llm_result(args, messages1, vllm_bags)

        res['pred_decision'] = deal_fusion(llm_result)

    elif args.fusion_strategy == "probability":

        single_prompt = get_single_prompt(args, 'fusion_probability', sample)

        messages.extend(get_new_message(res['cf_use']['answer'], single_prompt))
        llm_result = get_llm_result(args, messages, vllm_bags)
        _, answer_use, _ = deal_prediction_with_evidence(llm_result)
        fusion_probability_use = deal_fusion_probability(answer_use)
        print(fusion_probability_use)

        messages1.extend(get_new_message(res['cf_quality']['answer'], single_prompt))
        llm_result = get_llm_result(args, messages1, vllm_bags)
        _, answer_quality, _ = deal_prediction_with_evidence(llm_result)
        fusion_probability_quality = deal_fusion_probability(answer_quality)
        print(fusion_probability_quality)

        if fusion_probability_use > fusion_probability_quality:
            res['pred_decision'] = res['cf_use']['pred_decision']
        elif fusion_probability_use < fusion_probability_quality:
            res['pred_decision'] = res['cf_quality']['pred_decision']
        else:
            res['pred_decision'] = "discard"


def main():

    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)
    inference_save_path = os.path.join(args.output_dir, args.outfile+'_inference.json')
    eval_save_path = os.path.join(args.output_dir, args.outfile+'_eval.json')

    if "gomall" in args.model_type:
        if "mistral" in args.model_type:
            args.url = "http://10.208.62.156:9999/mistral_7b"
        elif "llama3" in args.model_type:
            args.url = "http://10.208.62.156:9999/llama3_8b_instruct"


    if "mistral" in args.model_type:
        args.model_name_or_path='/home/gomall/models/Mistral_7B_Instruct_v0-2/'
    elif "llama3" in args.model_type:
        args.model_name_or_path='/home/gomall/models/llama3_8b_instruct/'

    vllm_bags = None
    if "vllm" in args.model_type:
        sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
        llm_model = LLM(model=args.model_name_or_path)
        vllm_bags = (sampling_params, llm_model)

    all_data = load_source(args.source)
    if args.do_inference and not args.do_annotate:
        annotated_data = load_source(args.annotated_file)
    if args.re_fusion and not args.do_inference:
        res_data = load_source(args.res_file)


    if args.do_eval:
        AK, AD, UK, UD = 0, 0, 0, 0


    for sample in tqdm(all_data[args.data_start_index:args.data_end_index]):

        if args.do_inference:

            res = {'rag': {}, 'cf_use': {}, 'cf_quality': {}}

            if args.do_annotate:
               messages = rag_generation_annotation(args, res, sample, vllm_bags)

            else:
                res['rag'] = annotated_data[sample['id']]['rag']
                res['label_decision'] = annotated_data[sample['id']]['label_decision']

                single_prompt = get_single_prompt(args, 'rag', sample)
                messages = get_init_message(single_prompt)

            ours_generation_decision_fusion(args, res, sample, vllm_bags, messages)

            if res['pred_decision'] == "keep" and res['rag']['reject']: ####
                res['pred_decision'] = "discard"
            
            res['id'] = sample['id']
            res['answer'] = sample['reference']
            save_result(inference_save_path, res)
            
        if args.do_eval:
            if not args.do_inference:
                # res = sample
                res = res_data[sample['id']]

                if args.re_fusion:
                    single_prompt = get_single_prompt(args, 'rag', sample)
                    messages = get_init_message(single_prompt)
                    messages1 = copy.deepcopy(messages)

                    single_prompt = get_single_prompt(args, 'cf_use', sample)
                    messages.extend(get_new_message(res['rag']['answer'], single_prompt))

                    single_prompt = get_single_prompt(args, 'cf_quality', sample)
                    messages1.extend(get_new_message(res['rag']['answer'], single_prompt))

                    fusion(args, res, sample, vllm_bags, messages, messages1)

                    if res['pred_decision'] == "keep" and res['rag']['reject']: ####
                        res['pred_decision'] = "discard"

                    save_result(inference_save_path, res)

            AK, AD, UK, UD = count_fold(res, AK, AD, UK, UD)


    if args.do_eval:
        risk = UK/(AK+UK)
        print('risk: '+str(risk))
        overcaution = AD/(UD+AD)
        print('overcaution: '+str(overcaution))
        recall = AK/(AK+AD)
        print('recall: '+str(recall))
        carefulness = UD/(UK+UD)
        print('carefulness: '+str(carefulness))
        alignment = (AK+UD)/(AK+AD+UK+UD)
        print('alignment: '+str(alignment))
        coverage = (AK+UK)/(AK+AD+UK+UD)
        print('coverage: '+str(coverage))

        print('AK, AD: '+str(AK)+' '+str(AD))
        print('UK, UD: '+str(UK)+' '+str(UD))

        eval_res = {
            "risk": risk,
            "overcaution": overcaution,
            "recall": recall,
            "carefulness": carefulness,
            "alignment": alignment,
            "coverage": coverage,
            "AK": AK,
            "AD": AD,
            "UK": UK,
            "UD": UD
        }
        
        save_result(eval_save_path, eval_res)


if __name__ == '__main__':
    main()