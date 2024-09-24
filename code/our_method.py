from utils import match


def annotation(res, sample):
    
    if match(res, res['rag']['answer'], sample['reference'], 'rag'):
        res['label_decision'] = "keep"
    else:
        res['label_decision'] = "discard"


def decision(res, prompt_type):

    # 增加拒绝的判断
    if res[prompt_type]['answer'] and res['rag']['answer']: 
        answer_matching = match(res, res[prompt_type]['answer'], [res['rag']['answer']], prompt_type)

        if not answer_matching:
            res[prompt_type]['pred_decision'] = 'discard'
        else:
            res[prompt_type]['pred_decision'] = 'continue'
    else:
        if not res[prompt_type]['reject'] and (res[prompt_type]['evidence'] and res['rag']['evidence']):
            evidence_matching = match(res, res[prompt_type]['evidence'], [res['rag']['evidence']], prompt_type)
            if not evidence_matching:
                res[prompt_type]['pred_decision'] = 'discard'
            else:
                res[prompt_type]['pred_decision'] = 'continue'
        else:
            res[prompt_type]['pred_decision'] = 'discard'


def count_fold(res, AK, AD, UK, UD):
    if res['label_decision'] == 'keep' and res['pred_decision'] == 'keep':
        AK+=1
    elif res['label_decision'] == 'keep' and res['pred_decision'] == 'discard':
        AD+=1
    elif res['label_decision'] == 'discard' and res['pred_decision'] == 'keep':
        UK+=1
    elif res['label_decision'] == 'discard' and res['pred_decision'] == 'discard':
        UD+=1
    return AK, AD, UK, UD