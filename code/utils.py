import collections
import json
import copy
import re
import logging
import string
import regex
import unicodedata
from datasets import load_metric
metric = load_metric('rouge.py')


logger = logging.getLogger()


def has_answer(answers, text, match_type="string"):
    class Tokens(object):
        """A class to represent a list of tokenized text."""
        TEXT = 0
        TEXT_WS = 1
        SPAN = 2
        POS = 3
        LEMMA = 4
        NER = 5

        def __init__(self, data, annotators, opts=None):
            self.data = data
            self.annotators = annotators
            self.opts = opts or {}

        def __len__(self):
            """The number of tokens."""
            return len(self.data)

        def slice(self, i=None, j=None):
            """Return a view of the list of tokens from [i, j)."""
            new_tokens = copy.copy(self)
            new_tokens.data = self.data[i: j]
            return new_tokens

        def untokenize(self):
            """Returns the original text (with whitespace reinserted)."""
            return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

        def words(self, uncased=False):
            """Returns a list of the text of each token
            Args:
                uncased: lower cases text
            """
            if uncased:
                return [t[self.TEXT].lower() for t in self.data]
            else:
                return [t[self.TEXT] for t in self.data]

        def offsets(self):
            """Returns a list of [start, end) character offsets of each token."""
            return [t[self.SPAN] for t in self.data]

        def pos(self):
            """Returns a list of part-of-speech tags of each token.
            Returns None if this annotation was not included.
            """
            if 'pos' not in self.annotators:
                return None
            return [t[self.POS] for t in self.data]

        def lemmas(self):
            """Returns a list of the lemmatized text of each token.
            Returns None if this annotation was not included.
            """
            if 'lemma' not in self.annotators:
                return None
            return [t[self.LEMMA] for t in self.data]

        def entities(self):
            """Returns a list of named-entity-recognition tags of each token.
            Returns None if this annotation was not included.
            """
            if 'ner' not in self.annotators:
                return None
            return [t[self.NER] for t in self.data]

        def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
            """Returns a list of all ngrams from length 1 to n.
            Args:
                n: upper limit of ngram length
                uncased: lower cases text
                filter_fn: user function that takes in an ngram list and returns
                True or False to keep or not keep the ngram
                as_string: return the ngram as a string vs list
            """

            def _skip(gram):
                if not filter_fn:
                    return False
                return filter_fn(gram)

            words = self.words(uncased)
            ngrams = [(s, e + 1)
                    for s in range(len(words))
                    for e in range(s, min(s + n, len(words)))
                    if not _skip(words[s:e + 1])]

            # Concatenate into strings
            if as_strings:
                ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

            return ngrams

        def entity_groups(self):
            """Group consecutive entity tokens with the same NER tag."""
            entities = self.entities()
            if not entities:
                return None
            non_ent = self.opts.get('non_ent', 'O')
            groups = []
            idx = 0
            while idx < len(entities):
                ner_tag = entities[idx]
                # Check for entity tag
                if ner_tag != non_ent:
                    # Chomp the sequence
                    start = idx
                    while (idx < len(entities) and entities[idx] == ner_tag):
                        idx += 1
                    groups.append((self.slice(start, idx).untokenize(), ner_tag))
                else:
                    idx += 1
            return groups


    class Tokenizer(object):
        """Base tokenizer class.
        Tokenizers implement tokenize, which should return a Tokens class.
        """

        def tokenize(self, text):
            raise NotImplementedError

        def shutdown(self):
            pass

        def __del__(self):
            self.shutdown()


    class SimpleTokenizer(Tokenizer):
        ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
        NON_WS = r'[^\p{Z}\p{C}]'

        def __init__(self, **kwargs):
            """
            Args:
                annotators: None or empty set (only tokenizes).
            """
            self._regexp = regex.compile(
                '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
                flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
            )
            if len(kwargs.get('annotators', {})) > 0:
                logger.warning('%s only tokenizes! Skipping annotators: %s' %
                            (type(self).__name__, kwargs.get('annotators')))
            self.annotators = set()

        def tokenize(self, text):
            data = []
            matches = [m for m in self._regexp.finditer(text)]
            for i in range(len(matches)):
                # Get text
                token = matches[i].group()

                # Get whitespace
                span = matches[i].span()
                start_ws = span[0]
                if i + 1 < len(matches):
                    end_ws = matches[i + 1].span()[0]
                else:
                    end_ws = span[1]

                # Format data
                data.append((
                    token,
                    text[start_ws: end_ws],
                    span,
                ))
            return Tokens(data, self.annotators)

    tokenizer = SimpleTokenizer()
    text = unicodedata.normalize('NFD', text)
    if match_type == 'string':
        text = tokenizer.tokenize(text).words(uncased=True)
        for single_answer in answers:
            single_answer = unicodedata.normalize('NFD', single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i+ len(single_answer)]:
                    return 1
    return 0


def deal_prediction_with_evidence(pred):
    reject, answer, evidence = False, pred, pred
    if pred is None:
        return True, answer, evidence
    elif pred.lower().startswith("answer:"):
        pred = pred[7:]
    if has_answer(["no specific", "unconfirmed", "uncertain", "unavailable", "sorry", "not have access", "no information", 
                   "not provide", "no clear", "not mentioned", "not mention", "no mention", "no answer", "N/A", "unclear", 
                   "not clear", "unknown", "partially correct", "partially incorrect", "not correct", 
                   "cannot determine", "cannot answer", "not incorrect", "incomplete"], pred): # 放弃回答
        reject = True
    else:
        reject = False
    
    pred = re.sub(r'\n', " ", pred)
    answer, evidence = extract_ans_evi(pred)

    return reject, answer, evidence


def extract_ans_evi(text):
    text = text.replace("Answer: ","")
    evi_pattern = None
    ans_pattern = None
    
    if "Evidence: " in text:
        evi_pattern=r'(.*)Evidence: '
        ans_pattern=r'Evidence: (.*)'

    ans = re.sub(ans_pattern, "", text) if ans_pattern is not None else text
    # ans = re.sub("(.)$", "", ans)
    ans = get_answer_from_text(ans)

    evi = re.sub(evi_pattern, "", text) if evi_pattern is not None else text

    if ans and 'Passage-' in ans:
        pattern = re.compile(r'Passage-(.)')
        temp = re.sub(pattern, '', ans).strip()
        # if len(temp)<3:
        #     ans = get_answer_from_text(evi)

    evi = get_evidence_from_text(evi)

    return ans.strip(), evi.strip()


def get_evidence_from_text(sentence):
    evi = sentence
    pattern = re.compile(r'##(.*?)##')
    evi_find = re.findall(pattern, sentence)
    if len(evi_find):
        evi = evi_find[-1].strip()
    else:
        evi_cnt = sentence.count("Passage-")
        if evi_cnt>1:
            evi_lst = []
            start_index=0
            for i in range(evi_cnt):
                evi_index = sentence[start_index:].find("Passage-")
                if evi_index > -1 and evi_index+9 < len(sentence)-start_index:
                        evi_i = sentence[start_index+evi_index:start_index+evi_index+9]
                        evi_lst.append(evi_i)
                        start_index = start_index+evi_index+9
                else:
                    break
            evi_lst=list(set(evi_lst))
            evi = ', '.join(evi_lst)

        else:
            evi_index = sentence.find("Passage-")
            if evi_index > -1 and evi_index+9 < len(sentence):
                    evi = sentence[evi_index:evi_index+9]
    return evi


def get_answer_from_text(sentence):
    pattern = re.compile(r'##(.*?)##')

    ans = re.sub(pattern, '', sentence).strip()
    # if len(ans) < 3:
    #     ans = None
    return ans


def eval_em_f1_rl(predicted_answer, reference_answer):
    if predicted_answer is None:
        return 0, 0, 0
    return EM_compute(reference_answer, predicted_answer), F1_compute(reference_answer, predicted_answer), RougeL_compute(reference_answer, predicted_answer)


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def EM_compute(answer_list, prediction):
    return max([int(_normalize_answer(prediction) == _normalize_answer(ground_truth)) for ground_truth in answer_list])


def F1_compute(answers, pred):
    def get_tokens(s):
        if not s: return []
        return _normalize_answer(s).split()

    def compute_f1(a_gold, a_pred):
        gold_toks = get_tokens(a_gold)
        pred_toks = get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    return max([compute_f1(x, pred) for x in answers])


def RougeL_compute(answers, pred):
    def compute_RL(a_gold, a_pred):
        result = metric.compute(predictions=[a_pred], references=[a_gold], use_stemmer=True)
        return result['rougeL'].mid.recall
    return max(compute_RL(answer, pred) for answer in answers)


def str2paras(s):
        if s is None:
            return None
        paras = []
        for text in s.split('\n'):
            if text.strip() != '':
                paras.append(": " + text)
        return paras


def load_source(file):
    data = []
    f = open(file, 'r', encoding='utf-8')
    for line in f.readlines():
        data.append(json.loads(line))
    f.close()
    return data


def save_result(save_path, res):
    # save llm result
    # with open(args.output_dir + '/' + args.outfile, 'a+', encoding='utf-8') as fout:
    with open(save_path, 'a+', encoding='utf-8') as fout:
        fout.write(json.dumps(res) + "\n")


def deal_fusion(sentence):
    pattern = re.compile(r'$$(.+)$$')
    judgment_find = re.findall(pattern, sentence)
    if len(judgment_find):
        judgment = judgment_find[-1].strip()
    else:
        if "keep" in sentence:
            judgment = "keep"
        else:
            judgment = "discard"
    return judgment


def deal_fusion_probability(text):
    if text.lower().startswith("probability: "):
        text=text[13:]
    p_lst = re.findall("\d+\.\d+", text)
    if len(p_lst):
        confidence = float(p_lst[0])
    else:
        p_lst = re.findall("\d+\.", text)
        if len(p_lst):
            confidence = float(p_lst[0])
        else:
            confidence = 0.0
    return confidence





def match(res, text1_prediction, text2_reference, prompt_type):
    res[prompt_type]['EM'], res[prompt_type]['F1'], res[prompt_type]['RL'] = eval_em_f1_rl(text1_prediction, text2_reference)

    res[prompt_type]['has_answer'] = False
    if text1_prediction is not None:
        for ref in text2_reference:
            if text1_prediction.find(ref) > -1:
                res[prompt_type]['has_answer'] = True
                break

    if (res[prompt_type]['EM'] == 1 or res[prompt_type]['has_answer'] == True 
        or res[prompt_type]['F1'] > 0.7 or res[prompt_type]['RL'] > 0.7):
        return True
    else:
        return False
