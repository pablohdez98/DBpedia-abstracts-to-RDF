"""
Functions to extract triples from a set of sentences
Author: Fernando Casabán Blasco
"""

import spacy
from utils.preprocess_sentences import get_num_verbs
from utils.process_triples import fix_subj_complex_sentences
from spacy import displacy
from spacy.matcher import DependencyMatcher, Matcher


class Triple:
    def __init__(self, subj, pred, objct, sent):
        """list of tokens"""
        self.subj = subj
        self.pred = pred
        self.objct = objct
        self.sent = sent

    def get_copy(self):
        return Triple(self.subj.copy(), self.pred.copy(), self.objct.copy(), self.sent)

    def get_all_tokens(self):
        """
        Returs a list with all the tokens in the triple
        """
        return self.subj + self.pred + self.objct

    def set_rdf_triples(self, subj, pred, objct):
        self.subj_rdf = subj
        self.pred_rdf = pred
        self.objct_rdf = objct

    def get_rdf_triple(self):
        return f"{self.subj_rdf} | {self.pred_rdf} | {self.objct_rdf}"

    def __repr__(self):
        return f"{' '.join([x.text for x in self.subj])} | {' '.join([x.text for x in self.pred])} | {' '.join([x.text for x in self.objct])}"

    def __str__(self):
        return f"{' '.join([x.text for x in self.subj])} {' '.join([x.text for x in self.pred])} {' '.join([x.text for x in self.objct])}"

    
################################
# Triples extraction functions #
################################

def get_simple_triples(sentence):
    """
    Get the triples from each sentence in <subject, predicate, object> format.
    Firs identify the root verb of the dependency tree and explore each subtrees.
    If a subtree contains any kind of subject, all the subtree will be classified as subject, 
    the same happens with the objects.
    This function only works with simple sentences.
    """
    triples = []
    subjs = []
    objs = []
    preds = []
    root_token = sentence.root
    preds.append(root_token)
    for children in root_token.children:
        if(children.dep_ in ["aux","auxpass"]):
            # children.pos == AUX
            #preds.insert(0,children)
            preds.append(children)
        elif(children.dep_ == "neg"):
            #negative
            #preds.insert(1,children)
            preds.append(children)
        elif(children.dep_ == "xcomp"):
            # consider the prepositions between both verbs (was thought to result)
            xcomp_lefts = [tkn for tkn in children.lefts]
            preds.extend(xcomp_lefts)
            preds.append(children)
        elif children.dep_.find("mod"):
            # advmod
            pass
            #preds.append(children)
        
        preds.sort(key=lambda token: token.i)
        # retrieve subtrees
        is_subj = False
        is_obj = False
        temp_elem = []
        for token_children in children.subtree:
            if token_children in sentence:
                if token_children.dep_.find("subj") == True:
                    is_subj = True
                elif token_children.dep_.find("obj") == True:
                    is_obj = True
                elif token_children.dep_ == "attr":
                    is_obj = True
                if token_children not in preds:
                    temp_elem.append(token_children)
        if is_subj:
            subjs.append(temp_elem)
        elif is_obj:
            objs.append(temp_elem)
    # Build triples
    for s in subjs:
        for o in objs:
            triples.append(Triple(s,preds.copy(),o, sentence))
    return triples


def simplify_sentence(nlp, sentence):
    subsentences = []

    doc = nlp(sentence.text)

    patt_who_relcl_attr = [
        {
            "RIGHT_ID": "relcl",
            "RIGHT_ATTRS": {"DEP": "relcl"}
         },
        {
            "LEFT_ID": "relcl",
            "REL_OP": ">>",
            "RIGHT_ID": "who",
            "RIGHT_ATTRS": {"LOWER": {"IN": ["who", "which"]}}
         },
        {
            "LEFT_ID": "relcl",
            "REL_OP": "<<",
            "RIGHT_ID": "attr",
            "RIGHT_ATTRS": {"DEP": "attr"}
         },
        {
            "LEFT_ID": "attr",
            "REL_OP": ">>",
            "RIGHT_ID": "suj",
            "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}}
         }
    ]
    patt_who_relcl_obj = [
        {
            "RIGHT_ID": "relcl",
            "RIGHT_ATTRS": {"DEP": "relcl"}
        },
        {
            "LEFT_ID": "relcl",
            "REL_OP": ">>",
            "RIGHT_ID": "who",
            "RIGHT_ATTRS": {"LOWER": {"IN": ["who", "which"]}}
        },
        {
            "LEFT_ID": "relcl",
            "REL_OP": "<<",
            "RIGHT_ID": "obj",
            "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "pobj", "appos"]}}
        }
    ]
    patt_adv_relcl = [
        {
            "RIGHT_ID": "relcl",
            "RIGHT_ATTRS": {"DEP": "relcl"}
        },
        {
            "LEFT_ID": "relcl",
            "REL_OP": ">>",
            "RIGHT_ID": "adv",
            "RIGHT_ATTRS": {"LOWER": {"IN": ["who", "which", "where"]}}
        }
    ]

    matcher = DependencyMatcher(nlp.vocab)
    matcher.add("patt_adv_relcl", [patt_adv_relcl])
    matcher.add("patt_who_relcl_attr", [patt_who_relcl_attr])
    matcher.add("patt_who_relcl_obj", [patt_who_relcl_obj])

    matches = matcher(doc)

    if not matches:
        subsentences.append(sentence.text)
        return subsentences

    main_sentence = get_sentence_subtree_from_token(sentence.root, ["relcl"], nlp)
    new_sentence = []

    match_id, token_ids = matches[-1]
    string_id = nlp.vocab.strings[match_id]
    span = doc[token_ids[0]]
    second_sentence = get_sentence_subtree_from_token(span)
    if string_id == "patt_who_relcl_attr":
        subj = [tkn for tkn in main_sentence if "subj" in tkn.dep_]
        subj = get_sentence_subtree_from_token(subj[0])
        [new_sentence.append(subj) if tkn.i == doc[token_ids[1]].i else new_sentence.append(tkn) for tkn in second_sentence]
    elif string_id == "patt_who_relcl_obj":
        obj = doc[token_ids[2]]
        # obj = doc[token_ids[0]].head
        obj = get_sentence_subtree_from_token(obj, ["relcl"], nlp)
        [new_sentence.append(obj) if tkn.i == doc[token_ids[1]].i else new_sentence.append(tkn) for tkn in second_sentence]
    elif string_id == "patt_adv_relcl":
        adv = doc[token_ids[1]].text
        if adv in ["who", "which"]:
            subj = [tkn for tkn in main_sentence if "subj" in tkn.dep_]
            subj = get_sentence_subtree_from_token(subj[0])
            [new_sentence.append(subj) if tkn.i == doc[token_ids[1]].i else new_sentence.append(tkn) for tkn in second_sentence]
        elif adv in ["where"]:
            [new_sentence.append(tkn) for tkn in second_sentence if tkn.i != doc[token_ids[1]].i]  # remove 'where'
    second_sentence = ''.join([t.text_with_ws for t in new_sentence])
    subsentences.append(main_sentence.text)
    subsentences.append(second_sentence)
    return subsentences


def get_sentence_subtree_from_token(token, stop_condition=None, nlp=None):
    if stop_condition is None:
        stop_condition = []
    sent = []
    for child in token.subtree:
        if child.dep_ in stop_condition and child != token:
            continue
        ancestors = [t for t in child.ancestors if t in token.subtree]
        if any([t for t in ancestors if t.dep_ in stop_condition and t != token]):
            continue
        sent.append(child)
    sent.sort(key=lambda tkn: tkn.i)
    if len(sent) == (sent[-1].i+1 - sent[0].i):
        try:
            return sent[0].doc[sent[0].i: sent[-1].i + 1]
        except:
            return []
    if nlp:
        result = ''.join([tkn.text_with_ws for tkn in sent])
        return nlp(result)
    else:
        return []


def check_ascending_order_token_id(token_id):
    """
    Check whether positions of tokens matched are in ascending order. The first token must be a verb,
    so matches must be at right side of the verb, avoiding some token occurring before the verb
    """
    up = True
    for i in range(1, len(token_id)):
        if token_id[i] < token_id[i - 1]:
            up = False
            break
    return up


def get_all_triples(nlp, sentences, use_comp_sents=False):
    """ 
    Extract all the triples from the input list of sentences. Triples can be extracted from simple and complex senteces.
    Returns a list of objects of class Triple.
    """
    # displacy.serve(sentences, style='dep')
    import os
    str = "results2.txt"
    if os.path.exists(str):
        os.remove(str)
    else:
        print("The file does not exist")
    triples = []
    for sentence in sentences:
        sententes_list = simplify_sentence(nlp, sentence)
        with open('results2.txt', 'a') as f:
            f.write(f"COMPLEX {sentence}\n")
        f.close()
        for s in sententes_list:
            with open('results2.txt', 'a') as f:
                f.write("SIMPLE {}\n".format(s))
            get_simple_triples(nlp, s)
        # complex sentence
        '''if get_num_verbs(sentence) > 1:
            if use_comp_sents:
                simple_sentences = simplify_sentence(sentence)
                for sent in simple_sentences:
                    tps = get_simple_triples(nlp, sent)
                    triples.extend(tps)
        # simple sentence
        else:
            tps = get_simple_triples(nlp, sentence)
            triples.extend(tps)'''
    triples = fix_subj_complex_sentences(triples)
    return triples
