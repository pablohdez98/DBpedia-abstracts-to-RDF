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

def get_simple_triples(nlp, sentence):
    doc = nlp(sentence)

    ## PREDICATE
    # The verb forms analyzed are:
    # aux (be)
    # aux (did, will, have, be) verb
    # aux (have) aux (be) verb
    # verb to xcomp
    # In the first three forms, the verb is the root of the sentence and represents the action
    # In the last form, there are 2 verbs. We consider that xcomp represents the action (if there is
    # a preposition it affects the xcomp)

    patt_ROOT = [{"DEP": "ROOT"}]
    patt_ROOT_xcomp = [{"DEP": "ROOT", "POS": "VERB"}, {"DEP": "aux"}, {"DEP": "xcomp"}]

    matcher = Matcher(nlp.vocab)
    matcher.add("patt_ROOT", [patt_ROOT])
    matcher.add("patt_ROOT_xcomp", [patt_ROOT_xcomp])

    matches = matcher(doc)
    # Get only the last match (the longest one) and select the token representing the action
    preds = None
    if matches:
        (match_id, start, end) = matches[-1]
        string_id = nlp.vocab.strings[match_id]
        if string_id == "patt_ROOT":
            span = doc[start:end]
        elif string_id == "patt_ROOT_xcomp":
            span = doc[start + 2:end]
        preds = span[0].lemma_  # save the infinitive form
        preds = nlp(preds)
        pos_of_verb = span[0].i  # save the token position

    ## SUBJECT
    # The subject matches tokens with dep = "nsubj" or "subjpass".
    # When a subject is located, other subjects joined with conjunctions are searched for.
    # To prevent the search for conjunctions outside the limits of the subject, the search
    # is limited between the located subject and the verb of the predicate (pos_of_verb)

    patt_SUBJS = [{"DEP": {"IN": ["nsubj", "nsubjpass"]}}]
    matcher = Matcher(nlp.vocab)
    matcher.add("patt_SUBJS", [patt_SUBJS])

    subjs = []
    matches = matcher(doc)
    for match_id, start, end in matches:
        #        span_subj = doc[start:end][0]   # The matched span
        #        span = get_sentence_subtree_from_token(span_subj)
        #        subjs.append(span)

        span_subj = doc[start:end][0]  # The matched span
        subjs.append(get_sentence_subtree_from_token(span_subj, ["cc", "conj"], nlp))
        rest_of_span = doc[end:pos_of_verb]
        conjs = [t for t in rest_of_span if t.dep_ == "conj"]
        for c in conjs:
            s = get_sentence_subtree_from_token(doc[c.i], ["cc", "conj"], nlp)
            subjs.append(s)

    ## OBJECT
    # Here we use dependency rules, since we cannot guarantee that tokens included in the pattern
    # are correlative in the sentence

    patt_attr = [
        {
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"LEMMA": "be"}
        },
        {
            "LEFT_ID": "verb",
            "REL_OP": ">", "RIGHT_ID": "attr",
            "RIGHT_ATTRS": {"DEP": "attr"}
        }
    ]
    patt_advmod_obj = [
        {
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"DEP": {"IN": ["ROOT", "xcomp"]}}
        },
        {
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "advmod",
            "RIGHT_ATTRS": {"DEP": {"IN": ["advmod", "pobj", "dobj"]}}
        }
    ]
    patt_prep_obj = [
        {
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"DEP": {"IN": ["ROOT", "xcomp"]}}
        },
        {
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "prep",
            "RIGHT_ATTRS": {"DEP": "prep"}
        },
        {
            "LEFT_ID": "prep",
            "REL_OP": ">", "RIGHT_ID": "pobj",
            "RIGHT_ATTRS": {"DEP": {"IN": ["pobj", "dobj"]}}
        }
    ]

    dep_matcher = DependencyMatcher(nlp.vocab)
    dep_matcher.add("patt_attr", [patt_attr])
    dep_matcher.add("patt_advmod_obj", [patt_advmod_obj])
    dep_matcher.add("patt_prep_obj", [patt_prep_obj])
    dep_matches = dep_matcher(doc)

    objs = []
    # Iterate over the matches and print the span text
    for match_id, token_id in dep_matches:
        if not check_ascending_order_token_id(token_id):  # verify tokens matched are at right side from verb
            continue
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        if string_id in ["patt_attr", "patt_advmod_obj", "patt_prep_obj"]:
            span = get_sentence_subtree_from_token(doc[token_id[1]])
            objs.append(get_sentence_subtree_from_token(doc[token_id[1]], ["cc", "conj"], nlp))
            conjs = [t for t in span if (t.dep_ == "conj") and (t.i > token_id[1])]
            for c in conjs:
                objs.append(get_sentence_subtree_from_token(doc[c.i], ["cc", "conj"], nlp))

    # Build triples
    triples = []
    with open('results2.txt', 'a') as f:
        for s in subjs:
            for o in objs:
                new_triple = Triple(s, preds, o, sentence)
                f.write(f"--> {s} | {preds} | {o}\n")
                triples.append(new_triple)
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
