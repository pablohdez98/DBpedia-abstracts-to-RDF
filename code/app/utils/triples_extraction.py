"""
Functions to extract triples from a set of sentences
Author: Fernando Casabán Blasco and Pablo Hernández Carrascosa
"""
from spacy.matcher import DependencyMatcher, Matcher
from utils.log_generator import tracking_log


class Triple:
    def __init__(self, subj, pred, objct, sent):
        """list of tokens"""
        self.subj = subj
        self.pred = pred
        self.objct = objct
        self.sent = sent

    def get_copy(self):
        return Triple(self.subj.copy(), self.pred.copy(), self.objct.copy(), self.sent)

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
    """
    Get the triples from each sentence in <subject, predicate, object> format.
    First extract all complements of the verb that form the predicate of the triple.
    Then look for subject of the triplet with a matcher rule.
    Finally we get the object as the span from the last token of the predicate until the end of the sentence.
    With subject, predicate and object build an object Triple
    This function only works with simple sentences.
    """
    doc = nlp(sentence)

    ## PREDICATE
    patt_ROOT = [{"DEP": "ROOT"}]
    patt_ROOT_xcomp = [{"DEP": "ROOT", "POS": "VERB"}, {"DEP": "aux"}, {"DEP": "xcomp"}]

    matcher = Matcher(nlp.vocab)
    matcher.add("patt_ROOT", [patt_ROOT])
    matcher.add("patt_ROOT_xcomp", [patt_ROOT_xcomp])

    matches = matcher(doc)
    # Get only the last match (the longest one) and select the token representing the action
    preds = []
    if matches:
        (match_id, start, end) = matches[-1]
        string_id = nlp.vocab.strings[match_id]
        if string_id == "patt_ROOT":
            span = doc[start:end]
        if string_id == "patt_ROOT_xcomp":
            span = doc[start + 2:end]
        preds.append(span[0])
        pos_of_verb = span[0].i

    ## SUBJECT
    patt_SUBJS = [{"DEP": {"IN": ["nsubj", "nsubjpass"]}}]
    matcher = Matcher(nlp.vocab)
    matcher.add("patt_SUBJS", [patt_SUBJS])

    subjs = []
    matches = matcher(doc)
    for match_id, start, end in matches:
        span_subj = doc[start:end][0]  # The matched span
        subjs.append(get_sentence_subtree_from_token(span_subj, ["cc", "conj"]))
        rest_of_span = doc[end:pos_of_verb]
        conjs = [t for t in rest_of_span if t.dep_ == "conj"]
        for c in conjs:
            s = get_sentence_subtree_from_token(doc[c.i], ["cc", "conj"])
            subjs.append(s)

    ## OBJECT
    patt_attr = [{"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": "be"}},
                 {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "attr", "RIGHT_ATTRS": {"DEP": "attr"}}]
    patt_advmod_obj = [{"RIGHT_ID": "verb", "RIGHT_ATTRS": {"DEP": {"IN": ["ROOT", "xcomp"]}}},
                       {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "advmod",
                        "RIGHT_ATTRS": {"DEP": {"IN": ["advmod", "pobj", "dobj"]}}}]
    patt_prep_obj = [{"RIGHT_ID": "verb", "RIGHT_ATTRS": {"DEP": {"IN": ["ROOT", "xcomp"]}}},
                     {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "prep", "RIGHT_ATTRS": {"DEP": "prep"}},
                     {"LEFT_ID": "prep", "REL_OP": ">", "RIGHT_ID": "obj",
                      "RIGHT_ATTRS": {"DEP": {"IN": ["pobj", "dobj"]}}}]
    patt_be_acomp = [{"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": "be"}},
                     {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "acomp", "RIGHT_ATTRS": {"DEP": "acomp"}}]
    patt_agent_obj = [{"RIGHT_ID": "verb", "RIGHT_ATTRS": {"DEP": "ROOT"}},
                      {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "agent", "RIGHT_ATTRS": {"DEP": "agent"}},
                      {"LEFT_ID": "agent", "REL_OP": ">", "RIGHT_ID": "obj",
                       "RIGHT_ATTRS": {"DEP": {"IN": ["pobj", "dobj"]}}}]
    patt_verb_conj = [{"RIGHT_ID": "verb", "RIGHT_ATTRS": {"DEP": "ROOT"}},
                      {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "conj", "RIGHT_ATTRS": {"DEP": "conj"}}]

    dep_matcher = DependencyMatcher(nlp.vocab)
    dep_matcher.add("patt_attr", [patt_attr])
    dep_matcher.add("patt_advmod_obj", [patt_advmod_obj])
    dep_matcher.add("patt_prep_obj", [patt_prep_obj])
    dep_matcher.add("patt_be_acomp", [patt_be_acomp])
    dep_matcher.add("patt_agent_obj", [patt_agent_obj])
    dep_matcher.add("patt_verb_conj", [patt_verb_conj])
    dep_matches = dep_matcher(doc)

    objs = []
    if not dep_matches:
        return []
    for match_id, token_id in dep_matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        if not check_ascending_order_token_id(token_id):  # verify tokens matched are at right side from verb
            continue

        if string_id in ["patt_attr", "patt_advmod_obj"]:
            conjs = doc[token_id[1]].conjuncts  # coordinated tokens, not including the token itself
            if conjs:
                objs.append(get_sentence_subtree_from_token(doc[token_id[1]], ["cc", "conj"]))
                for c in conjs:
                    objs.append(get_sentence_subtree_from_token(doc[c.i], ["cc", "conj"]))
            else:
                objs.append(get_sentence_subtree_from_token(doc[token_id[1]]))

        if string_id == "patt_be_acomp":
            conjs = doc[token_id[1]].conjuncts  # coordinated tokens, not including the token itself
            if conjs:
                objs.append(get_sentence_subtree_from_token(doc[token_id[1]], ["cc", "conj"]))
                for c in conjs:
                    objs.append(get_sentence_subtree_from_token(doc[c.i], ["cc", "conj"]))
            else:
                objs.append(get_sentence_subtree_from_token(doc[token_id[1]]))

        if string_id == "patt_prep_obj":
            conjs = doc[token_id[1]].conjuncts  # coordinated tokens with preposition (not including token itself)
            if conjs:
                # several prep + objects
                prep_object = get_sentence_subtree_from_token(doc[token_id[1]], ["cc", "conj"])  # get first (prep + object)
                simpler_objs = split_conjunctions_obj(prep_object, doc[token_id[1]], doc[token_id[2]])  # coordinated tokens with the object
                objs.extend(simpler_objs)
                for c in conjs:
                    prep_object = get_sentence_subtree_from_token(doc[c.i], ["cc", "conj"])  # get next (prep + object)
                    inside_object = [tk for tk in prep_object if (tk.dep_.find("obj") != -1)]
                    simpler_objs = split_conjunctions_obj(prep_object, doc[c.i], inside_object[-1])  # coordinated tokens with the object
                    objs.extend(simpler_objs)
            else:
                prep_object = get_sentence_subtree_from_token(doc[token_id[1]])  # there is only one object
                simpler_objs = split_conjunctions_obj(prep_object, doc[token_id[1]], doc[token_id[2]])  # coordinated tokens within the object
                objs.extend(simpler_objs)

        if string_id == "patt_agent_obj":
            conjs = doc[token_id[1]].conjuncts  # coordinated tokens with agent (not including token itself)
            if conjs:
                # several agent + object
                agent_object = get_sentence_subtree_from_token(doc[token_id[1]], ["cc", "conj"])  # get first (agent + object)
                simpler_objs = split_conjunctions_obj(agent_object, doc[token_id[1]], doc[token_id[2]])  # coordinated tokens with the object
                objs.extend(simpler_objs)
                for c in conjs:
                    agent_object = get_sentence_subtree_from_token(doc[c.i], ["cc", "conj"])  # get next (agent + object)
                    inside_object = [tk for tk in agent_object if (tk.dep_.find("obj") != -1)]
                    simpler_objs = split_conjunctions_obj(agent_object, doc[c.i], inside_object[-1])  # coordinated tokens with the object
                    objs.extend(simpler_objs)
            else:
                agent_object = get_sentence_subtree_from_token(doc[token_id[1]])  # there is only one object
                simpler_objs = split_conjunctions_obj(agent_object, doc[token_id[1]], doc[token_id[2]])  # coordinated tokens within the object
                objs.extend(simpler_objs)

        if string_id == "patt_verb_conj":
            if doc[token_id[1]].pos_ == "VERB":
                continue
            conjs = doc[token_id[0]].conjuncts  # coordinated tokens, not including the token itself
            for c in conjs:
                objs.append(get_sentence_subtree_from_token(doc[c.i], ["cc", "conj"]))

    # Build triples
    triples = []
    for s in subjs:
        for o in objs:
            if o[0].pos_ == "ADP":
                pred_prep = [preds[0], o[0]]
                o = o[1:]
            else:
                pred_prep = preds

            if type(o) == list:
                o = o[0]

            subject_token_list = [token for token in s]
            object_token_list = [token for token in o]

            new_triple = Triple(subject_token_list, pred_prep, object_token_list, sentence)
            triples.append(new_triple)
    return triples


def simplify_sentence(nlp, sentence):
    subsentences = []

    doc = nlp(sentence.text)

    patt_relcl_attr = [
        {
            "RIGHT_ID": "relcl",
            "RIGHT_ATTRS": {"DEP": "relcl"}
         },
        {
            "LEFT_ID": "relcl",
            "REL_OP": ">>",
            "RIGHT_ID": "pron",
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
    patt_relcl_obj = [
        {
            "RIGHT_ID": "relcl",
            "RIGHT_ATTRS": {"DEP": "relcl"}
        },
        {
            "LEFT_ID": "relcl",
            "REL_OP": ">>",
            "RIGHT_ID": "pron",
            "RIGHT_ATTRS": {"LOWER": {"IN": ["who", "which"]}}
        },
        {
            "LEFT_ID": "relcl",
            "REL_OP": "<<",
            "RIGHT_ID": "obj",
            "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "pobj", "appos"]}}
        }
    ]
    patt_generic_relcl = [
        {
            "RIGHT_ID": "relcl",
            "RIGHT_ATTRS": {"DEP": "relcl"}
        },
        {
            "LEFT_ID": "relcl",
            "REL_OP": ">>",
            "RIGHT_ID": "pron",
            "RIGHT_ATTRS": {"LOWER": {"IN": ["who", "which", "where"]}}
        }
    ]

    matcher = DependencyMatcher(nlp.vocab)
    matcher.add("patt_generic_relcl", [patt_generic_relcl])
    matcher.add("patt_relcl_attr", [patt_relcl_attr])
    matcher.add("patt_relcl_obj", [patt_relcl_obj])

    matches = matcher(doc)

    if not matches:
        subsentences.append(sentence.text)
        return subsentences

    match_id, token_ids = matches[-1]
    string_id = nlp.vocab.strings[match_id]
    span = doc[token_ids[0]]

    if sentence[token_ids[1]-1].dep_ == 'punct':
        new_sentence = ' '.join([sentence[0:token_ids[1]-1].text, sentence[token_ids[1]:-1].text])
        sentence = nlp(new_sentence)[0:-1]
    main_sentence = get_sentence_subtree_from_token(sentence.root, ["relcl"], nlp)
    second_sentence = get_sentence_subtree_from_token(span)

    new_sentence = []
    if string_id == "patt_relcl_attr":
        subj = [tkn for tkn in main_sentence if "subj" in tkn.dep_]
        subj = get_sentence_subtree_from_token(subj[0])
        [new_sentence.append(subj) if tkn.i == doc[token_ids[1]].i else new_sentence.append(tkn) for tkn in second_sentence]
    elif string_id == "patt_relcl_obj":
        obj = doc[token_ids[2]]
        # obj = doc[token_ids[0]].head
        obj = get_sentence_subtree_from_token(obj, ["relcl"], nlp)
        [new_sentence.append(obj) if tkn.i == doc[token_ids[1]].i else new_sentence.append(tkn) for tkn in second_sentence]
    elif string_id == "patt_generic_relcl":
        pron = doc[token_ids[1]].text
        if pron in ["who", "which"]:
            subj = [tkn for tkn in main_sentence if "subj" in tkn.dep_]
            subj = get_sentence_subtree_from_token(subj[0])
            [new_sentence.append(subj) if tkn.i == doc[token_ids[1]].i else new_sentence.append(tkn) for tkn in second_sentence]
        elif pron in ["where"]:
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


def split_conjunctions_obj(span, prep, object_token):
    """
    Search for conjunctions in the object and splits in simpler objects, here is an example:
    Original: in China, India, the Muslim world, and Europe
    Result: in China | in India | in the Muslim world | in Europe
    Returns a list of objects
    """
    new_objects = []

    conjunctions = object_token.conjuncts
    if conjunctions:
        # Build the first object
        new_objects.append([prep, get_sentence_subtree_from_token(object_token, ["cc", "conj"])])
        for conjunction in conjunctions:
            new_objects.append([prep, get_sentence_subtree_from_token(conjunction, ["cc", "conj"])])
    else:
        # No conjunction tokens at the object part of the triple
        new_objects.append(span)
    return new_objects


def get_all_triples(nlp, sentences):
    """ 
    Extract all the triples from the input list of sentences. Triples can be extracted from simple and complex senteces.
    Returns a list of objects of class Triple.
    """
    triples = []
    simple_sentences_tracking = []  # tracking
    for sentence in sentences:
        sententes_list = simplify_sentence(nlp, sentence)
        [simple_sentences_tracking.append(simple) for simple in sententes_list]  # tracking
        for s in sententes_list:
            tps = get_simple_triples(nlp, s)
            triples.extend(tps)
    tracking_log(simple_sentences_tracking, level=3)
    return triples
