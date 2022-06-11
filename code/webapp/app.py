import base64
import glob
import json
import os
import time

import pandas as pd
import spacy
import coreferee
import streamlit as st
import utils.update_ontology as uo
import validators
from rdflib.term import URIRef

import utils.build_RDF_triples as brt
import utils.lookup_tables_services as lts
import utils.preprocess_sentences as pps
import utils.process_triples as pt
import utils.triples_extraction as te

timestr = time.strftime("%Y%m%d-%H%M%S")
SPOTLIGHT_ONLINE_API = "https://api.dbpedia-spotlight.org/en/annotate"
PROP_LEXICALIZATION_TABLE = "datasets/verb_prep_property_lookup.json"
CLA_LEXICALIZATION_TABLE = "datasets/classes_lookup.json"


def pipeline(nlp, raw_text, dbo_graph, prop_lex_table, cla_lex_table):
    raw_text = pps.clean_text(raw_text)
    doc = nlp(raw_text)

    # correferences resolution
    if doc._.coref_chains:
        rules_analyzer = nlp.get_pipe('coreferee').annotator.rules_analyzer
        interchange_tokens_pos = []  # list of tuples (pos.i, mention.text)
        interchangeable_subjects = ["he", "she", "it", "they"]
        for token in doc:
            if token.text.lower() in interchangeable_subjects and bool(doc._.coref_chains.resolve(token)):
                # there is a coreference
                mention_head = doc._.coref_chains.resolve(token)  # get the mention
                full_mention = rules_analyzer.get_propn_subtree(doc[mention_head[0].i])  # get the complex proper noun
                mention_text = ''.join([token.text_with_ws for token in full_mention])
                interchange_tokens_pos.append((token.i, mention_text))

        if interchange_tokens_pos:
            resultado = ''
            pointer = 0
            for tupla in interchange_tokens_pos:
                resultado = resultado + doc[pointer:tupla[0]].text_with_ws + tupla[1]
                pointer = tupla[0] + 1
            resultado = resultado + doc[pointer:].text_with_ws

            doc = nlp(resultado)

    sentences = pps.get_sentences(doc)
    triples = te.get_all_triples(nlp, sentences)
    triples = pt.split_amod_conjunctions_subj(nlp, triples)
    triples = pt.split_amod_conjunctions_obj(nlp, triples)

    try:
        term_URI_dict, term_types_dict = brt.get_annotated_text_dict(raw_text, service_url=SPOTLIGHT_ONLINE_API)
    except:
        return [], []

    rdf_triples = brt.replace_text_URI(triples, term_URI_dict, term_types_dict, prop_lex_table, cla_lex_table, dbo_graph)
    return triples, rdf_triples


def print_debug(triples):
    """ Print the final result: Original sentence, text triples and rdf triples"""
    if triples:
        for t in triples:
            st.write(t.sent + '\n')
            st.write('--> ' + t.__repr__() + '\n')
            st.write('--> ' + t.get_rdf_triple() + '\n')
            st.write('----')


def get_only_triples_URIs(rdf_triples):
    """ Keep just the triples that are entirely made of URIRef. """
    return [t for t in rdf_triples if isinstance(t.pred_rdf, URIRef) and isinstance(t.objct_rdf, URIRef)]


@st.cache(allow_output_mutation=True)
def init():
    """ Function to load all the external components. """
    # Load dependency model and add coreferee support
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe('coreferee')
    # Load datastructures
    prop_lex_table = lts.load_lexicalization_table(PROP_LEXICALIZATION_TABLE)
    cla_lex_table = lts.load_lexicalization_table(CLA_LEXICALIZATION_TABLE)
    dbo_graph = brt.load_dbo_graph(DBPEDIA_ONTOLOGY)

    return nlp, prop_lex_table, cla_lex_table, dbo_graph


def clear_form():
    st.session_state["inputtext"] = ""


if __name__ == "__main__":
    st.set_page_config(page_title='Text to RDF', layout='wide')
    local_ontology_path = 'datasets/'
    local_ontology_files = glob.glob(f'{local_ontology_path}*.owl')
    names = [os.path.basename(x) for x in local_ontology_files]
    namesSorted = sorted(names, reverse=True)
    DBPEDIA_ONTOLOGY = local_ontology_path + namesSorted[0]

    nlp, prop_lex_table, cla_lex_table, dbo_graph = init()

    # Create a page dropdown list at sidebar
    page = st.sidebar.selectbox("Choose your task", ["Text to RDF", "Update look up tables"])
    if page == "Text to RDF" or not page:

        st.header('DBpedia abstracts to RDF')
        st.write('This app translates any kind of text into RDF!')

        with st.form('my_form'):
            st.subheader('User input parameters')
            raw_text = st.text_area('Insert abstract or any other kind of text', key='inputtext', height=200,
                                    help='Insert abstract or any other kind of text in order to generate RDF based on the input')
            show_text_triples = st.checkbox('Show text triples',
                                            help='Print the text tripels from the sentences. This is useful to understand how the pipeline processed the input')
            print_debg = st.checkbox('Print debug information',
                                    help='Print the text triples and RDF triples extracted for every sentence')

            col1, col2, _, _, _, _, _, _, _, _, _, _ = st.columns(12)
            submitted = col1.form_submit_button("Submit")
            col2.form_submit_button("Clear text", on_click=clear_form)

        if submitted:
            # primitive user input check
            if len(raw_text) < 20:
                st.write("Invalid input text, try something bigger")
            else:
                with st.spinner('Processing text...'):
                    text_triples, rdf_triples = pipeline(nlp, raw_text, dbo_graph, prop_lex_table, cla_lex_table)

                if not rdf_triples:
                    st.warning('No RDF triples where obtained')
                    st.stop()
                st.success('Done!')

                # save triples in a graph and returns the graph serialized (ttl format)
                graph, graph_serialized = brt.build_result_graph(rdf_triples)

                # RDF triples
                st.write('----')
                st.subheader("RDF triples:")
                # read triples from rdf graph
                for s, p, o in graph.triples((None, None, None)):
                    st.write(s, ' | ', p, ' | ', o)

                # download_ttl(graph_serialized)
                st.download_button(label='Download ".ttl" file', data=graph_serialized, file_name='graph.ttl',
                                   mime='file/ttl')

                # Text triples
                if show_text_triples:
                    st.write('----')
                    st.subheader("Text triples:")
                    for t in text_triples:
                        st.write(t.__repr__())

                # Debug info
                if print_debg:
                    st.write('----')
                    st.subheader("Debug:")
                    print_debug(rdf_triples)

    elif page == "Update look up tables":
        st.header('Look up tables')
        update_ontology = st.sidebar.button('Update DBpedia ontology',
                                            help='Download latest DBpedia ontology in format owl')

        if "class_tbl" not in st.session_state:
            df_classes = pd.DataFrame.from_dict(cla_lex_table, orient="index", columns=["URI"])
            st.session_state.class_tbl = df_classes.sort_index()
        if "prop_tbl" not in st.session_state:
            sorted_json = json.dumps(prop_lex_table, sort_keys=True)  # sort in ascending order
            st.session_state.prop_tbl = sorted_json

        col1, col2 = st.columns(2)
        with col1:
            st1 = st.expander("Class table").empty()
            st1.dataframe(st.session_state.class_tbl)
            with st.form("insert_class_form"):
                st.subheader('New entry - Classes Table')
                class_name = st.text_input('Enter class name:', key="name")
                class_URI = st.text_input('Enter URI:', key="URI")
                if submitted_insert_class_form := st.form_submit_button("Submit"):
                    if validators.url(class_URI):
                        class_name = class_name.lower()
                        if res := lts.update_classes_lookup(class_name, class_URI, cla_lex_table, CLA_LEXICALIZATION_TABLE):
                            st.success(f'Inserted "{class_name}" with URI {class_URI}')
                            st.session_state.class_tbl.loc[class_name] = class_URI  # adding a row
                            st.session_state.class_tbl = st.session_state.class_tbl.sort_index()
                            st1.dataframe(st.session_state.class_tbl)  # show current table content
                    else:
                        st.error(f'The URI {class_URI} is not valid')

        with col2:
            st2 = st.expander("Properties table").empty()
            st2.json(st.session_state.prop_tbl, expanded=False)
            with st.form("insert_prop_form"):
                st.subheader('New entry - Verbs Table')
                verb = st.text_input('Enter verb (in infinitive form):', key="verb")
                prep = st.text_input('Enter preposition (if necessary):', key="prep")
                pURI = st.text_input('Enter URI:', key="pURI")
                if submitted_insert_prop_form := st.form_submit_button("Submit"):
                    if validators.url(pURI):
                        verb = verb.lower()
                        prep = prep.lower()
                        res, prop_lex_table = lts.update_verb_prep_property_lookup(verb, prep, pURI, prop_lex_table, PROP_LEXICALIZATION_TABLE)
                        if res:
                            st.success(f'Inserted "{verb} {prep}" with URI {pURI}')
                            sorted_json = json.dumps(prop_lex_table, sort_keys=True)  # update session_state table
                            st.session_state.prop_tbl = sorted_json
                            st2.json(st.session_state.prop_tbl, expanded=False)  # show current table content
                    else:
                        st.error(f'The URI {pURI} is not valid')

        if update_ontology:
            with st.spinner('This task can take a while. Please wait...'):
                download_state = uo.update_ontology_file()
            if not download_state[0]:
                st.sidebar.error(download_state[1])
            else:
                st.sidebar.success(download_state[1])
                st.sidebar.success('Please, re-run app to load changes')
