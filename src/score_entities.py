import os
import re
from collections import namedtuple, defaultdict
from lexicons import load_connotation_frames, load_power_split, load_agency_split, load_hannah_split
from xml_helpers import process_xml_text
from weighted_tests import logistic_regression, find_logistic_regression_weights
import h5py
import ast
from tqdm import tqdm
import numpy as np
import pickle
import stanza
import xml.etree.ElementTree as ET
import argparse

# Custom types to handle embeddings and articles
# TODO: Decide what information we really need. Putting
# info specific to our code will be a pain in the ass
# later on (if we try to use an external corpus)
# NOTE: Assumes that the article_idx are unique over
# a corpus.. Check this is a valid assumption
#Article = namedtuple("Article", "riot src article_id")
#EmbeddingTuple = namedtuple("Embedding", "article sent_idx verb_idx word embedding")
CONNO_DIR="../frames/annotated_connotation_frames/"
POWER_AGENCY_DIR="../frames/agency_power.csv"

government_kw = {
    'government', 'government forces', 'state', 'authority', 'authorities', 'regime',
    'prime minister', 'president', 'police', 'police forces', 'army', 'armed forces', 
    'premier', 'officers', 'officer', 'military', 'troops', 'security forces', 'forces',
    'nicolas', 'maduro', 'hugo', 'chavez', 'diosdado', 'cabello',
    'emmanuel', 'macron', 'elysee', 'christophe', 'castaner',  'philippe', 'edouard',
    'carrie', 'lam', 'xi', 'jinping', 'beijing', 'chief executive',
    'irgc', 'ali', 'khamenei', 'ayatollah', 'fadavi', 'hassan', 'rouhani', 'zarif'
}
protesters_kw = {
    'protest', 'protester', 'protesters', 'demonstrator', 'demonstrators', 'rioter', 'rioters',
    'striker', 'agitator', 'dissenter', 'disrupter', 'revolter', 'marcher', 'dissident',
    'militant', 'activists', 'activist',
    'venezuelans', 'yellow vests', 'yellow vest', 'jackets',
    'juan', 'guaido', 'leopoldo', 'lopez',
    'maxime', 'nicolle', 'eric', 'drouet', 'christophe', 'chalençon', 'priscillia', 'ludosky',
    'jacline', 'mouraud', 'jerome', 'rodrigues', 'etienne', 'chouard', 'francois', 'boulo',
    'joshua', 'wong', 'chan', 'ho-tin', 'kongers',
    'mir-hossein', 'mousavi',
}

PRONOUNS = {"i", "me", "my", "mine", "myself",
          "you", "your", "yours", "yourself", "yourselves",
          "he", "him", "his", "himself",
          "she", "her", "hers", "herself",
          "it", "its", "itself",
          "we", "us", "our", "ours", "ourselves",
          "they", "them", "their", "theirs", "themselves"}


# Lemmatize the kew-words for government and protesters.
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma')
def to_lemma(kw_set):
    lemmas_set = set()
    for kw in kw_set:
        doc = nlp(kw)
        # Discard multi-word key-words.
        if len(doc.sentences[0].words) == 1:
            lemmas_set.add(doc.sentences[0].words[0].lemma)
    return lemmas_set

government_kw_lemmas = to_lemma(government_kw)   
protesters_kw_lemmas = to_lemma(protesters_kw)

def articleToName(article, append_str = ""):
    """ Helper function that converts the 
    article tuple into a str (e.g. filename)"""
    return "_".join(article) + append_str

# TODO: Refactor these functions...
def loadSentimentType(header):
    def loadSentiment():
        return load_hannah_split(CONNO_DIR, header, binarize=True, remove_neutral=False)
    return loadSentiment

def loadPower():
    return load_power_split(POWER_AGENCY_DIR)

def loadAgency():
    return load_agency_split(POWER_AGENCY_DIR)

# TODO: Rename me plz :(
OPERATIONS=[("Perspective(wo)", loadSentimentType("Perspective(wo)")), ("Perspective(ws)", loadSentimentType("Perspective(ws)")), ("Power", loadPower), ("Agency", loadAgency)]

# Class that houses embeddings and manage operations over them 
class EmbeddingManager:
    def __init__(self):
        self.tuples = []
        self.word_to_tuple = defaultdict(list)
        self.article_to_tuple = {}
   
    def fetchArticles(self, search_criterion):
        """ Returns tuples using a search tuple of the same type, with wildcard fields marked as None or '' """
        def match(t1, search_tupl):
            return all([not search_term or term == search_term for term,search_term in zip(t1,search_tupl)])
        return [tupl for tupl in self.tuples if match(tupl['article'], search_criterion)]

    def __getitem__(self, item):
        if type(item) == str:
            return [self.tuples[idx] for idx in self.word_to_tuple.get(item, [])]
        elif type(item) == tuple:
            return self.tuples[self.article_to_tuple[item]]
        else:
            raise "Invalid data type"
    
    def addItem(self, tupl):
        self.tuples.append(tupl)
        self.word_to_tuple[tupl['verb_lemma']] += [len(self.tuples) - 1]
        self.article_to_tuple[tupl['article']] = len(self.tuples) - 1

    def decontextualizeWord(self, word):
        word_embs = np.stack([self.tuples[idx]['embedding'] for idx in self.word_to_tuple[word]])
        return np.mean(word_embs, axis=0)

def getArticleList(dir_path, split_str="[_\.]"):
    """ Function that loads all the files in a directory,
    verifies their titles are correctly formatted,
    returns them in a named tuple """

    article_list = []
    split_regex = re.compile(split_str)
    
    for file_name in os.listdir(dir_path):
        if not file_name.endswith('hdf5'):
            continue
        split_str = split_regex.split(file_name)
        if len(split_str) < 4:
            print("The name contains {} splits".format(len(split_str)))
            continue
        current_article = tuple(split_str[:3])
        article_list.append(current_article)
    return article_list

def extractItem(head, find_nodes, iter_node = None):
    """ A generic way of iterating through XMLs 
    and returning a list of iter_type"""
    data = []
    final_node = head
    for node in find_nodes:
        final_node = final_node.find(node)

    if iter_node:
        data = final_node.iter(iter_node)
    else:
        data = final_node

    return data

def filterSentence(args, article, sent_idx, dependencies, lemmas, pos, doc_coreference_dict=None):
    # Go over dependencies:
    # if active:                    |    if passive:
    #   if type == nsubj, nobj      |        if type == nsubpass, agent
    # check if lemma of the word is in the lemmas of protests or government forces
    # IF coref included:
    #       pass a coref structure that associates if a pronoun is government/protest related
    tuples = []
    verb_dependencies = defaultdict(list)
    for dep in dependencies:
        # Check if we are looking at a subject or an object:
        if dep.get("type") in {"nsubj", "dobj", "nsubjpass", "agent", "nmod"}:
            dependent = dep.find("dependent")
            dependent_idx = int(dependent.get("idx")) - 1
            dependent_lemma = lemmas[dependent_idx].lower()

            governor = dep.find("governor") # verb
            governor_idx = int(governor.get("idx")) - 1
            governor_lemma = lemmas[governor_idx].lower()
            governor_pos = pos[governor_idx]

            # In case nmod was not pointing to a verb.
            if not pos[governor_idx].startswith("VB"):
                continue
            # Representative is not None if this was a coreference.
            representative = None
            # Check if the dependent is in the government list or the protesters list:
            entity_group = None
            if dependent_lemma in protesters_kw:
                entity_group = "protester"
            elif dependent_lemma in government_kw:
                entity_group = "government"
            elif doc_coreference_dict is not None and (sent_idx, dependent_idx) in doc_coreference_dict:
                entity_group, representative = doc_coreference_dict[(sent_idx, dependent_idx)]
                # print("FOUND AN EXAMPLE")
                # print("PRONOUN", sent_idx, dependent_idx)
                # print(lemmas)
                # print(entity_group)
                # print(lemmas[dependent_idx])
            example = {
                "dep_type": dep.get("type"),
                "entity_lemma": dependent_lemma,
                "entity_idx": dependent_idx,
                "entity_group": entity_group
            }
            if representative is not None:
                example["representative"] = representative

            verb_dependencies[(governor_idx, governor_lemma)].append(example)

    # In case we missed verbs add them back to the verb_dependencies structure without any dependents.
    if args.use_all_verbs:
        for verb_idx, (verb_lemma, pos_tag) in enumerate(zip(lemmas, pos)):
            if pos_tag.startswith("VB") and (verb_idx, verb_lemma) not in verb_dependencies:
                verb_dependencies[(verb_idx, verb_lemma)] = []

    # Iterate over verbs in the sentences filter out wrong nmod examples build single verb dictionary.
    for governor_idx, governor_lemma in verb_dependencies:
        verb_example_dict = {
            "article": article,
            "sent_idx": sent_idx,
            "verb_idx": governor_idx,
            "verb_lemma": governor_lemma
        }
        dependencies = verb_dependencies[governor_idx, governor_lemma]
        if len(dependencies) > 0:
            dep_type_set = {dependency['dep_type'] for dependency in dependencies}
            for dependency in dependencies:
                # agent
                if dependency['dep_type'] in ["nsubj", "agent", "nmod"]:
                    # We keep example with type nmod only in passive sentences that have a nsubjpass for the verb.
                    if dependency['dep_type'] == "nmod" and not "nsubjpass" in dep_type_set:
                        continue
                    verb_example_dict["agent"] = dependency
                # patient
                elif dependency['dep_type'] in ["nsubjpass", "dobj"]:
                    verb_example_dict["patient"] = dependency
        tuples.append(verb_example_dict)
    return tuples

def build_coreference_dict(doc_coreference):
    doc_coreference_dict = dict()
    for coref in doc_coreference:
        mentions = [mention for mention in coref.iter("mention")]
        pronoun_mentions = []
        entity_group = None
        representative = None
        for mention in mentions:
            text_head_idx = int(mention.find("head").text) - int(mention.find("start").text)
            text_head = mention.find("text").text.lower().split()[text_head_idx]
            if mention.get("representative"):
                representative = text_head
            # Check if the head of the mention belongs to a keyword group.
            if text_head in government_kw:
                entity_group = "government"
            elif text_head in protesters_kw:
                entity_group = "protester"
            # Check if the mention is a pronoun (only case we really care about).
            if text_head in PRONOUNS:
                pronoun_mentions.append(mention)
        if entity_group is not None and len(pronoun_mentions) > 0:
            for mention in pronoun_mentions:
                sent_idx = int(mention.find("sentence").text)-1
                head_idx = int(mention.find("head").text)-1
                doc_coreference_dict[(sent_idx, head_idx)] = (entity_group, representative)
            # print(len(doc_coreference_dict), doc_coreference_dict)
    return doc_coreference_dict

def buildDataset(lexicon, embeddings):
    inputs = []
    outputs = []
    words = []
    for word in lexicon:
        if embeddings[word]:
            inputs.append(embeddings.decontextualizeWord(word))
            outputs.append(lexicon[word])
            words.append(word)

    return np.vstack(inputs), np.array(outputs), words

def predictions(model, embs):
    ''' Adjusts preds to tenarized (3 discrete values) values '''
    if embs:
        return [pred - 1 for pred in model.predict(embs)]
    else:
        return []

root_path = os.getcwd()
data_path = os.path.join(root_path, "..", "our_training_data")
# load articles

def extractEmbeddings(args, articles):
    # Initializes a dictionary that lets us go from (article, sent_id) -> sentence
    sentences = {}

    training_embeddings = EmbeddingManager()
    list_invalid_articles = []
    example_number = 0

    for article in tqdm(articles):
        try:
            h5_path = os.path.join(data_path, articleToName(article,append_str = ".txt.xml.hdf5"))
            xml_path = os.path.join(data_path, articleToName(article,append_str = ".txt.xml"))
        except OSError:
            print("Unable to read file {}".format(articleToName(article)))
            list_invalid_articles.append(article)
            continue

        root, document = process_xml_text(xml_path, correct_idx=False, stem=False, lower=True)
        sentence_nodes = [sent_node for sent_node in extractItem(root, ['document', 'sentences'], 'sentence')]
        try:
            with h5py.File(h5_path, 'r') as h5py_file:
                sent_to_idx = ast.literal_eval(h5py_file.get("sentence_to_index")[0])
                idx_to_sent = {int(idx):sent for sent, idx in sent_to_idx.items()}
                # TODO: Replace with logging
                if len(sentence_nodes) != len(idx_to_sent):
                    print("Mismatch in number of sentences, {} vs {} for article {}. Skipping article".format(len(document), len(idx_to_sent), article))
                    list_invalid_articles.append(article)
                    continue

                # Get coreference for the doc and parse it in a format that we can work with.
                doc_coreference_dict = None
                if args.use_coref:
                    doc_coreference = [coreference for coreference in extractItem(root, ['document', 'coreference', 'coreference'], 'coreference')]
                    doc_coreference_dict = build_coreference_dict(doc_coreference)
                
                for sent_idx in idx_to_sent:
                    sent_tokens = [tok for tok in  extractItem(sentence_nodes[sent_idx] , ['tokens'], 'token')]
                    sent_words = [extractItem(tok, ['word']).text.lower() for tok in sent_tokens]
                    sent_POS = [extractItem(tok, ['POS']).text for tok in sent_tokens]
                    sent_lemmas = [extractItem(tok, ['lemma']).text.lower() for tok in sent_tokens]
                    sent_embeddings = h5py_file.get(str(sent_idx))
                    sent_dependencies = [dep for dep in extractItem(sentence_nodes[sent_idx], ['dependencies'], 'dep')] 

                    if sent_embeddings.shape[1] != len(sent_lemmas):
                        print("Mismatch in number of token in sentence {} : {} vs {}. Skipping sentence".format(sent_idx, sent_embeddings.shape[1], len(sent_lemmas)))
                        continue

                    # Filter sentences based on word content
                    # These are to be used in the evaluation portion
                    examples = filterSentence(args, article, sent_idx, sent_dependencies, sent_lemmas, sent_POS, doc_coreference_dict)
                    example_number += len(examples)
                    
                    # NOTE: Weights refer to the accumulated layers of the 0) Inputs 1) Left context 2) Right context
                    def retrieveWordEmbedding(sent_embedding, verb_idx, weights = [0,1,0]):
                        return sent_embedding[0][verb_idx] * weights[0] + sent_embedding[1][verb_idx] * weights[1] + sent_embedding[2][verb_idx] * weights[2]
                    
                    for example in examples:
                        # TODO: Keep track of the other fields of example in particular: - ENTITYGROUP (whether the obj/subj was prot/gov)
                        #                                                                - DEPTYPE --> needed to establish which model we use
                        example['embedding'] =retrieveWordEmbedding(sent_embeddings, example['verb_idx']) 
                        training_embeddings.addItem(example)
        except OSError:
            list_invalid_articles.append(article)
            print("Invalid HDF5 file {}".format(articleToName(article)))
        except Exception as e:
            # Catch all for other errors 
            list_invalid_articles.append(article)
            print("{} occured. Skipping article {}".format(e, articleToName(article)))
    print("Total number of examples processed:", example_number)

    # TODO: Store the trained models and embeddings for future reference
    with open(args.emb_file, 'wb+') as embedding_fh:
        pickle.dump(training_embeddings, embedding_fh)

def trainModels(articles, args):
    with open(args.emb_file, 'rb') as embed_fh:
        training_embeddings = pickle.load(embed_fh)
    models = {}
    for operation, load_function in OPERATIONS:
        TRAIN = 0
        DEV = 1
        TEST = 2
        X = 0
        Y = 1
        WORDS = 2

        cf_splits = load_function()
        # TODO: Choose between type vs embedding prediction task
        splits = [buildDataset(split,training_embeddings) for split in cf_splits]
        print("Starting to tune {} model".format(operation))
        dev_score, optimized_weights = find_logistic_regression_weights(
                splits[TRAIN][X], splits[TRAIN][Y],
                splits[DEV][X], splits[DEV][Y],
                verbose=False)
        clf, test_score = logistic_regression(splits[TRAIN][X], splits[TRAIN][Y], splits[TEST][X], splits[TEST][Y], weights=optimized_weights, do_print=True, return_clf = True)
        models[operation] = {'model': clf,
                          'test_score' : test_score,
                          'dev_score': dev_score,
                          'weights': optimized_weights}

    with open(args.model_file, 'wb+') as models_fh:
        pickle.dump(models, models_fh)


def evaluateModels(articles, args):
    with open(args.emb_file, 'rb') as embed_fh:
        training_embeddings = pickle.load(embed_fh)
    with open(args.model_file,'rb') as model_fh:
        models = pickle.load(model_fh)

    entity_scores = {}
    riots, sources, _  = zip(*articles)
    riots = set(riots)
    sources = set(sources)

    agent_sentiment_model = models[OPERATIONS[0][0]]['model']
    patient_sentiment_model = models[OPERATIONS[1][0]]['model']
    power_model = models[OPERATIONS[2][0]]['model']
    agency_model = models[OPERATIONS[3][0]]['model']

    for riot in riots:
        entity_scores[riot] = defaultdict(dict)
        for source in sources:
            verb_insts = training_embeddings.fetchArticles((riot, source, ""))
            patient_instances = [inst for inst in verb_insts if 'patient' in inst]
            agent_instances = [inst for inst in verb_insts if 'agent' in inst]
            for current_entity in government_kw | protesters_kw:
                agent_embs = [instance['embedding'] for instance in agent_instances if instance['agent']['entity_lemma'] == current_entity]
                patient_embs = [instance['embedding'] for instance in patient_instances if instance['patient']['entity_lemma'] == current_entity]
                if agent_embs or patient_embs:
                    num_instances = len(agent_embs) + len(patient_embs)
                    power = sum(predictions(power_model, agent_embs))- sum(predictions(power_model, agent_embs))
                    agency = sum(predictions(agency_model, agent_embs))
                    sentiment = sum(predictions(agent_sentiment_model, agent_embs)) + sum(predictions(patient_sentiment_model, patient_embs))
                    entity_scores[riot][source][current_entity] = {'sentiment': int(sentiment),
                                                                   'power': int(power),
                                                                   'agency': int(agency),
                                                                   'count': int(num_instances)}
    
    import json
    with open(args.results_file,'w+') as scores_fh:
        json.dump(entity_scores, scores_fh)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['extract', 'train', 'evaluate'], required=True, default="evaluate")
    parser.add_argument('--emb_file', type=str, default="embedding.pkl")
    parser.add_argument('--model_file', type=str, default="model.pkl")
    parser.add_argument('--results_file', type=str, default="results.json")
    parser.add_argument('--use_all_verbs', action='store_true')
    parser.add_argument('--use_coref', action='store_true')
    args = parser.parse_args()

    articles = getArticleList(data_path)

    if args.mode == 'extract':
        extractEmbeddings(args, articles)
    elif args.mode == 'train': 
        trainModels(articles, args)
    elif args.mode == 'evaluate':
        evaluateModels(articles, args) #training_embs, models)
