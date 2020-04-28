import os
import re
from collections import namedtuple, defaultdict
from lexicons import load_connotation_frames, load_power_verbs, load_hannah_split
from xml_helpers import process_xml_text
from weighted_tests import logistic_regression, find_logistic_regression_weights
import h5py
import ast
from tqdm import tqdm
import numpy as np
import pickle
import stanza
import xml.etree.ElementTree as ET


# Custom types to handle embeddings and articles
# TODO: Decide what information we really need. Putting
# info specific to our code will be a pain in the ass
# later on (if we try to use an external corpus)
# NOTE: Assumes that the article_idx are unique over
# a corpus.. Check this is a valid assumption
Article = namedtuple("Article", "riot src article_id")
EmbeddingTuple = namedtuple("Embedding", "article sent_idx verb_idx word embedding")
CONNO_DIR="../frames/annotated_connotation_frames/"

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

def articleToName(article, append_str = ""):
    """ Helper function that converts the 
    article tuple into a str (e.g. filename)"""
    return "_".join(article) + append_str

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


def filterSentence(article, sent_idx, dependencies, lemmas, pos, doc_coreference_dict=None):
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
                # import pdb; pdb.set_trace()
            else:
                continue
            example = {
                "article": article,
                "sent_idx": sent_idx,
                "dep_type": dep.get("type"),
                "entity": dependent_lemma,
                "entity_idx": dependent_idx,
                "verb": governor_lemma,
                "verb_idx": governor_idx,  
                "entity_group": entity_group
            }
            if representative is not None:
                example["representative"] = representative

            verb_dependencies[governor_idx].append(example)

    # Iterate over verbs in the sentences filter out wrong nmod examples.
    for governor_idx in verb_dependencies:
        examples = verb_dependencies[governor_idx]
        dep_type_set = {example['dep_type'] for example in examples}
        for example in examples:
            # We keep example with type nmod only in passive sentences that have a nsubjpass for the verb.
            if example['dep_type'] is "nmod":
                if not "nsubjpass" in dep_type_set:
                    continue
            tuples.append(example)
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

# TODO: Complete the code and test
# Class that houses embeddings and manage operations over them 
class EmbeddingManager:
    def __init__(self):
        self.tuples = []
        self.word_to_tuple = defaultdict(list)
        self.article_to_tuple = {}

    def __getitem__(self, item):
        if type(item) == str:
            return [self.tuples[idx] for idx in self.word_to_tuple.get(item, [])]
        elif type(item) == Article:
            return self.tuples[self.article_to_tuple[item]]
        else:
            raise "Invalid data type"
    
    def __setstate__(self, state):
        self.tuples = state[0]
        self.word_to_tuple = state[1]
        self.article_to_tuple = state[2]

    def __getstate__(self):
        return (self.tuples, self.word_to_tuple, self.article_to_tuple)

    def addItem(self, tupl):
        self.tuples.append(tupl)
        self.word_to_tuple[tupl.word] += [len(self.tuples) - 1]
        self.article_to_tuple[tupl.article] = len(self.tuples) - 1

    def decontextualizeWord(self, word):
        word_embs = np.stack([self.tuples[idx].embedding for idx in self.word_to_tuple[word]])
        return np.mean(word_embs, axis=0)

def getArticleList(dir_path, split_str="[_\.]"):
    """ Function that loads all the files in a directory,
    verifies their titles are correctly formatted,
    returns them in a named tuple """

    article_list = []
    split_regex = re.compile(split_str)
    
    for file_name in os.listdir(dir_path):
        split_str = split_regex.split(file_name)
        if len(split_str) != 4:
            print("The name contains {} splits".format(len(split_str)))
            continue
        current_article = Article(*split_str[:3])
        article_list.append(current_article)
    return article_list

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


root_path = os.getcwd()
article_path = os.path.join(root_path, "..", "our_articles")
data_path = os.path.join(root_path, "..", "our_training_data")
# load articles
articles = getArticleList(article_path)

# Initializes a dictionary that lets us go from (article, sent_id) -> sentence
sentences = {}

training_embeddings = EmbeddingManager()
list_invalid_articles = []
example_number = 0
for article in tqdm(articles[:100]):
    # This loop imitates the extract_entities and get_embeddings function in "match_parse.py"
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
            # doc_coreference_dict = None
            doc_coreference = [coreference for coreference in extractItem(root, ['document', 'coreference', 'coreference'], 'coreference')]
            doc_coreference_dict = build_coreference_dict(doc_coreference)

            for sent_idx in idx_to_sent:
                sent_tokens = [tok for tok in extractItem(sentence_nodes[sent_idx] , ['tokens'], 'token')]
                sent_words = [extractItem(tok, ['word']).text.lower() for tok in sent_tokens]
                sent_POS = [extractItem(tok, ['POS']).text for tok in sent_tokens]
                sent_lemmas = [extractItem(tok, ['lemma']).text.lower() for tok in sent_tokens]
                sent_embeddings = h5py_file.get(str(sent_idx))
                sent_dependencies = [dep for dep in extractItem(sentence_nodes[sent_idx], ['dependencies'], 'dep')] 
                
                if sent_embeddings.shape[1] != len(sent_lemmas):
                    print("Mismatch in number of token in sentence {} : {} vs {}. Skipping sentence".format(sent_idx, sent_embeddings.shape[1], len(sent_lemmas)))
                    continue

                # TODO: Filter sentences based on word content
                # These are to be used in the evaluation portion
                # NOTE: This could be placed elsewhere
                examples = filterSentence(article, sent_idx, sent_dependencies, sent_lemmas, sent_POS, doc_coreference_dict)
                example_number += len(examples)
                # TODO: Look into Anjalie's paper and see if she discusses the weights
                # Ditto for the elmo.py script
                def retrieveWordEmbedding(sent_embedding, verb_idx, weights = [0,1,0]):
                    return sent_embedding[0][verb_idx] * weights[0] + sent_embedding[1][verb_idx] * weights[1] + sent_embedding[2][verb_idx] * weights[2]
                
                for example in examples:
                    # TODO: Keep track of the other fields of example in particular: - ENTITYGROUP (whether the obj/subj was prot/gov)
                    #                                                                - DEPTYPE --> needed to establish which model we use
                    verb_embedding = EmbeddingTuple(
                        example['article'],
                        example['sent_idx'],
                        example['verb_idx'],
                        sent_words[example['verb_idx']],
                        retrieveWordEmbedding(sent_embeddings, example['verb_idx'])
                    )
                    training_embeddings.addItem(verb_embedding)
    except OSError:
        list_invalid_articles.append(article)
        print("Invalid HDF5 file {}".format(articleToName(article)))
    except Exception as e:
        # Catch all for other errors 
        list_invalid_articles.append(article)
        print("{} occured. Skipping article {}".format(e, articleToName(article)))

print("Example number", example_number)



#embeddings = [tpl.word_embedding for tpl in training_embeddings.tuples]
HEADERS=["Perspective(wo)", "Perspective(ws)"]
# Load split test_frames

#import pdb;pdb.set_trace()
# TODO: Store the trained models and embeddings for future reference

with open("embedding.pkl", 'wb+') as embedding_fh:
    pickle.dump(training_embeddings, embedding_fh)

models = {}
for header in HEADERS:
    TRAIN = 0
    DEV = 1
    TEST = 2
    X = 0
    Y = 1
    cf_splits = load_hannah_split(CONNO_DIR, header, binarize=True, remove_neutral=False)
    # TODO: Choose between type vs embedding prediction task
    splits = [buildDataset(split,training_embeddings) for split in cf_splits]
    # TODO: Add in class weight tuning
    optimized_weights = None 
    '''
    find_logistic_regression_weights(
            splits[TRAIN][X], splits[TRAIN][Y],
            splits[DEV][X], splits[DEV][Y])
    '''
    clf = logistic_regression(splits[TRAIN][X], splits[TRAIN][Y], splits[TEST][X], splits[TEST][Y], weights=optimized_weights, do_print=True, return_clf = True)
    models[header] = clf

with open('models.pkl', 'wb+') as models_fh:
    pickle.dump(models, model_fh)

# NOTE: Is the paper_runs the comp Anjalie ran against the other papers?
# TODO: Visualize and interpret results
# Apply using word embeddings to rest of dataset
# Store entities and score for different combos
# Make pretty graphs
