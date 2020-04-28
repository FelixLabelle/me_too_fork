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

# Custom types to handle embeddings and articles
# TODO: Decide what information we really need. Putting
# info specific to our code will be a pain in the ass
# later on (if we try to use an external corpus)
# NOTE: Assumes that the article_idx are unique over
# a corpus.. Check this is a valid assumption
Article = namedtuple("Article", "riot src article_id")
EmbeddingTuple = namedtuple("Embedding", "article sent_idx verb_idx word embedding")
CONNO_DIR="../frames/annotated_connotation_frames/"

def articleToName(article, append_str = ""):
    """ Helper function that converts the 
    article tuple into a str (e.g. filename)"""
    return "_".join(article) + append_str


# TODO: Implement a filter. Start by using the keywords defined above
def filterSentence(sentence):
    return True


# TODO: Complete the code and test
# Class that houses embeddings and manage operations over them 
class EmbeddingManager:
    def __init__(self):
        self.tuples = []
        self.word_to_tuple = defaultdict(list)
        self.article_to_tuple = {}
   
    def fetchArticle(self, search_criterion):
        """ Returns tuples using a search tuple of the same type, with wildcard fields marked as None or '' """
        def match(t1, search_tupl):
            return all([not search_term or term == search_term for term,search_term in zip(t1,search_tupl)])
        return [tupl for tupl in self.tuples if match(tupl.article, search_criterion)]

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

for article in tqdm(articles[:10]):
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
            
            for sent_idx in idx_to_sent:
                sent_tokens = [tok for tok in  extractItem(sentence_nodes[sent_idx] , ['tokens'], 'token')]
                sent_words = [extractItem(tok, ['word']).text.lower() for tok in sent_tokens]
                sent_POS = [extractItem(tok, ['POS']).text for tok in sent_tokens]
                sent_lemmas = [extractItem(tok, ['lemma']).text.lower() for tok in sent_tokens]
                sent_embeddings = h5py_file.get(str(sent_idx))
                
                if sent_embeddings.shape[1] != len(sent_lemmas):
                    print("Mismatch in number of token in sentence {} : {} vs {}. Skipping sentence".format(sent_idx, sent_embeddings.shape[1], len(sent_lemmas)))
                    continue

                # TODO: Filter sentences based on word content
                # These are to be used in the evaluation portion
                # NOTE: This could be placed elsewhere
                if filterSentence(sent_lemmas):
                    # Check if there is a better way to index, we want to be easily able to remove types 
                    # of sentences (source, topic, etc..)
                    sentences[(article, sent_idx)] = sent_lemmas
                # NOTE: Weights refer to the accumulated layers of the 0) Inputs 1) Left context 2) Right context
                def retrieveWordEmbedding(sent_embedding, verb_idx, weights = [0,1,0]):
                    return sent_embedding[0][verb_idx] * weights[0] + sent_embedding[1][verb_idx] * weights[1] + sent_embedding[2][verb_idx] * weights[2]
                
                for word_idx, (word,word_POS) in enumerate(zip(sent_lemmas, sent_POS)):
                    if word_POS.startswith("VB"):
                        verb_embedding = EmbeddingTuple(article,sent_idx, word_idx, word, retrieveWordEmbedding(sent_embeddings, word_idx))
                        training_embeddings.addItem(verb_embedding)
    except OSError:
        list_invalid_articles.append(article)
        print("Invalid HDF5 file {}".format(articleToName(article)))
    except Exception as e:
        # Catch all for other errors 
        list_invalid_articles.append(article)
        print("{} occured. Skipping article {}".format(e, articleToName(article)))

HEADERS=["Perspective(wo)", "Perspective(ws)"]

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
    WORDS = 2

    cf_splits = load_hannah_split(CONNO_DIR, header, binarize=True, remove_neutral=False)
    # TODO: Choose between type vs embedding prediction task
    splits = [buildDataset(split,training_embeddings) for split in cf_splits]
    print("Starting to tune {} model".format(header))
    dev_score, optimized_weights = find_logistic_regression_weights(
            splits[TRAIN][X], splits[TRAIN][Y],
            splits[DEV][X], splits[DEV][Y],
            verbose=False)
    clf, test_score = logistic_regression(splits[TRAIN][X], splits[TRAIN][Y], splits[TEST][X], splits[TEST][Y], weights=optimized_weights, do_print=True, return_clf = True)
    models[header] = {'model': clf,
                      'test_score' : test_score,
                      'dev_score': dev_score}

with open('models.pkl', 'wb+') as models_fh:
    pickle.dump(models, models_fh)

# NOTE: Is the paper_runs the comp Anjalie ran against the other papers?
# TODO: Visualize and interpret results
# Apply using word embeddings to rest of dataset
# Store entities and score for different combos
# Make pretty graphs
