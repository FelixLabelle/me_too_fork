import os
import re
from collections import namedtuple, defaultdict
from lexicons import load_connotation_frames, load_power_verbs, load_hannah_split
from xml_helpers import process_xml_text
from weighted_tests import run_connotation_frames
import h5py
import ast
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

# Custom types to handle embeddings and articles
# TODO: Decide what information we really need. Putting
# info specific to our code will be a pain in the ass
# later on (if we try to use an external corpus)
# NOTE: Assumes that the article_idx are unique over
# a corpus.. Check this is a valid assumption
Article = namedtuple("Article", "riot src article_id")
EmbeddingTuple = namedtuple("Embedding", "article sent_idx verb_idx word word_embedding")

def articleToName(article, append_str = ""):
    """ Helper function that converts the 
    article tuple into a str (e.g. filename)"""
    return "_".join(article) + append_str


# TODO: Add in words for keyword filtering
# Consider making this dictionary specific to each conflict to improve performance
keywords = {"Government": [],
            "Protestors": []}

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

    def __getitem__(self, ref_tupl):
        return self.tupl_to_embeddings[ref_tupl]
    
    def addItem(self, tupl):
        self.tuples.append(tupl)
        self.word_to_tuple[tupl.word] += [len(self.tuples) - 1]
        self.article_to_tuple[tupl.article] = len(self.tuples) - 1

    def _calculateAverage(self, word):
        word_embs = np.stack([self.__getitem__(tupl) for tupl in self.word_to_tuple[word]])
        return np.mean(word_embs, axis=0)

    def decontextualize(self):
        return {word : self._calculateAverage(word) for word in self.verb_to_embeddings}

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

root_path = os.getcwd()
article_path = os.path.join(root_path, "..", "our_articles")
data_path = os.path.join(root_path, "..", "our_training_data")
# load articles
articles = getArticleList(article_path)

# Initializes a dictionary that lets us go from (doc, sent_id) -> sentence
sentences = {}

training_embeddings = EmbeddingManager()
list_invalid_articles = []

for article in tqdm(articles):
    # This loop imitates the extract_entities and get_embeddings function in "match_parse.py"
    try:
        h5_path = os.path.join(data_path, articleToName(article,append_str = ".txt.xml.hdf5"))
        xml_path = os.path.join(data_path, articleToName(article,append_str = ".txt.xml"))
    except OSERROR:
        print("Unable to read file {}".format(articleToName(article))
        continue

    root, document = process_xml_text(xml_path, correct_idx=False, stem=False, lower=True)
    sentence_nodes = [sent_node for sent_node in extractItem(root, ['document', 'sentences'], 'sentence')]
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

            # TODO: Extract all the verbs in the sentence. Store the article, sent id, word, and embedding
            # Anjalie's code does a parse bassed on entities, but then does a second pass
            # over the entire corpus to capture any missing verbs. Write this in the cf_parse function 
            # NOTE: We don't need the entities just yet. We can recover those from the filtered sentences
            # NOTE: Work over the parsed_sentence, not just the filtered ones.
            
            # TODO: Look into Anjalie's paper and see if she discusses the weights
            # Ditto for the elmo.py script
            def retrieveWordEmbedding(sent_embedding, verb_idx, weights = [0,1,0]):
                return sent_embedding[0][verb_idx] * weights[0] + sent_embedding[1][verb_idx] * weights[1] + sent_embedding[2][verb_idx] * weights[2]

            for word_idx, (word,word_POS) in enumerate(zip(sent_POS, sent_lemmas)):
                if word_POS == "VB":
                    verb_embedding = EmbeddingTuple(article,sent_idx, word_idx, word, retrieveWordEmbedding(sent_embeddings, word_idx))
                    training_embeddings.addItem(verb_embedding)
            

import pdb;pdb.set_trace()
# NOTE: Is the paper_runs the comp Anjalie ran against the other papers?
# Take the embeddings and average OVER the CF lex
# Anjalie also keeps track of all the verbs present in the lexicon
'''
# Also keep all verbs that are in lex
for s in root.find('document').find('sentences').iter('sentence'):
    sent = []
    for tok in s.find('tokens').iter('token'):
        sent.append(tok.find("word").text.lower())
        sent_id = int(s.get("id")) - 1
        verb_id = int(tok.get("id")) - 1
        key = (sent_id, verb_id)
        if key in final_verb_dict:d
            continue

        if tok.find('POS').text.startswith("VB"):
            final_verb_dict[key] = VerbInstance(sent_id, verb_id, tok.find("word").text, tok.find('lemma').text.lower(), "", "", "", filename)
    id_to_sent[sent_id] = " ".join(sent)

'''

run_connotation_frames(embeddings, avg_embeddings)

# TODO: Visualize and interpret results
# Apply using word embeddings to rest of dataset
# Store entities and score for different combos
# Make pretty graphs
