import os
import re
from collections import namedtuple, defaultdict
from lexicons import load_connotation_frames, load_power_verbs, load_hannah_split
import h5py
import ast
import spacy

nlp = spacy.load("en_core_web_sm")

# Custom types to handle embeddings and articles
# TODO: Decide what information we really need. Putting
# info specific to our code will be a pain in the ass
# later on (if we try to use an external corpus)
# NOTE: Assumes that the article_idx are unique over
# a corpus.. Check this is a valid assumption
Article = namedtuple("Article", "riot src article_id")
SentenceTuple = namedtuple("SentenceTuple", "article_idx sent_idx") 
EmbeddingTuple = namedtuple("Embedding", "article_id sent_idx verb_idx word")

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
        self.tuple_to_embeddings = {}
        self.word_to_tuple = defaultdict(list)

    def __getitem__(self, ref_tupl):
        return self.tupl_to_embeddings[ref_tupl]
    
    def addItem(self, tupl, embedding):
        self.tuple_to_embeddings[tupl] = embedding
        self.word_to_tuple[tupl["word"]] += tupl

    def _calculateAverage(self, word):
        word_embs = np.stack([self.__getitem__(tupl) for tupl in self.word_to_tuple[word]])
        return np.mean(, axis=0)

    def decontextualize(self):
        return = {word : self._calculateAverage(word) for word in self.verb_to_embeddings}

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


root_path = os.getcwd()
article_path = os.path.join(root_path, "..", "our_articles")
data_path = os.path.join(root_path, "..", "our_training_data")
# load articles
articles = getArticleList(article_path)
import pdb;pdb.set_trace()

# NOTE: Initialize a dictionary that lets us go from (doc, sent_id) -> sentence
sentences = {}

# NOTE: Store embeddings according to (doc, sent_idx, verb_idx)
# TODO: Finish this class. At it's core is a dictionary, but it
# implements functionalities such as averaging
embeddings = EmbeddingManager()

# TODO: Repeat this for every article and store results in a dict
h5_file = os.path.join(data_path, articleToName(articles[0],append_str = ".txt.xml.hdf5"))
with h5py.File(h5_file, 'r') as h5py_file:
    sent_to_idx = ast.literal_eval(h5py_file.get("sentence_to_index")[0])
    idx_to_sent = {int(idx):sent for sent, idx in sent_to_idx.items()}
    # This loop imitates the extract_entities and get_embeddings function in "match_parse.py"  
    for idx, sent in idx_to_sent.items():
        parsed_sentence = nlp(sent)
        # TODO: Filter sentences based on word content
        # These are to be used in the evaluation portion
        # NOTE: This could be placed elsewhere
        if filterSentence(parsed_sentence):
            sentences[idx] = parsed_sentence

        # TODO: Extract all the verbs in the sentence. Store the article, sent id, word, and embedding
        # Anjalie's code does a parse bassed on entities, but then does a second pass
        # over the entire corpus to capture any missing verbs. Write this in the cf_parse function 
        # NOTE: We don't need the entities just yet. We can recover those from the filtered sentences
        # NOTE: Work over the parsed_sentence, not the filtered ones.
        '''
                sentence = root.find('document').find('sentences')[sent_id]
                for dep in sentence.find('dependencies').iter('dep'):
                    if int(dep.find('dependent').get("idx")) != int(mention.find('end').text) - 1:
                        continue

                    parent_id = int(dep.find('governor').get("idx")) - 1
                    parent = dep.find('governor').text

                    parent_lemma = sentence.find('tokens')[int(parent_id)].find('lemma').text

                    # We save the sentence id, the parent id, the entity name, the relationship, the article number
                    # With sentence id and parent id we can find embedding
                    if dep.get("type") in ["nsubj", "nsubjpass", "dobj"]:
                        verbs_to_cache.append(VerbInstance(sent_id, parent_id, parent, parent_lemma, dep.get("type"),  mention.find('text').text, "", filename))

            # end coreff chain
            # We do it this way so that if we set the name in the middle of the chain we keep it for all things in the chain
            if verbs_to_cache:
                name_to_verbs[name] += verbs_to_cache

        '''
        # Anjalie also keeps track of all the verbs present
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

        # TODO: Store the word embeddings for these verbs.  
        # With the verbs extracted for a specific sentence, we get their embeddings using
        '''
                s1 = h5py_file.get(str(idx))
                tupl_to_embeds[tupl] = (s1[0][tupl.verb_id] * weights[0] +
                                        s1[1][tupl.verb_id] * weights[1] +
                                        s1[2][tupl.verb_id] * weights[2])
        '''
        # This is currently done by the get_token_representation in Line 391, representations.py

# TODO: Logistic regression
# Logistic regression to predict the output using the decontextualized word embeddings
#cf = load_connotation_frames()
# NOTE: For now this will be done using Anjalie's code

# NOTE: Is the paper_runs the comp Anjalie ran against the other papers?
# Take the embeddings and average
# load_hannah_split for each header
# This fetches the test, train and dev splits in our connotation frames (???)
# format_runs for each

# TODO: Visualize and interpret results
# Apply using word embeddings to rest of dataset
# Store entities and score for different combos
# Make pretty graphs
