import os
import re
from collections import namedtuple, defaultdict
from lexicons import load_connotation_frames, load_power_verbs
import h5py
import ast

Article = namedtuple("Article", "riot src article_id")
# Consider making this dictionary specific to each conflict to improve performance
keywords = {"Government": [],
            "Protestors": []}

def get_embeddings(f, verb_dict, nlp_id_to_sent, weights=[0,1,0]):
    tupl_to_embeds = {}
    idx_to_sent = {}

    try:
        with h5py.File(f, 'r') as h5py_file:
            sent_to_idx = ast.literal_eval(h5py_file.get("sentence_to_index")[0])
            logger.info("Length of file {} is {} compared to {}".format(f,len(h5py_file),len(nlp_id_to_sent)))
            assert(len(h5py_file) - 1 == len(nlp_id_to_sent)), str(len(h5py_file) - 1)

            for s in sent_to_idx:
                idx = int(sent_to_idx[s])
                idx_to_sent[idx] = s.split()

            for _,tupl in verb_dict.items():
                # assert what we can, some sentences are missing cause the keys in sentence_to_index are not unique
                # we're just going to ignore the missing ones for now and hope they don't matter to much
                # We care more about ones with entities. If we're just doing this to get verb scores it's
                # not a big deal if we skip a bunch
                if not tupl.sent_id in idx_to_sent:
                    sent = nlp_id_to_sent[tupl.sent_id]
                    idx = int(sent_to_idx[sent])
                    tupl = tupl._replace(sent_id=idx)
                else:
                    idx = tupl.sent_id

                if tupl.verb.lower() != idx_to_sent[idx][tupl.verb_id]:
                    print("Mismatch", tupl.verb, str(idx_to_sent[idx][tupl.verb_id]), tupl.entity_name, f)
                    continue

                s1 = h5py_file.get(str(idx))
                tupl_to_embeds[tupl] = (s1[0][tupl.verb_id] * weights[0] +
                                        s1[1][tupl.verb_id] * weights[1] +
                                        s1[2][tupl.verb_id] * weights[2])
    except UnicodeEncodeError:
        logger.error("Unicode error, probably on mismatch")
    except OSError:
        logger.error("OSError", f)
    except KeyError:
        logger.error("KeyError", f)
    except Exception as e:
        logger.error("An unexpected error {} occured for file {}".format(e,f))

    return tupl_to_embeds, idx_to_sent


def getArticleList(dir_path, split_str="[_\.]"):
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

def articleToName(article, append_str = ""):
    return "_".join(article) + append_str

root_path = os.getcwd()
article_path = os.path.join(root_path, "..", "our_articles")
data_path = os.path.join(root_path, "..", "our_training_data")
# load articles
articles = getArticleList(article_path)
import pdb;pdb.set_trace()
import nltk
h5_file = os.path.join(data_path, articleToName(articles[0],append_str = ".txt.xml.hdf5"))
with h5py.File(h5_file, 'r') as h5py_file:
    sent_to_idx = ast.literal_eval(h5py_file.get("sentence_to_index")[0])
    idx_to_sent = {int(idx):sent for sent, idx in sent_to_idx.items()}
    sentences = [''] * len(idx_to_sent)
    for idx, sent for idx_to_sent.items():
        sentences[idx] = nltk.word_tokenize(sent)
        # Check that verb is ok
        pos = nltk.pos_tag(sentences[idx])
        # apply CFs to verbs
        # distribute power to the correct entities

    # Filter down these sentences
    logger.info("Length of file {} is {} compared to {}".format(f,len(h5py_file),len(nlp_id_to_sent)))

# filter them for references to keywords
# TODO: Consider adding coref resolution + ebedding filters (to remove references to other govs..)
# Calculate the scores for these sentences using the connotation frames
# Create training data for each conflict for each source
# Train LR
# Apply using word embeddings to rest of dataset
# Store entities and score for different combos
# Make pretty graphs
