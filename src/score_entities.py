import os
import re
from collections import namedtuple, default_dict

Article = namedtuple("Article", "riot src article_id")
# Consider making this dictionary specific to each conflict to improve performance
keywords = {"Government": []
            "Protestors": []}


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
    return "_".join(*article) + append_str

root_path = os.getcwd()
data_path = os.path.join(root_path, "..", "our_articles")
import pdb;pdb.set_trace()
articles = getArticleList(data_path)
# load articles
# filter them for references to keywords
# TODO: Consider adding coref resolution + ebedding filters (to remove references to other govs..)
# Calculate the scores for these sentences using the connotation frames
# Create training data for each conflict for each source
# Train LR
# Apply using word embeddings to rest of dataset
# Store entities and score for different combos
# Make pretty graphs
