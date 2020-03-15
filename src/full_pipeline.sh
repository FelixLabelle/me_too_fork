# This file gives an overview of the steps for processing files, extracting embeddings, and running evaluations
# Running this script end-to-end has NOT been tested
# These directories need to be pre-created as specified
RAW_ARTICLES_DIR="/mnt/data/data" # Each article should be in a separate file, where the filename is article_id.txt
STANFORD_DIR="../tools/stanford-corenlp-full-2018-10-05" # Downloaded Stanford parser
# These are files are created by this script or directories that will be populated (directories need to exist)
ELMO_INPUT_DIR="../outputs" # Directory to store input to ELMo
ELMO_OUTPUT_DIR="../outputs" # This should be the same as ELMO_INPUT_DIR but with "raw_tokenized" replaced with "embeddings"
NLP_OUTPUT_DIR="../pretrained_outputs" # I copied these over to tir
MATCHED_EMBEDDING_CACHE="../tools/metoo010_matched_tupl.pickle"
PRETRAINING_DATA="/mnt/data/NYT_data"
# TODO: Add an argument as input that makes this generalize
EVAL_SCORE_CACHE="subject_entities_limit.pickle"


# Run stanford NLP pipleine over all texts
find $PRETRAINING_DATA -name "*txt" > filelist.txt
java -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse, dcoreref,depparse -filelist filelist.txt -outputDirectory $NLP_OUTPUT_DIR

# Use output of parser to build tokenized files
python prep_elmo.py --input_glob "$NLP_OUTPUT_DIR/*.xml" --output_dir $ELMO_INPUT_DIR

# Extract elmo embeddings over all files
./make_run_scripts.sh "$ELMO_INPUT_DIR/*.elmo" 
chmod -R 777 ../run_scripts
for file in ../run_scripts/*.sh; do
	[ -f "$file" ] && [-x "$file"] && "$file"
done

# Cache all verbs and entities from elmo embeddings
python match_parse.py --cache $MATCHED_EMBEDDING_CACHE --nlp_path $NLP_OUTPUT_DIR --embed_path $ELMO_OUTPUT_DIR

# Run evaluation over verbs
python weighted_tests.py --cache $MATCHED_EMBEDDING_CACHE --from_scratch

# Run evalulations against power scripts
python metoo_eval.py --embedding_cache $MATCHED_EMBEDDING_CACHE --score_cache $EVAL_SCORE_CACHE

# Run analyses in paper
python metoo_analysis.py --embedding_cache $MATCHED_EMBEDDING_CACHE
