RAW_ARTICLES_DIR="/mnt/data/data" # Each article should be in a separate file, where the filename is article_id.txt
STANFORD_DIR="../tools/stanford-corenlp-full-2018-10-05" # Downloaded Stanford parser
# These are files are created by this script or directories that will be populated (directories need to exist)
ELMO_INPUT_DIR="../pretraining_data" # Directory to store input to ELMo
ELMO_OUTPUT_DIR="../pretraining_data" # This should be the same as ELMO_INPUT_DIR but with "raw_tokenized" replaced with "embeddings"
NLP_OUTPUT_DIR="../pretraining_data" # I copied these over to tir
MATCHED_EMBEDDING_CACHE="." #./tools/metoo010_matched_tupl.pickle"
PRETRAINING_DATA="/mnt/data/NYT_data"
EVAL_SCORE_CACHE="subject_entities_limit.pickle"

PARSE_FILES=false
EXTRACT_ELMO=false
CREATE_CACHE=true
EVALUATE_DATA=false

if $PARSE_FILES; then
	find $PRETRAINING_DATA -name "*txt" > filelist.txt
	java -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse, dcoreref,depparse -filelist filelist.txt -outputDirectory $NLP_OUTPUT_DIR
fi

if $EXTRACT_ELMO; then
	python prep_elmo.py --input_glob "$NLP_OUTPUT_DIR/*.xml" --output_dir $ELMO_INPUT_DIR
	chmod -R 777 $NLP_OUTPUT_DIR
	# Extract elmo embeddings over all files
	./make_run_scripts.sh "$ELMO_INPUT_DIR/*.elmo" 
	chmod -R 777 ../run_scripts
	for file in run_scripts/*.sh; do
		echo $file
		"$file"
	done
fi

if $CREATE_CACHE; then
	# Cache all verbs and entities from elmo embeddings
	python match_parse.py --cache $MATCHED_EMBEDDING_CACHE --nlp_path $NLP_OUTPUT_DIR --embed_path $ELMO_OUTPUT_DIR
fi

if $EVALUATE_DATA; then
	# Run evaluation over verbs
	python weighted_tests.py --cache $MATCHED_EMBEDDING_CACHE --from_scratch

	# Run evalulations against power scripts
	python metoo_eval.py --embedding_cache $MATCHED_EMBEDDING_CACHE --score_cache $EVAL_SCORE_CACHE

	# Run analyses in paper
	python metoo_analysis.py --embedding_cache $MATCHED_EMBEDDING_CACHE
fi
