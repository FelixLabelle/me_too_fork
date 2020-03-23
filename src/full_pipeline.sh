STANFORD_DIR="../tools/stanford-corenlp-full-2018-10-05" # Downloaded Stanford parser
EVAL_SCORE_CACHE="subject_entities_limit.pickle"
PRETRAINED_MATCHED_EMBEDDING_CACHE="./NYT_matched_tupl.pickle" #./tools/metoo010_matched_tupl.pickle"

PRETRAINING_OUTPUT_DIR="../pretraining_data" 
PRETRAINING_DATA="/mnt/data/NYT_data"

TRAINING_DATA="../our_articles"
TRAINING_OUTPUT_DIR="../our_training_data"

PARSE_FILES=true
EXTRACT_ELMO=false
CREATE_CACHE=false
EVALUATE_DATA=false
PRETRAIN=false

if $PRETRAIN; then
	$TRAINING_DATA=$PRETRAINING_DATA
	$TRAINING_OUTPUT_DIR=$PRETRAINING_OUTPUT_DIR
fi

if $PARSE_FILES; then
	find $TRAINING_DATA -name "*txt" > filelist.txt
	java -Xmx15g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse,dcoref,depparse -filelist filelist.txt -outputDirectory $TRAINING_OUTPUT_DIR
fi

if $EXTRACT_ELMO; then
	chmod -R 777 $TRAINING_OUTPUT_DIR
	python prep_elmo.py --input_glob "$TRAINING_OUTPUT_DIR/*.xml" --output_dir $TRAINING_OUTPUT_DIR
	# Extract elmo embeddings over all files
	./make_commands.sh "$TRAINING_OUTPUT_DIR/*.elmo" 
	#chmod -R 777 ../run_scripts
	parallel < commands.txt	
	#for file in run_scripts/*.sh; do
	#	echo $file
	#	"$file"
	#done
fi

if $CREATE_CACHE; then
	# Cache all verbs and entities from elmo embeddings
	python match_parse.py --cache $MATCHED_EMBEDDING_CACHE --nlp_path $TRAINING_OUTPUT_DIR --embed_path $TRAINING_OUTPUT_DIR
fi

if $EVALUATE_DATA; then
	# Run evaluation over verbs
	python weighted_tests.py --cache $MATCHED_EMBEDDING_CACHE --from_scratch
	# Run evalulations against power scripts
	#python metoo_eval.py --embedding_cache $MATCHED_EMBEDDING_CACHE --score_cache $EVAL_SCORE_CACHE
	# Run analyses in paper
	python metoo_analysis.py --embedding_cache $MATCHED_EMBEDDING_CACHE
fi
