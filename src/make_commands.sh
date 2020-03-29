# Prepare scripts for running in parallel
# Argument to the script is the directory containing files to process
# The input file path should contain a folder called "raw_tokenized"
# Output files will be in the folder "embeddings" instead of "raw_tokenized"
# The scripts to run will be placed in a directory called "run_scripts"

run_scripts=()
IN_DIR=$1

jobs_per_file=50
count=0
file_idx=0

run_script="./commands.txt"
chmod +r $run_script

for f in ${IN_DIR}; do

  new_name=${f/.elmo/.hdf5}
  full_new_name=${new_name/raw_tokenized/embeddings}
  echo "! ls $full_new_name && allennlp elmo $f $full_new_name --all" >> $run_script

done
echo Num items: ${#run_scripts[@]}
echo Data: ${run_scripts[@]}
