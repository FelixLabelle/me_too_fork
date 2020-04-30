To run Anjalie's code, you use fullpipeline.sh. This should serve as a guide
The script is divided, more or less, into 4 portions
0. Preprocessing: Calculating POS; corefs; dependencies; elmo embeddings (First 2 ifs)
1. Build a list of verbs and entities over the entire corpus (Third if)
2. Training and evaluating a logistic regression based on the contextual embeddings
3. Loading the scores over entities and running displays

Our version of the code, score\_entities.py, only requires implementing points 1 -> 3

