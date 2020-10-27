# Topic Modeling using the BERTopic process
Source link: https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6

* Not directly use BERTopic
* But a combination of BERT, UMAP and HDBSCAN
* Step 1: use BERT to create word embeddings
* Step 2: use UMAP reduce dimensionality
* Step 3: use HDBSCAN to form clusters

## 1. Preprocess
* Import all the files
* Get the section for "History of present illness" from each file
* Remove numbers and other preprocess, but did not do lemmatization because word embedding would take care of that

## 2. Word embedding using BERT
* With 300 documents it only took about 20s to process

## 3. Use umap and hdbscan to form clusters based on word embedding
* Tune the following parameters 
* n_neighbors: smaller number results in more topics
* n_components: larger number results in more topics
* min_cluster_size: smaller number results in more topics, larger number also results in more documents without topics
* leaf_size: smaller number results in more topics

## 4. Step-wise topic adjustment by merging neighboring topics together
* Observation: it resulted in very imbalance distribution of topics (such as topic 0 is assigned to over half of the documents). Should use with caution.

## 5. Tested a few different parameters.
* The topics change when parameters change
* The topics change using the same set of parameters
* But topics look coherent due to word embedding

## 6. Observation and Future work
* Word embedding is definitely the way to go for topic modeling, because it put words with similar meanings in the same topic
* Need to work on how to stablize the model

## Example results:
topic 1
['rr', 'tachycardia', 'ray', 'respiratory', 'hallucinations', 'cc', 'device', 'sats', 'upper', 'control', 'pleural', 'antibiotics', 'htn', 'leg', 'neg', 'pericardial', 'initially', 'effusion', 'worsening', 'mg']  
topic 24
['fall', 'trach', 'female', 'woman', 'daughter', 'ivf', 'notes', 'paroxysmal', 'fell', 'old', 'benadryl', 'year', 'time', 'doses', 'fibrillation', 'management', 'vent', 'fracture', 'wbc', 'ms']  
topic 11
['ercp', 'lobe', 'gi', 'rbc', 'hickman', 'hct', 'biliary', 'afib', 'epigastric', 'zosyn', 'hypertensive', 'pneumonia', 'procedure', 'platelets', 'etoh', 'infections', 'drain', 'hypoxia', 'fluid', 'change']  
topic 17
['il', 'lasix', 'therapy', 'melanoma', 'disease', 'rca', 'post', 'coronary', 'soft', 'dilantin', 'high', 'mr', 'groin', 'lung', 'subcutaneous', 'erythema', 'despite', 'dose', 'stenting', 'plts']  

<img src = "https://github.com/sindhri/topic_modeling/blob/main/img/model14.png" width = "800">


topic 0
['female', 'woman', 'ms', 'daughter', 'cm', 'old', 'year', 'arrival', 'fall', 'ivf', 'seizure', 'dysuria', 'endometrial', 'noted', 'pressure', 'acute', 'bowel', 'past', 'right', 'changes']  
topic 4
['tachycardia', 'ray', 'respiratory', 'hallucinations', 'cc', 'device', 'sats', 'rr', 'control', 'pleural', 'antibiotics', 'leg', 'pericardial', 'neg', 'initially', 'effusion', 'worsening', 'placed', 'recent', 'lower']  
topic 17
['man', 'says', 'episode', 'bmt', 'chronic', 'called', 'cpap', 'steroids', 'verbal', 'ffp', 'weakness', 'hemorrhage', 'denied', 'swelling', 'wernicke', 'levofloxacin', 'ceftriaxone', 'years', 'bed', 'ems']  
topic 9
['stones', 'stone', 'vt', 'headaches', 'onset', 'po', 'syndrome', 'ruq', 'mm', 'cholecystitis', 'large', 'crampy', 'days', 'volume', 'stool', 'hernia', 'report', 'non', 'pericholecystic', 'positional']  

<img src = "https://github.com/sindhri/topic_modeling/blob/main/img/model15.png" width = "800">
