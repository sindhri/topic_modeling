import numpy as np
import pandas as pd
import re, nltk, spacy, gensim, string

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn

from os import listdir


# load one file
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# load all the files in a directory
def process_docs(directory):
    filenames = list()
    docs = list()
    for filename in listdir(directory):
        path = directory + '/' + filename
        # load document
        filenames.append(filename)
        docs.append(load_doc(path))
    print('Loaded %d files' % len(filenames))
    return filenames, docs

# load both txt and ann files and put them in a dataframe sorted by id
def build_df(directory):
    filenames,docs = process_docs(directory)
    df = pd.DataFrame(list(zip(filenames, docs)), columns=['filename','content'])
    df['id'] = df.filename.apply(lambda x: x.split('.')[0])
    df['ext'] = df.filename.apply(lambda x: x.split('.')[1])
    df = df.pivot(index='id',columns='ext')
    df.columns = ['filename_ann','filename_txt','content_ann','content_txt']
    df = df.reset_index()
    df = df.sort_values(by=['id'],ignore_index=True)
    return df

# text cleaning
# Make text lowercase, remove text in square brackets, remove punctuation, remove words containing numbers, remove line breaks.
def clean_text(text):
    text = text.lower() #Make text lowercase
    text = re.sub(r'\[.*?\]', ' ', text) #remove text in square brackets,
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text) #remove punctuation
    text = re.sub(r'\w*\d\w*', ' ', text) #remove words containing numbers
    text = re.sub(r'\n', ' ', text) #remove line breaks
    return text

# text lemmatization
nlp = spacy.load('en')
def lemmatizer(text):        
    sent = []
    doc = nlp(text)
    for word in doc:
        sent.append(word.lemma_)
    output = " ".join(sent)
    output = re.sub('-PRON-', '', output)
    return output

# preprocess text, cleaning and lemmatization
def text_preprocess(text):
    text_clean = clean_text(text)
    text_lemmatized = lemmatizer(text_clean)
    return text_lemmatized

# create a new column with preprocessed text
def column_preprocess(df, col):
    data_list = df[col].values.tolist()
    data_lemmatized = [text_preprocess(data) for data in data_list]

    #add the processed content as a new column to the dataframe
    df[col + '_preprocessed'] = data_lemmatized
    return df


# find the potential field names from the original uncleaned text, 
# which were marked with a : at the end of the line
def find_field_names(df, col):
    field_names = set()
    for row in df[col]:
        row_sentences = list(row.split('\n'))
        row_sentences = [sentence for sentence in row_sentences if sentence!='']
        new_field_names = [sentence[:-1] for sentence in row_sentences if sentence[-1]==':']
        field_names = field_names.union(new_field_names)
    field_names = list(field_names)
    
    percent = []
    for field_name in field_names:
        count = 0
        for row in df[col]:
            if field_name in row:
                 count = count+1 
        current_percent = count/df.shape[0]*100
        percent.append(current_percent)
    df = pd.DataFrame(list(zip(field_names, percent)), columns = ['field_name', 'percent'])
    df = df.sort_values('percent',ascending = False,ignore_index=True)
    
    plt.figure(figsize=(20, 4))
    plt.plot(df.percent)
    plt.xlabel('index of stop words')
    plt.ylabel('percentage of  occurance in all the files')
    plt.show()
    return df

# extract content by a keyword
# the content should start at the location where there is keyword:, eg, 'Chief Complaint:'
# It continues search for the next line that has another keyword from a keyword list, or until the end of the file
# the keep the content in between 

KEYWORD_LIST = ['Admission Date', 'Discharge Date', 'Date of Birth', 'Sex','Service', 'Allergies', 'Attending', 
                'Chief Complaint', 'Major Surgical or Invasive Procedure', 'History of Present Illness', 
                'Past Medical History', 'Social History', 'Family History', 'Physical Exam', 
                'Pertinent Results', 'Brief Hospital Course','Medications on Admission', 
                'Discharge Medications', 'Discharge Disposition', 'Facility', 
                'Discharge Diagnosis', 'Discharge Condition', 'Discharge Instructions', 
                'Followup Instructions','PHYSICAL EXAMINATION ON ADMISSION', 'History of the Present Illness',
               'LABORATORY DATA', 'HOSPITAL COURSE', 'DISPOSITION', 'MAJOR PROCEDURES', 'CONDITION ON DISCHARGE',
               'Addendum', 'DISCHARGE SUMMARY ADDENDUM','DISCHARGE DIAGNOSES']

# After comparing the stop_words list from nltk, word cloud and sklearn, 
# slkean has the most extensive list, thus was chosen as the default stop_words list
def default_stop_words():
    stop_words = text.ENGLISH_STOP_WORDS
    return stop_words

def extract_by_keyword(original_text, keyword):
    # make everything lowercase
    original_lower = original_text.lower()
    
    keyword = keyword.lower()
    if keyword[-1] != ':':
        keyword = keyword + ':'
    
    keyword_list = KEYWORD_LIST
    keyword_list = [keyword.lower() for keyword in keyword_list]
    keyword_list = [keyword + ':' for keyword in keyword_list]

    #split by lines because the beginning of a section always starts from a new line
    t = pd.DataFrame(original_lower.split('\n'))
    t.columns = ['text']

    #fine the index where the line of text contains the keyword immediately followed by :
    content_start_index = t.index[t['text'].str.contains(keyword)].to_list() #The majority of the files
    if content_start_index == [] and 'history of present illness' in keyword:
        content_start_index = t.index[t['text'].str.contains('history of the present illness')].to_list() #accommodating one special case
    if content_start_index == [] and 'discharge diagnosis' in keyword:
        content_start_index = t.index[t['text'].str.contains('discharge diagnoses')].to_list() #accommodating one special case
        
    # if the content_start_index is found, look for content_end_index
    if content_start_index !=[]:
        content_start_index = content_start_index[0]
        content_end_index = 0
        for row_index in range(content_start_index+1,t.shape[0]):
            for akeyword in keyword_list:
                if akeyword in t['text'][row_index]:
                    content_end_index = row_index -1
                    break
            if content_end_index > 0:
                break
        pre_cleaning = ',,'.join(t['text'][content_start_index:content_end_index]) #use ,, to mark the original end of line
        post_cleaning = text_preprocess(pre_cleaning)[(len(keyword)+1):]
    else:
        pre_cleaning = ''
        post_cleaning = ''
    return [pre_cleaning, post_cleaning]
        
def extract_by_keyword_df(df, keyword):
    if keyword[-1] != ':':
        keyword = keyword + ':'
    new_colname = keyword[:-1].lower()
    new_colname = re.sub(' ', '_', new_colname)
    df[new_colname] = df['content_txt'].apply(lambda x: extract_by_keyword(x, keyword)[0])    
    df[new_colname + '_preprocessed'] = df['content_txt'].apply(lambda x: extract_by_keyword(x, keyword)[1])
    n_empty_rows = df[df[new_colname + '_preprocessed']==''].shape[0]
    total_rows = df.shape[0]
    print(keyword, 'total number of rows', total_rows, 'empty rows', n_empty_rows)
    return df

# Extract the first discharge diagnosis
# first get the full text of discharge diagnosis
# then extract either the first line, or the first line of primary (diagnosis)
def extract_first_diagnosis(original_text):
    discharge_diagnosis = extract_by_keyword(original_text, 'discharge diagnosis')[0]
    if discharge_diagnosis == '':
        first_diagnosis = ''
    else:
        discharge_diagnosis = discharge_diagnosis.split(',,')
        first_diagnosis = ''
        if len(discharge_diagnosis) == 1:
            discharge_diagnosis = discharge_diagnosis[0]
            discharge_diagnosis = list(filter(None, discharge_diagnosis.split(' ')))
            if discharge_diagnosis[0].lower()=='discharge':
                discharge_diagnosis.pop(0)
            if discharge_diagnosis[0].lower()=='primary:':
                discharge_diagnosis.pop(0)
            if discharge_diagnosis[0].lower()=='diagnosis:' or discharge_diagnosis[0].lower()=='diagnoses:':
                discharge_diagnosis.pop(0)
            first_diagnosis = ' '.join(discharge_diagnosis)
        else:
            for line in discharge_diagnosis:
                if text_preprocess(line) not in ['discharge diagnosis', 'primary', 'primary diagnosis','discharge diagnoses']:
                    first_diagnosis = line
                    first_diagnosis = re.sub('primary', '', first_diagnosis, flags=re.IGNORECASE)
                    break
    return first_diagnosis

# extract first diagnosis from the dataframe and clean
def extract_first_diagnosis_df_clean(df):

    new_colname = 'first_diagnosis'
    df[new_colname] = df['content_txt'].apply(lambda x: extract_first_diagnosis(x))    
    df[new_colname + '_preprocessed'] = df[new_colname].apply(lambda x: text_preprocess(x))
    n_empty_rows = df[df[new_colname]==''].shape[0]
    total_rows = df.shape[0]
    print('First Diagnosis:', 'total number of rows', total_rows, 'empty rows', n_empty_rows)
    return df

# create a new column that contains potential label for topic generation
# looks takes the following components in order to fill the field
# if it couldn't fill it using the first component, go to the next component 
# first_diagnosis - chief complaint - Service (if it's not 'medicine') -- Addendum
def get_label_candidate(df):
    df['label_candidate'] = df['first_diagnosis_preprocessed']
    empty_indics = df.index[df['label_candidate']==''].to_list()
    for index in empty_indics:
        df['label_candidate'][index] = df['chief_complaint_preprocessed'][index]
    empty_indics = df.index[df['label_candidate']==''].to_list()
    for index in empty_indics:
        df['label_candidate'][index] = df['service_preprocessed'][index]
    empty_indics = df.index[df['label_candidate']==''].to_list()
    print('label candidate empty line', len(empty_indics))


    return df

#make a word cloud giving the text in a list, and stop words
def make_word_cloud(data_list, stop_words=default_stop_words()):

    wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stop_words,
                          max_words=500,
                          max_font_size=40, 
                          random_state=42
                         ).generate(' '.join(data_list))
    print(wordcloud)
    fig = plt.figure(1)
    rcParams['figure.figsize']=(12.0,12.0)  
    rcParams['font.size']=12            
    rcParams['savefig.dpi']=100             
    rcParams['figure.subplot.bottom']=.1 
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show();
    
#calculate unigram/bigram/trigram with stop_words
def get_top_ngram(data, ngram=1, top_n=None, stop_words=default_stop_words()):
    ngram_names = ['unigram','bigram','trigram']
    ngram_name = ngram_names[ngram-1]
    #calculate unigram
    vec = CountVectorizer(ngram_range = (ngram, ngram), stop_words=stop_words).fit(data)
    bag_of_words = vec.transform(data)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    common_words = words_freq[:top_n]
    #visulization
    df = pd.DataFrame(common_words, columns = [ngram_name , 'count'])
    fig = go.Figure([go.Bar(x=df[ngram_name], y=df['count'])])
    fig.update_layout(title=go.layout.Title(text="Top " + str(top_n) + ' '+ ngram_name))
    fig.show()

# vectorize the content
def get_data_vectorized(data, stop_words=default_stop_words(), max_features=3000):
    vectorizer = CountVectorizer(analyzer='word',       
                             min_df=1,                         # minimum requied occurences of a word 
                             stop_words=stop_words,             # remove stop words
                             lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             max_features=max_features,             # max number of unique words
                            )

    data_vectorized = vectorizer.fit_transform(data)
    # Materialize the sparse data
    data_dense = data_vectorized.todense()

    # Compute Sparsicity = Percentage of Non-Zero cells
    print("Vectorized data sparsicity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")
    return data_vectorized, vectorizer

def do_vectorization_lda(data, n_components, stop_words, max_features):
  
    data_vectorized, vectorizer = get_data_vectorized(data, stop_words=default_stop_words(), max_features=3000)
    lda = LatentDirichletAllocation(n_components=n_components,
                                     learning_decay=.7,
                                     batch_size=64,
                                     learning_offset=10,
                                     max_iter=10,
                                     max_doc_update_iter=100)
    lda.fit(data_vectorized)
    lda_output = {}
    lda_output['best_lda_model'] = lda
    lda_output['data_vectorized'] = data_vectorized
    lda_output['vectorizer'] = vectorizer
    print(lda)
    return lda_output

# create document topic matrix
def get_document_topic_matrix(best_lda_model, data_vectorized):
    # Create Document - Topic Matrix
    lda_output = best_lda_model.transform(data_vectorized)

    # column names
    topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]

    # index names
    docnames = ["Doc" + str(i) for i in range(data_vectorized.shape[0])]

    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic

    # Apply Style
    df_document_topic_top = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)
    return df_document_topic, df_document_topic_top, topicnames

    # Styling
def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)

def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)

# get_topic_distribution
def get_topic_distribution(df_document_topic):
    df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
    df_topic_distribution.columns = ['Topic Num', 'Num Documents']
    return df_topic_distribution

# get topic_keywords
def get_topic_keywords(vectorizer, lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]

    return topic_keywords, df_topic_keywords

# get topic-keyword matrix
def get_topic_keywords_matrix(lda_model, vectorizer, topicnames):
    df_topic_keywords_matrix = pd.DataFrame(lda_model.components_)

    df_topic_keywords_matrix.columns = vectorizer.get_feature_names()
    df_topic_keywords_matrix.index = topicnames

    return df_topic_keywords_matrix

# get several metrics from the lda model
def evalute_lda(lda_output):
    best_lda_model = lda_output['best_lda_model']
    data_vectorized = lda_output['data_vectorized']
    vectorizer = lda_output['vectorizer']
                                            
    df_document_topic, df_document_topic_top, topicnames = get_document_topic_matrix(best_lda_model, data_vectorized)
    df_topic_distribution = get_topic_distribution(df_document_topic)
    topic_keywords, df_topic_keywords = get_topic_keywords(vectorizer=vectorizer, lda_model=best_lda_model, n_words=20)
    df_topic_keywords_matrix = get_topic_keywords_matrix(best_lda_model, vectorizer, topicnames)
    lda_metrics = {}
    lda_metrics['df_document_topic'] = df_document_topic
    lda_metrics['df_document_topic_top'] = df_document_topic_top
    lda_metrics['df_topic_distribution'] = df_topic_distribution
    lda_metrics['df_topic_keywords'] = df_topic_keywords
    lda_metrics['df_topic_keywords_matrix'] = df_topic_keywords_matrix
    return lda_metrics

def add_topic_keywords_to_df(df, lda_metrics):
    df_document_topic = lda_metrics['df_document_topic']
    df_document_topic = df_document_topic.reset_index()
    df = df.join(df_document_topic['dominant_topic'])
    df_topic_keywords = lda_metrics['df_topic_keywords']
    label_list = ['Word ' + str(n) for n in range(0,11)]
    df_topic_keywords['first_10_keywords'] = df_topic_keywords[label_list].agg(', '.join, axis=1)
    dict_topic_keywords = {}
    for index in df_topic_keywords.index:
        dict_topic_keywords[index] = df_topic_keywords['first_10_keywords'][index]

    df['topic_first_10_keywords'] = df['dominant_topic'].apply(lambda x:dict_topic_keywords['Topic ' + str(x)])
    return df

# make single prediction by input text
def predict_topic(data, lda_output, lda_metrics):
    
    if type(data)==str: #single prediction
    # Step 1: Clean with the text
        data_1 = [text_preprocess(data)]
    else: # a list of content to predict
        data_1 = [text_preprocess(row) for row in data]

    # Step 2: Vectorize transform
    data_2 = lda_output['vectorizer'].transform(data_1)

    # Step 3: LDA Transform
    topic_probability_scores = lda_output['best_lda_model'].transform(data_2)
    topic = lda_metrics['df_topic_keywords'].iloc[np.argmax(topic_probability_scores), :].values.tolist()
    return topic, topic_probability_scores

# write predictions to a file
def write_prediction_to_file(df):
    output = df[['id','chief_complaint_preprocessed','first_diagnosis_preprocessed',
                 'label_candidate','dominant_topic', 'topic_first_10_keywords']]
    filename = 'topic_predicted.csv'
    output.to_csv(filename, index=False)
    print('Topic modeling complete!', filename, 'generated.')