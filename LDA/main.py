# This script makes a topic prediction using the text files in the folder 'training_20180910'
# It generate csv file "topic_predicted"

import helpers

# import all the files and build a data frame
df = helpers.build_df('training_20180910')

# extract different components
df = helpers.extract_by_keyword_df(df, 'Chief Complaint')
df = helpers.extract_first_diagnosis_df_clean(df)
df = helpers.extract_by_keyword_df(df, 'Service:')
# compose the label_candidate for making prediction
df = helpers.get_label_candidate(df)

# build the default stop_words
stop_words = helpers.default_stop_words()

# conduct the 30-components topic modeling using label_candidate
lda_output = helpers.do_vectorization_lda(df['label_candidate'], n_components=30, stop_words=stop_words, max_features=10000)

# create metrics and predictions
lda_metrics = helpers.evalute_lda(lda_output)

# add prediction to the original dataframe
df = helpers.add_topic_keywords_to_df(df, lda_metrics)

# write output to a file
helpers.write_prediction_to_file(df)