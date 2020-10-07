# Topic Modeling, from EMR to underlying conditions
This project uses information from electronic medical record (EMR) to generate a prediction of the underlying conditions of the patients
Latent Dirichlet allocation (LDA) was used in the project to generate the topics

* Input: For every patient there is a text file (.txt) with fields extracted from EMR, as well as an annotation file (.ann) with name entities extracted from the text file.  
* Output: The program generates a csv file that listed the dominant topic and keywords from the topic for each patient.  

<table>
  <tr>
    <th>Files</th>
    <th>Notes</th>
  </tr>
  <tr>
    <td>main.py</td>
    <td>Generate topic modeling for a folder of files</td>
  </tr>
  <tr>
    <td>helpers.py</td>
    <td>support functions for main.py</td>
  </tr>
  <tr>
    <td>topic_modeling_explore.ipynb</td>
    <td>Stepwise exploritary analysis to discover the data, EDA, and find the best parameters</td>
  </tr>
</table>
<br>

## 1. Overall observations of LDA models using different fields in the text files
The text files have a few fields that are good candidates for topic modeling. The following content have been tried for LDA modeling:
* The full text file
* Chief Complaint
* History of Present Illness
* Discharge Diagnosis

Observations: 
* Grid search generated the optimal number of topics being 10. It does not seem to be enough topics to cover the complexity of the data, even though it is algorithmically optimal. 
* In addition there are problems when using the content of each field:
* When using either the full text file, or history of present illness, the topic keywords were very unspecific. Probably due to the many non-relevant words in a large body of text
* When using chief complaint, the words were a lot of times only symptoms (such as fever, chest pain), but not the underlying conditions (infection, myocardio infarction).
* When using discharge diagnosis, there were many secondary diagnosis that shadowed the main underlying condiiton.

Based on the observation, I extracted first diagnosis from discharge diagnosis. It in some cases were the same as chief complaints, and in other cases were the names of the underlying conditions. Since diagnosis was blank for some patients, I substituted it with chief complaints, then service (if it is not medicine). I also tried to substitude it with addendum, but addendum was too long compaired to other first diagnosis, chief complaint and service.

I experimented with setting the number of topics = 20, 30, 40. 
* 20 topics, there was heavy overlap of conditions in each topic. Many documents fell in topic 0.
* 30 topics, the overlap was better. The distribution of topics between documents were a lot more balanced.
* 40 topics, some topics did not really have any meaning at more. Some topics only had a couple files. 
Thus 30 was chosen as the number of topics for the main model.

## 2. Data overview

## Afterthoughts
Within a contrained amount of time, I did extensive research on topic modeling, LDA, embedding, deep learning NLP. I concluded that for topic modeling per se, which is to find out what was mentioned most, the bag-of-word approach by LDA was sufficient. It was especially true when no prior labels were given, and LDA was good for label discovery.

The real world data is messy. In order to extract the right content from the file I had gone through several iterations. I also discovered that even though the majority of files shared a similar structure, a small amount of the files seemed to be generated from a different system, with slightly different keywords and format. 

I originally thought five days were enough time because the models would take much less time to run compared to some CNN models on images. But now I realized I hardly scratched the surface of the problem. Below are what to improve on:

* Compare LDA and NMF(Non-negative Matrix Factorization) 
* Take advantage of the annotation files
* Introducing more stop words when modeling large body of text to reduce the probability of unspecific words being picked up as part of the condition. However, this needs to be done with caution, because the appearance of nonspecific words may contribute to certain underlying conditions.
* I feel there are a lot of complex underlying condition in the 303 cases, and if there are 30 or more topics, 303 case is simply insufficient to automatically generate them. If this is the only data we have, I would recommend mannual label the data based on chief complaint and diagnosis. Then run classification models using the labels. LDA and NMF can be used for the classification. In addition, random forest, k-nearest neighbor will work too afer word embedding. Deep learning CNN and LSTM can be used as well.
* Word embedding is a very interesting field that I didn't have time to get in. (I spent the time, but couldn't figure out how to make it work with LDA which can only count words). There are general word embedding such as GoogleNews-vectors-negative300.bin and glove.6B. There are also more specific resource built in the medical space. 
* sparkNLP has a clinical module which is worth looking into.

