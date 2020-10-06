# Topic Modeling, from EMR to underlying problem
This project uses information from electronic medical record (EMR) to generate a prediction of the underlying problem of the patients
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
