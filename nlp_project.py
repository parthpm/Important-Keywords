import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import PyPDF2
import string


# creating a pdf file object
pdfFileObj = open('JavaBasics-notes.pdf', 'rb')

pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
pageObj = pdfReader.getPage(0)
text=pageObj.extractText()
print(text)
text=[]
for i in range(22):
    pageObj=pdfReader.getPage(i)
    text.append(pageObj.extractText())
print(len(text))

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

text_df=pd.DataFrame(data=text,columns=['Page'],index=range(22))
print(text_df.head())

text_df['Page'].head().apply(text_process)
print(text_df.head())
# print(text)
# clean_text=text_process(text[:])
#
# print(type(clean_text))
# print(clean_text)

from sklearn.feature_extraction.text import CountVectorizer
bow_transformer=CountVectorizer(analyzer=text_process).fit(text_df['Page'])

print(len(bow_transformer.vocabulary_))
feature_names=bow_transformer.get_feature_names()
print(len(feature_names))
#
# p2=text_df['Page'][1]
# print(p2)
#
# bow2=bow_transformer.transform([p2])
# print(bow2)
# print(bow2.shape)

messages_bow=bow_transformer.transform(text_df['Page'])
print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf_matrix=tfidf_transformer.transform(messages_bow)

print(tfidf_matrix.shape)
print(type(tfidf_matrix))
print(tfidf_matrix)
print(feature_names)

# print(tfidf_matrix.head())

matrix=pd.DataFrame(tfidf_matrix.todense())
print(matrix)
matrix=matrix.transpose()
print(matrix)

matrix=matrix.values
print(matrix)
print(np.sort(matrix,axis=0))

mat=pd.DataFrame(data=np.sort(matrix,axis=0),index=range(1068))
print(mat)





#
# matrix.sort_values(by=list(range(1068)) ,ascending=[False*1068],inplace=True)
# print(matrix)