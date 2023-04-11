#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install yake pandarallel pandas matplotlib tqdm seaborn')


# In[8]:


get_ipython().system('pip install wordcloud')


# In[9]:


# Check KEYBERT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer
import re
import gc
import swifter
import yake
import wordcloud
from wordcloud import WordCloud, STOPWORDS
from pandarallel import pandarallel # works bad on windows
#from multi_rake import Rake


# Setup tqdm and parallel
tqdm.pandas()
pandarallel.initialize(progress_bar=True)


# Task A (cleaning) + Task B

# In[11]:


complaints = pd.read_csv("./data/train.csv")


# In[12]:


complaints.head(10)


# Clear dataset from unneeded columns

# In[5]:


complaints = complaints.drop(["Date received", "Sub-product", "Issue", "Sub-issue", "Company public response", "Company", "State", "ZIP code", "Tags", "Consumer consent provided?", "Submitted via", "Date sent to company", "Company response to consumer", "Timely response?", "Consumer disputed?"], axis=1)


# In[6]:


# Rename columns to convinient
complaints = complaints.rename(columns={"Consumer complaint narrative": "narrative"})


# In[7]:


# Drop nan
complaints = complaints.dropna()
complaints.isna().sum()


# In[11]:


complaints.Product = pd.Categorical(complaints.Product)
complaints['class'] = complaints.Product.cat.codes


# In[12]:


complaints.head(5)


# In[17]:


# Columns description table
# Product - Target class name (not used for training)
# narrative - Original text of complaint
# Complaint id - id of complaint (will be usefull for final mapping, this is not a feature)
# class - mapping of Prduct to categorial feature (use for training)


# In[18]:


# Get class distribution (and verify that class to complaint id mapping correct)
complaints.groupby('Product').count().plot(kind='bar')


# In[15]:


get_ipython().system('mkdir processed')


# In[16]:


# Checkpoint of first data preparation stage
with open("./processed/clean_0.csv", "wb") as fd:
    pickle.dump(complaints, fd)


# In[20]:


# Save each row text words count (messy splitted, not tokens)
words_statistic = []
for i in tqdm(complaints['narrative'].iloc):
    words_statistic.append(len(i.split()))
words_statistic = np.array(words_statistic)


# In[22]:


print("The average number of words in a document is: {}.".format(np.mean(words_statistic)))
print("The minimum number of words in a document is: {}.".format(min(words_statistic)))
print("The maximum number of words in a document is: {}.".format(max(words_statistic)))
print("Lower then 5 words documents: {}".format(sum(words_statistic <= 5))) 


# In[32]:


# Distribution graph
fig, ax = plt.subplots(figsize=(15,6))
ax.set_title("Distribution of number of words", fontsize=16)
ax.set_xlabel("Number of words")
ax.set_xticks(range(0, 6001, 500))
sns.histplot(words_statistic, bins=50, ax=ax, kde=True)


# In[35]:


pivot_data = complaints.copy()
pivot_data["words_cnt"] = words_statistic
pd.pivot_table(pivot_data, values="words_cnt", index=['Product'], aggfunc={'words_cnt': [np.mean, min, max,np.std]})


# This table visualize mean/max/min/std for each class (product). Here we can see that there is a normal mean words for each class in document

# In[98]:


# Clear ram

gc.collect()
del pivot_data
del words_statistic
gc.collect()


# In[51]:


get_ipython().system('pip install pyspellchecker')


# In[97]:


stop_words = stopwords.words('english')
stop_words.extend(["xxxx", "xxx", "xxxxxxxx", "xx"]) # placeholders for names and other sensetive info
spell = SpellChecker()
wordnet_lemmatizer = WordNetLemmatizer()
punct_tokenizer = RegexpTokenizer(r"\w+")

def spell_correct(text_tokens):
    return [spell.correction(word) for word in text_tokens]

def preprocess_text(text: str):
    # Mutable function
    text = text.lower() # all text to lower case
    text = " ".join(text.split()) # remove extra spaces
    text_tokens = word_tokenize(text) # tokenize
    text_tmp = " ".join(text_tokens)
    text_tmp = re.sub(r'\d+', '', text_tmp) # remove numbers
    text_tokens = punct_tokenizer.tokenize(text_tmp) # remove punct
    text_tokens = [wordnet_lemmatizer.lemmatize(text_t) for text_t in text_tokens if text_t not in stop_words] # remove stop words and lematize them
    #text = spell_correct(text_tokens) # fix spell errors
    return text_tokens

# Example work
_ = preprocess_text(complaints.iloc[0].narrative)


# In[101]:


complaints["tokens"] = complaints["narrative"].parallel_apply(preprocess_text)


# In[111]:


# Multiprocessing apply with dask backends
complaints["tokens"] = complaints['narrative'].swifter.progress_bar(True).apply(lambda x: preprocess_text(x))
#complaints['narrative'].progress_apply(lambda x: x.split())


# In[114]:


complaints.head(5)


# Find ngrams

# In[117]:


# Checkpoint of second stage: clean tokens
with open("./processed/tokenized_1.csv", "wb") as fd:
    pickle.dump(complaints, fd)


# In[125]:


stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='lemonchiffon',
                          stopwords=stopwords,
                          max_words=40,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(complaints['tokens']))

fig = plt.figure(figsize=(10,7))
plt.title("Wordcloud: top 40")
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[ ]:





# Show top 40 words from keywords

# In[6]:


with open("./processed/tokenized_1.csv", "rb") as fd:
    complaints = pickle.load(fd)


# In[7]:


def kw_extract_parallel(text):
    import yake
    return yake.KeywordExtractor(lan="en", n=3, dedupLim=0.4, top=4, features=None).extract_keywords(" ".join(text))


# In[8]:


complaints["key_words"] = complaints["tokens"].parallel_apply(kw_extract_parallel)


# In[9]:


complaints


# In[ ]:





# In[ ]:


# kw_extractor = yake.KeywordExtractor()
# language = "en"
# dedup_threshhold = 0.4
# numOfKeywords = 4
# # Init extractors
# extractor_n_1 = yake.KeywordExtractor(lan=language, n=1, dedupLim=dedup_threshhold, top=numOfKeywords, features=None)
# extractor_n_2 = yake.KeywordExtractor(lan=language, n=2, dedupLim=dedup_threshhold, top=numOfKeywords, features=None)
# extractor_n_3 = yake.KeywordExtractor(lan=language, n=3, dedupLim=dedup_threshhold, top=numOfKeywords, features=None)
# # summary is 9 key phrases(3 per each extractor)
# #print(extractor_n_3.extract_keywords(" ".join(complaints.iloc[0].tokens)))

# #def extract_key(text):
# #    keywords = extractor.extract_keywords(text)
# #    return [i[0] for i in keywords]

# #complaints["key_words"] = complaints["narrative"].swifter.progress_bar(True).apply(lambda x: [i[0] for i in extractor_n_3.extract_keywords(" ".join(x))])
# complaints["key_words"] = complaints["tokens"].swifter.progress_bar(True).allow_dask_on_strings(True).force_parallel(True).apply(lambda x: [i[0] for i in extractor_n_3.extract_keywords(" ".join(x))])


# In[ ]:


# keybert
from keybert import KeyBERT

kw_model = KeyBERT()
  keywords = kw_model.extract_keywords('''Supervised learning is the machine learning task of learning a function that
         maps an input to an output based on example input-output pairs. It infers a
         function from labeled training data consisting of a set of training examples.
         In supervised learning, each example is a pair consisting of an input object
         (typically a vector) and a desired output value (also called the supervisory signal).
         A supervised learning algorithm analyzes the training data and produces an inferred function,
         which can be used for mapping new examples. An optimal scenario will allow for the
         algorithm to correctly determine the class labels for unseen instances. This requires
         the learning algorithm to generalize from the training data to unseen situations in a
         'reasonable' way (see inductive bias).
  ''', keyphrase_ngram_range=(1, 3))

