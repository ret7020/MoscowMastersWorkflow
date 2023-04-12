#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from keybert import KeyBERT
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer, HashingVectorizer


# In[2]:


get_ipython().system('pip3 install keybert')


# In[4]:


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


# In[7]:


get_ipython().run_cell_magic('time', '', "keywords = kw_model.extract_keywords('''Supervised learning is the machine learning task of learning a function that\n         maps an input to an output based on example input-output pairs. It infers a\n         function from labeled training data consisting of a set of training examples.\n         In supervised learning, each example is a pair consisting of an input object\n         (typically a vector) and a desired output value (also called the supervisory signal).\n         A supervised learning algorithm analyzes the training data and produces an inferred function,\n         which can be used for mapping new examples. An optimal scenario will allow for the\n         algorithm to correctly determine the class labels for unseen instances. This requires\n         the learning algorithm to generalize from the training data to unseen situations in a\n         'reasonable' way (see inductive bias).\n''', keyphrase_ngram_range=(1, 3))\n")


# In[20]:


complaints = pd.read_csv("./complaints/complaints.csv").dropna().head(3000)


# In[24]:


complaints = complaints.rename(columns={"Consumer complaint narrative": "narrative"})


# In[26]:


from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)


# In[30]:


def f(text):
    from keybert import KeyBERT
    return [i[0] for i in KeyBERT().extract_keywords(text, keyphrase_ngram_range=(1, 3))]


# In[31]:


f(complaints.iloc[0].narrative)


# In[34]:


complaints["narrative_keyngrams"] = complaints["narrative"].parallel_apply(f)


# In[35]:


complaints.head(5)


# In[44]:


all_ngrams = []
complaints["narrative_keyngrams"].apply(lambda x: all_ngrams.extend(x))
#set(all_ngrams)


# In[ ]:


sns.heatmap(complaints, x="")


# In[48]:


import plotly.express as px


# In[47]:


get_ipython().system('pip install plotly')


# In[58]:


tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')


# In[61]:


dtm = tfidf.fit_transform(complaints['narrative_keyngrams'].apply(lambda x: " ".join(x)))


# In[67]:


dtm


# In[72]:


bow = CountVectorizer()
bow_vectors = bow.fit_transform(complaints['narrative_keyngrams'].apply(lambda x: " ".join(x)))


# In[71]:


bow_vectors


# In[ ]:





# In[85]:


hash_vectorizer = HashingVectorizer(n_features=10000,norm=None,alternate_sign=False)
hash_vectorizer.fit(complaints['narrative_keyngrams'].apply(lambda x: " ".join(x)))


# In[86]:


hash_vectors = hash_vectorizer.transform(complaints['narrative_keyngrams'].apply(lambda x: " ".join(x)))


# In[87]:


hash_vectors


# In[91]:


from gensim.corpora.dictionary import Dictionary
from gensim.models.nmf import Nmf
from gensim.models.coherencemodel import CoherenceModel
from operator import itemgetter


texts = complaints['narrative_keyngrams'].apply(lambda x: " ".join(x))
dataset = complaints["narrative_keyngrams"]

dictionary = Dictionary(dataset)


dictionary.filter_extremes(
    no_below=3,
    no_above=0.85,
    keep_n=5000
)


corpus = [dictionary.doc2bow(text) for text in dataset]


topic_nums = list(np.arange(5, 10, 1))

coherence_scores = []

for num in topic_nums:
    nmf = Nmf(
        corpus=corpus,
        num_topics=num,
        id2word=dictionary,
        chunksize=2000,
        passes=5,
        kappa=.1,
        minimum_probability=0.01,
        w_max_iter=300,
        w_stop_condition=0.0001,
        h_max_iter=100,
        h_stop_condition=0.001,
        eval_every=10,
        normalize=True,
        random_state=42
    )
    
    cm = CoherenceModel(
        model=nmf,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    
    coherence_scores.append(round(cm.get_coherence(), 5))

scores = list(zip(topic_nums, coherence_scores))
best_num_topics = sorted(scores, key=itemgetter(1), reverse=True)[0][0]

print(best_num_topics)


# In[92]:


# As we can see: 5 topics


# In[113]:


# from sklearn.decomposition import NMF
# nmf_model = NMF(n_components=5, random_state=40)
# nmf_model.fit(hash_vectors)

# # View the number of features
# len(bow.get_feature_names_out())


# In[129]:


from sklearn.decomposition import NMF
nmf_model = NMF(n_components=5, random_state=40)
nmf_model.fit(dtm)

# View the number of features
len(tfidf.get_feature_names_out())


# In[130]:


single_topic = nmf_model.components_[0]
single_topic.argsort()
top_word_indices = single_topic.argsort()[-10:]
for index in top_word_indices:
    print(tfidf.get_feature_names_out()[index])


# In[131]:


for index,topic in enumerate(nmf_model.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([tfidf.get_feature_names_out()[i] for i in topic.argsort()[-15:]])
    print('\n')


# In[162]:


from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)
fig, axes = plt.subplots(ncols=5, nrows=1)
fig.set_figheight(15)
fig.set_figwidth(15)

for index, topic in enumerate(nmf_model.components_):
    wordcloud = WordCloud(
                            stopwords=stopwords,
                            max_words=40,
                            max_font_size=40, 
                            random_state=42
                        ).generate(" ".join([tfidf.get_feature_names_out()[i] for i in topic.argsort()[-40:]]))
    ax = axes.flatten()[index]
    ax.set_title(f"Topic: {index}")
    ax.imshow(wordcloud)
    ax.axis('off')
plt.tight_layout()


# In[100]:


topic_results = nmf_model.transform(dtm)
topic_results[0].round(2)
topic_results[0].argmax()
topic_results.argmax(axis=1)


# In[101]:


complaints['Topic'] = topic_results.argmax(axis=1)


# In[102]:


complaints.head()


# In[114]:


topic_presenter = complaints.groupby('Topic').head(5)
topic_presenter.sort_values('Topic')


# In[117]:


from sklearn.decomposition import LatentDirichletAllocation

number_of_topics = 5
model = LatentDirichletAllocation(n_components=number_of_topics, random_state=42)


# In[124]:


model.fit(dtm)


# In[125]:


topic_dict = {}
feature_names = tfidf.get_feature_names_out()
no_top_words = 10
for topic_idx, topic in enumerate(model.components_):
    topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                    for i in topic.argsort()[:-no_top_words - 1:-1]]
    topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                    for i in topic.argsort()[:-no_top_words - 1:-1]]
pd.DataFrame(topic_dict)


# In[ ]:


from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda_model.show_topics(formatted=False)

fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()


# In[83]:


# LEGACY


# In[81]:


from transformers import AutoTokenizer, pipeline, TFRobertaModel
import numpy as np
from transformers import AutoTokenizer, pipeline, TFDistilBertModel
from scipy.spatial.distance import cosine
def transformer_embedding(name,inp,model_name):

    model = model_name.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    pipe = pipeline('feature-extraction', model=model, 
                tokenizer=tokenizer)
    features = pipe(inp)
    features = np.squeeze(features)
    return features


roberta_features1 = transformer_embedding('roberta-base', complaints['narrative_keyngrams'].apply(lambda x: " ".join(x)), TFRobertaModel)


# In[288]:


from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift

kmeans = KMeans(n_clusters=3, random_state=42)
# fit the model
kmeans.fit(dtm)

clusters = kmeans.labels_


# In[285]:


dbscan_ = DBSCAN(eps=1, leaf_size=50)
# fit the model
dbscan_.fit(dtm)
clusters = dbscan_.labels_
complaints["dbscan_clusters"] = clusters


# In[297]:


agglomer = AgglomerativeClustering(n_clusters=5)
# fit the model
agglomer.fit(dtm.toarray())
clusters = agglomer.labels_
complaints["agglomer_clusters"] = clusters


# In[301]:


mean_shift = MeanShift(n_jobs=-1)
# fit the model
mean_shift.fit(dtm.toarray())
clusters = mean_shift.labels_
complaints["mean_shift_clusters"] = clusters


# In[304]:


complaints.head(20)


# In[270]:


from sklearn.decomposition import PCA


pca = PCA(n_components=2, random_state=42)
pca_vecs = pca.fit_transform(dtm.toarray())
x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]


# In[186]:


complaints['cluster'] = clusters
complaints['cluster_x0'] = x0
complaints['cluster_x1'] = x1



# In[187]:


def get_top_keywords(n_terms):
    """This function returns the keywords for each centroid of the KMeans"""
    df = pd.DataFrame(dtm.todense()).groupby(clusters).mean() # groups the TF-IDF vector by cluster
    terms = tfidf.get_feature_names_out() # access tf-idf terms
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]])) # for each row of the dataframe, find the n terms that have the highest tf idf score
            
get_top_keywords(10)


# In[188]:


cluster_map = {0: "Cluster_0", 1: "Cluster_1", 2: "Cluster_2", 3: "Cluster_3", 4: "Cluster_4"}


# In[189]:


complaints['cluster'] = complaints['cluster'].map(cluster_map)
complaints.head()


# In[191]:


plt.figure(figsize=(12, 7))
plt.title("TF-IDF + KMeans", fontdict={"fontsize": 18})
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
sns.scatterplot(data=complaints, x='cluster_x0', y='cluster_x1', hue='cluster', palette="viridis")
plt.show()


# In[286]:


plt.figure(figsize=(12, 7))
plt.title("TF-IDF + KMeans", fontdict={"fontsize": 18})
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
sns.scatterplot(data=complaints, x='cluster_x0', y='cluster_x1', hue='dbscan_clusters', palette="viridis")
plt.show()


# In[298]:


plt.figure(figsize=(12, 7))
plt.title("TF-IDF + Agglomer", fontdict={"fontsize": 18})
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
sns.scatterplot(data=complaints, x='cluster_x0', y='cluster_x1', hue='agglomer_clusters', palette="viridis")
plt.show()


# In[307]:


plt.figure(figsize=(12, 7))
plt.title("MeanShift + Agglomer", fontdict={"fontsize": 18})
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
sns.scatterplot(data=complaints, x='cluster_x0', y='cluster_x1', hue='mean_shift_clusters', palette="viridis")
plt.show()


# In[192]:


# Clusterization via vectors from word2vec


# In[196]:


i = 0
list_of_sent = []
for sent in complaints['narrative_keyngrams'].values:
    list_of_sent.append(sent)
list_of_sent[0]


# In[208]:


import gensim

w2v_model = gensim.models.Word2Vec(list_of_sent, vector_size=100, workers=8)


# In[209]:


import numpy as np

sent_vectors = []
for sent in list_of_sent:
    sent_vec = np.zeros(100) 
    cnt_words = 0; 
    for word in sent: 
        try:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    sent_vectors.append(sent_vec)
sent_vectors = np.array(sent_vectors)
sent_vectors = np.nan_to_num(sent_vectors)
sent_vectors.shape


# In[252]:


num_clus = [x for x in range(3,11)]
num_clus


# In[211]:


squared_errors = []
for cluster in num_clus:
    kmeans = KMeans(n_clusters = cluster).fit(sent_vectors) # Train Cluster
    squared_errors.append(kmeans.inertia_) # Appending the squared loss obtained in the list
    
optimal_clusters = np.argmin(squared_errors) + 2 # As argmin return the index of minimum loss. 
plt.plot(num_clus, squared_errors)
plt.title("Elbow Curve to find the no. of clusters.")
plt.xlabel("Number of clusters.")
plt.ylabel("Squared Loss.")
xy = (optimal_clusters, min(squared_errors))
plt.annotate('(%s, %s)' % xy, xy = xy, textcoords='data')
plt.show()

print ("The optimal number of clusters obtained is - ", optimal_clusters)
print ("The loss for optimal cluster is - ", min(squared_errors))


# In[225]:


from sklearn.cluster import KMeans
model2 = KMeans(n_clusters = 4) # n_clusters = optimal_clusters
model2.fit(sent_vectors)


# In[231]:


word_cluster_pred = model2.predict(sent_vectors)
word_cluster_pred_2 = model2.labels_
word_cluster_center = model2.cluster_centers_


# In[256]:


clusters = kmeans.labels_

pca = PCA(n_components=2, random_state=42)

pca_vecs = pca.fit_transform(w2v_model.toarray())

x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]


# In[233]:


complaints['cluster_w2v'] = clusters
complaints['cluster_w2v_x0'] = x0
complaints['cluster_w2v_x1'] = x1


# In[234]:


complaints.head(2)


# In[236]:


plt.figure(figsize=(12, 7))
plt.title("Word2Vec + KMeans + Auto Optimal clusters count", fontdict={"fontsize": 18})
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
sns.scatterplot(data=complaints, x='cluster_w2v_x0', y='cluster_w2v_x1', hue='cluster_w2v', palette="viridis")
plt.show()


# In[237]:


# Clustering with DBSCAN


# In[ ]:


from sklearn.cluster import DBSCAN

minPts = 2 * 100 # 200 соседей

# бинарный поиск для low bound
def lower_bound(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = int(l + (r - l) / 2)
        if nums[mid] >= target:
            r = mid - 1
        else:
            l = mid + 1
    return l

def compute200thnearestneighbour(x, data):
    dists = []
    for val in data:
        dist = np.sum((x - val) ** 2)
        if(len(dists) == 200 and dists[199] > dist): 
            l = int(lower_bound(dists, dist)) 
            if l < 200 and l >= 0 and dists[l] > dist:
                dists[l] = dist
        else:
            dists.append(dist)
            dists.sort()
    return dists[199]


# In[246]:


from tqdm import tqdm

twohundrethneigh = []
for val in tqdm(sent_vectors[:1500]):
    twohundrethneigh.append( compute200thnearestneighbour(val, sent_vectors[:1500]) )
twohundrethneigh.sort()


# In[243]:


plt.figure(figsize=(14,4))
plt.title("Elbow Method for Finding the right Eps hyperparameter")
plt.plot([x for x in range(len(twohundrethneigh))], twohundrethneigh)
plt.xlabel("Number of points")
plt.ylabel("Distance of 200th Nearest Neighbour")
plt.show()


# In[248]:


model = DBSCAN(eps = 0.002, min_samples = minPts, n_jobs=-1)
model.fit(sent_vectors)


# In[308]:


pca = PCA(n_components=2, random_state=42)
pca_vecs = pca.fit_transform(model.toarray())
x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]


# In[309]:


# ML Training


# In[310]:


complaints.head(10)


# In[313]:


complaints["target"] = complaints["Product"].factorize()[0]


# In[314]:


complaints.head(2)


# In[58]:


from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV


# In[322]:


X_train, X_test, y_train, y_test = train_test_split(dtm, complaints.target, test_size=0.25, random_state=42)


# In[316]:


complaints["target"].unique()


# In[57]:


def eval_model(y_test, y_pred, model_name):
    MAP_LABELS = ["0", "1"]
    print(f"CLASSIFICATION REPORT for {model_name}\n")
    print(classification_report(y_test, y_pred, target_names=MAP_LABELS))
    
    # plot confusion matrix of the classifier
    plt.figure(figsize=(10,6))
    plt.title(f"CONFUSION MATRIX for {model_name}\n")
    matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(matrix, annot=True, cbar=None, cmap="Blues", fmt='d', xticklabels=MAP_LABELS, yticklabels=MAP_LABELS)
    plt.show()
    
    return


# In[337]:


from sklearn.naive_bayes import MultinomialNB

model_name = 'NAIVE BAYES'
clf_nb = MultinomialNB(alpha=0.1, fit_prior=True)
get_ipython().run_line_magic('time', '')
clf_nb.fit(X_train, y_train)
y_pred_nb = clf_nb.predict(X_test)


# In[59]:


from sklearn.metrics import confusion_matrix, f1_score, classification_report


# In[338]:


f1_nb = f1_score(y_test, y_pred_nb, average="weighted")
f1_nb


# Кросс валом ищем лучшие гиперпараметры. 10 фолдов

# In[332]:


# Hyperparameter tuning to improve Naive Bayes performance
param_grid_nb = {
    'alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001),
    'fit_prior':[True, False]
}

grid_nb = GridSearchCV(estimator=clf_nb, 
                       param_grid=param_grid_nb,
                       verbose=1,
                       scoring='f1_weighted',
                       n_jobs=-1,
                       cv=10)
grid_nb.fit(X_train, y_train)
print(grid_nb.best_params_)


# In[340]:


eval_model(y_test, y_pred_nb, model_name)


# In[ ]:


# Other models

## Logreg
model_name = 'LOGISTIC REGRESSION'
clf_lr = LogisticRegression(solver='liblinear')
get_ipython().run_line_magic('time', '')
clf_lr.fit(X_train, y_train)
y_pred_lr = clf_lr.predict(X_test)

# # Hyperparameter tuning to improve Logistic Regression performance
# param_grid_lr = {
#     'penalty': ['l1', 'l2','elasticnet', 'none'],
#     'C': [0.001,0.01,0.1,1,10,100],
#     'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
# }

# grid_lr = GridSearchCV(estimator=clf_lr, 
#                        param_grid=param_grid_lr,
#                        verbose=1,
#                        scoring='f1_weighted',
#                        n_jobs=-1,
#                        cv=5)
# grid_lr.fit(X_train, y_train)
# print(grid_lr.best_params_)


## Decision Tree

model_name = 'DECISION TREE'
clf_dt = DecisionTreeClassifier()
get_ipython().run_line_magic('time', '')
clf_dt.fit(X_train, y_train)
y_pred_dt = clf_dt.predict(X_test)

# # Hyperparameter tuning to improve Decision Tree performance
# param_grid_dt = {
#     'criterion': ['gini', 'entropy'],
#     'max_depth' : [5, 10, 15, 20, 25, 30],
#     'min_samples_leaf':[1,5,10,15, 20, 25],
#     'max_features':['auto','log2','sqrt',None],
# }

# grid_dt = GridSearchCV(estimator=clf_dt, 
#                        param_grid=param_grid_dt,
#                        verbose=1,
#                        scoring='f1_weighted',
#                        n_jobs=-1,
#                        cv=5)
# grid_dt.fit(X_train, y_train)
# print(grid_dt.best_params_)


## Random Forest

model_name = 'RANDOM FOREST'
clf_rf = RandomForestClassifier()
get_ipython().run_line_magic('time', '')
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

# # Hyperparameter tuning to improve Random Forest performance
# param_grid_rf = {
#     'n_estimators': [100, 200, 300, 500, 800],
#     'criterion':['gini','entropy'],
#     'max_depth': [10, 30, 40],
#     'min_samples_split': [1, 5, 10],
#     'min_samples_leaf': [1, 5, 10],
#     'max_features': ['log2', 'sqrt', None]    
# }

# grid_rf = RandomizedSearchCV(estimator=clf_rf, 
#                        param_distributions=param_grid_rf,
#                        scoring='f1_weighted',
#                        verbose=1,
#                        n_jobs=-1,
#                        cv=5)
# grid_rf.fit(X_train, y_train)
# print(grid_rf.best_params_)


## SVM
model_name = 'SUPPORT VECTOR MACHINE'
clf_svm = SVC()
get_ipython().run_line_magic('time', '')
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)

# # Hyperparameter tuning to improve SVM performance
# param_grid_svm = {
#     'C': [10, 15],
#     'gamma': ['scale', 0.01],
#     'kernel': ['linear', 'rbf']
# }

# grid_svm = GridSearchCV(estimator=clf_svm, 
#                        param_grid=param_grid_svm,
#                        scoring='f1_weighted',
#                        verbose=1,
#                        n_jobs=-1,
#                        cv=2)
# grid_svm.fit(X_train, y_train)
# print(grid_svm.best_params_)

## Catboost
model_name = 'CATBOOST'
clf_cb = CatBoostClassifier(task_type="GPU",
                           loss_function='MultiClass')
get_ipython().run_line_magic('time', '')
clf_cb.fit(X_train, y_train)
y_pred_cb = clf_cb.predict(X_test)

# # Hyperparameter tuning to improve CatBoost performance
# param_grid_cb = {
#         'depth':[2, 3, 4],
#         'l2_leaf_reg':np.logspace(-20, -19, 3)
# }

# grid_cb = RandomizedSearchCV(estimator=clf_cb, 
#                               param_distributions=param_grid_cb,
#                               scoring='f1_weighted',
#                               verbose=1,
#                               n_jobs=-1,
#                               cv=2)
# grid_cb.fit(X_train, y_train)
# print(grid_cb.best_params_)




# In[ ]:





# In[ ]:





# In[2]:


complaints


# In[39]:


# Catboost only classification
from catboost import CatBoostClassifier, Pool

complaints = pd.read_csv("./complaints/complaints.csv").dropna().head(3000)

def fit_model(train_pool, test_pool, **kwargs):
    model = CatBoostClassifier(
        task_type='GPU',
        iterations=1,
        eval_metric='Accuracy',
        od_type='Iter',
        od_wait=500,
        **kwargs
    )
    return model.fit(
            train_pool,
            eval_set=test_pool,
            verbose=100,
            plot=True,
            use_best_model=True)


# In[42]:


complaints.head()


# In[44]:


# Factorize
complaints["product_id"] = complaints["Product"].factorize()[0]


# In[45]:


# Drop trash
complaints = complaints.drop(["Date received", "Sub-product", "Sub-issue", "Company", "State", "ZIP code", "Tags", "Consumer consent provided?", "Submitted via", "Date sent to company", "Company response to consumer", "Timely response?", "Consumer disputed?", "Complaint ID"], axis=1)


# In[48]:


complaints


# In[29]:


complaints.Product.unique()


# In[47]:


complaints = complaints.drop("Product", axis=1)


# In[49]:


from sklearn.model_selection import train_test_split

train, valid = train_test_split(
    complaints,
    train_size=0.75,
    random_state=42,
    stratify=complaints['product_id'])
y_train, X_train = \
    train['product_id'], train.drop(['product_id'], axis=1)
y_valid, X_valid = \
    valid['product_id'], valid.drop(['product_id'], axis=1)


# In[50]:


X_valid.shape


# In[51]:


train_pool = Pool(
    data=X_train,
    label=y_train,
    text_features=['Issue', 'Consumer complaint narrative', 'Company public response']
)
valid_pool = Pool(
    data=X_valid, 
    label=y_valid,
    text_features=['Issue', 'Consumer complaint narrative', 'Company public response']
)


# In[64]:


# тонкий конфиг катбуста
model = fit_model(
    train_pool, valid_pool,
    learning_rate=0.35,
    tokenizers=[
        {
            'tokenizer_id': 'Sense',
            'separator_type': 'BySense',
            'lowercasing': 'True',
            'token_types':['Word', 'Number', 'SentenceBreak'],
            'sub_tokens_policy':'SeveralTokens'
        }      
    ],
    dictionaries = [
        {
            'dictionary_id': 'Word',
            'max_dictionary_size': '50000'
        }
    ],
    feature_calcers = [
        'BoW:top_tokens_count=10000'
    ]
)


# In[54]:


y_pred_catboost =  model.predict(X_valid)


# In[62]:


from matplotlib import pyplot as plt


# In[63]:


eval_model(y_valid, y_pred_catboost, "Catboost")


# In[ ]:





# In[ ]:





# In[ ]:


#https://kyawkhaung.medium.com/multi-label-text-classification-with-bert-using-pytorch-47011a7313b9


# In[ ]:


def predict_topic(text):
    
    target_names = ["Bank Account services", "Credit card or prepaid card", "Others", "Theft/Dispute Reporting", "Mortgage/Loan"]

    loaded_vec = CountVectorizer(vocabulary=pickle.load(open("count_vector.pkl", "rb")))
    loaded_tfidf = pickle.load(open("tfidf.pkl","rb"))
    loaded_model = pickle.load(open("logreg_model.pkl","rb"))

    X_new_counts = loaded_vec.transform(text)
    X_new_tfidf = loaded_tfidf.transform(X_new_counts)
    predicted = loaded_model.predict(X_new_tfidf)

    return target_names[predicted[0]]

