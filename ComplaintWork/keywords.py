from rake_nltk import Rake
import time
import yake

a = '''
Supervised learning is the machine learning task of learning a function that
         maps an input to an output based on example input-output pairs. It infers a
         function from labeled training data consisting of a set of training examples.
         In supervised learning, each example is a pair consisting of an input object
         (typically a vector) and a desired output value (also called the supervisory signal).
         A supervised learning algorithm analyzes the training data and produces an inferred function,
         which can be used for mapping new examples. An optimal scenario will allow for the
         algorithm to correctly determine the class labels for unseen instances. This requires
         the learning algorithm to generalize from the training data to unseen situations in a
         'reasonable' way (see inductive bias).
'''
r = Rake()
print("RAKE")
s = time.time()
r.extract_keywords_from_text(a)
print(time.time() - s)
print(r.get_ranked_phrases())
print("YAKE")
s = time.time()
yake.KeywordExtractor(lan="en", n=3, dedupLim=0.4, top=4, features=None).extract_keywords(a)
print(time.time() - s)
