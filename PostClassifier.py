import pandas as pd
import praw

import nltk
from nltk.stem.snowball import SnowballStemmer

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


#Class for organizing posts into accessible objects
class post:
    def __init__(self, selftext, title, score, num_comments, subreddit):
        self.selftext = selftext
        self.title = title
        self.score = score
        self.num_comments = num_comments
        self.subreddit = subreddit


nltk.download('stopwords')
stemmer = SnowballStemmer("english", ignore_stopwords=True)

#Class for creating a stemmed count vectorizer
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


submission_list_aita = []
submission_list_pr = []
POST_LIMIT = 500

# Grab data from reddit
reddit = praw.Reddit(client_id='15yiyv0HjzLQZw', client_secret="iiB2UjyMDczHMVSYkkgYaS53fY4",
                     password='#Thisisagreatpassword1', user_agent='AnotherNormalRobot',
                     username='AnotherNormalRobot')

# Iterate through the posts and pick out what we want from them
# legaladvice, amitheasshole
for submission in reddit.subreddit('amitheasshole').hot(limit=POST_LIMIT):
    post_obj = vars(submission)
    submission_list_aita.append(post(post_obj['selftext'], post_obj['title'], post_obj['score'],
                                post_obj['num_comments'], post_obj['subreddit'].display_name))
# nostupidquestions, prorevenge
for submission in reddit.subreddit('prorevenge').hot(limit=POST_LIMIT):
    post_obj = vars(submission)
    submission_list_pr.append(post(post_obj['selftext'], post_obj['title'], post_obj['score'],
                                post_obj['num_comments'], post_obj['subreddit'].display_name))


# Here we set up our object sets for each data set
train_object_set = [submission_list_aita.pop(random.randrange(len(submission_list_aita)))
                    for _ in range(int(POST_LIMIT * .5))]
dev_object_set = [submission_list_aita.pop(random.randrange(len(submission_list_aita)))
                  for _ in range(int(POST_LIMIT * .25))]
test_object_set = [submission_list_aita.pop(random.randrange(len(submission_list_aita)))
                   for _ in range(int(POST_LIMIT * .25))]

train_object_set.extend([submission_list_pr.pop(random.randrange(len(submission_list_pr)))
                     for _ in range(int(POST_LIMIT * .5))])
dev_object_set.extend([submission_list_pr.pop(random.randrange(len(submission_list_pr)))
                   for _ in range(int(POST_LIMIT * .25))])
test_object_set.extend([submission_list_pr.pop(random.randrange(len(submission_list_pr)))
                    for _ in range(int(POST_LIMIT * .25))])

dev_text_bag, dev_answer_set = [], []
train_text_bag, train_answer_set = [], []
test_text_bag, test_answer_set = [], []

# Next we parse our object set and add the relevant features to our respective text bags
for obj in dev_object_set:
    dev_text_bag.append(obj.selftext)
    dev_answer_set.append(obj.subreddit)

for obj in train_object_set:
    # train_text_bag.append(obj.title + ' ' + obj.selftext)
    train_text_bag.append(obj.selftext)
    train_answer_set.append(obj.subreddit)

for obj in test_object_set:
    # test_text_bag.append(obj.title + ' ' + obj.selftext)
    test_text_bag.append(obj.selftext)
    test_answer_set.append(obj.subreddit)


# Specifying stop words
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(['aita', 'wibta', 'amita', 'revenge', 'tl', 'dr'])

# Setting up our count vectorizers
dev_count_vect = StemmedCountVectorizer(analyzer='word', stop_words=stopwords)
test_count_vect = StemmedCountVectorizer(analyzer='word', stop_words=stopwords)
train_count_vect = StemmedCountVectorizer(analyzer='word', stop_words=stopwords)

# Setting up our tfidf transformers
dev_tfidf_transformer = TfidfTransformer()
test_tfidf_transformer = TfidfTransformer()
train_tfidf_transformer = TfidfTransformer()

# Building a pipeline for our test SVC classifier
""" Support Vector Classifier"""
text_clf_test_svm = Pipeline(
    [('vect_test_svm', test_count_vect),
    ('tfidf_test_svm', test_tfidf_transformer),
    ('clf_test_svm', SGDClassifier()),
])

# Fit and predict
text_clf_test_svm.fit(test_text_bag, test_answer_set)
predicted_svm_test = text_clf_test_svm.predict(test_text_bag)

# Printing out the accuracy of the prediction
print("SVC Test Set: ", np.mean(predicted_svm_test == test_answer_set))

# Building a pipeline for our train SVC classifier
text_clf_train_svm = Pipeline(
    [('vect_train_svm', train_count_vect),
    ('tfidf_train_svm', train_tfidf_transformer),
    ('clf_train_svm', SGDClassifier()),
])

# Fit and predict
text_clf_train_svm.fit(train_text_bag, train_answer_set)
predicted_svm_train = text_clf_train_svm.predict(train_text_bag)

# Printing out the accuracy of the prediction
print("SVC Train Set: ", np.mean(predicted_svm_train == train_answer_set))
""" End support Vector Classifier"""

""" Random Forest Classifier"""
# Building a pipeline for our test RF classifier
text_clf_test_rf = Pipeline(
    [('vect_test_rf', test_count_vect),
    ('tfidf_test_rf', test_tfidf_transformer),
    ('clf_test_rf', RandomForestClassifier()),
])

# Fit and predict
text_clf_test_rf.fit(test_text_bag, test_answer_set)
predicted_rf_test = text_clf_test_rf.predict(test_text_bag)

# Printing out the accuracy of the prediction
print("RF Test Set: ", np.mean(predicted_rf_test == test_answer_set))

# Building a pipeline for our train RF classifier
text_clf_train_rf = Pipeline(
    [('vect_train_rf', train_count_vect),
    ('tfidf_train_rf', train_tfidf_transformer),
    ('clf_train_rf', RandomForestClassifier()),
])

# Fit and predict
text_clf_train_rf.fit(train_text_bag, train_answer_set)
predicted_rf_train = text_clf_train_rf.predict(train_text_bag)

# Printing out the accuracy of the prediction
print("RF Train Set: ", np.mean(predicted_rf_train == train_answer_set))
""" End Forest Classifier"""


#Code used for feature rankings
# X_train_vec = train_count_vect.fit_transform(train_text_bag)
#
# new_train_df = pd.DataFrame()
#
# for i, col in enumerate(train_count_vect.get_feature_names()):
#     new_train_df[col] = pd.Series(X_train_vec[:, i].toarray().ravel())
#
# rf = RandomForestClassifier()
# rf.fit(X_train_vec, train_answer_set)
#
# importances = rf.feature_importances_
# print(importances)
# print(train_count_vect.get_feature_names())
#
# std = np.std([tree.feature_importances_ for tree in rf.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]
#
# # Print the feature ranking
# print("Feature ranking:")
#
# for f in range(0,20): # X_train_vec.shape[1]
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]),
#           new_train_df.iloc[:,indices[f]].name)

