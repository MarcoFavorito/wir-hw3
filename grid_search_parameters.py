import utils
from utils import stemming_tokenizer
from nltk.corpus import stopwords

# generate ngrams
min_start = 1
max_start = 2
min_end = 1
max_end = 3
ngrams = [(x, y) for x in range(min_start, max_start) for y in range(min_end, max_end) if x <= y]



tfIdf_params = {
	# Parametri SVM, quelli #### sono quelli che non erano commentati.
	# 'input':[u'content'],
	# 'encoding':[u'utf-8'],
	# 'decode_error':[u'strict'],
	# 'strip_accents':[None],
	# 'lowercase':[True],
	# 'preprocessor':[None],
	# 'token_pattern':[u'(?u)\b\w\w+\b'],
	# 'dtype':[np.int64],
	# 'smooth_idf':[True], # prevents zero divisions
	# 'norm':['l1', 'l2'], # not so relevant, I think
	#### 'tokenizer':[None, stemming_tokenizer], # notice: it applies only if analyazer='word'
	#### 'analyzer':['word', 'char'], # analysis by word or by char
	#### 'stop_words':[None, stopwords.words('english')], # Only applies if analyzer == 'word'.
	#### 'ngram_range':ngrams,
	# 'max_df':[1.0], # is ignored if vocabulary is not None.
	# 'min_df':[1,2], # int: absolute count of documents; float: proportion of documents.  is ignored if vocabulary is not None.
	# 'max_features':[None], # maybe it is useful. is ignored if vocabulary is not None.
	# 'vocabulary':[None],
	#### 'binary':[True, False],
	# 'use_idf':[True],
	#### 'sublinear_tf':[False, True] #Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

	# Parametri usati per MultinomialNB
	# # 'smooth_idf':[True], # prevents zero divisions
	# # 'norm':['l1', 'l2'], # not so relevant, I think
	# 'tokenizer':[None, stemming_tokenizer], # notice: it applies only if analyazer='word'
	# # 'analyzer':['word', 'char'], # analysis by word or by char
	# 'stop_words': [stopwords.words('english')], # Only applies if analyzer == 'word'.
	# 'ngram_range': [(1, x) for x in range(1, 3)],
	# 'max_df': [0.175], # is ignored if vocabulary is not None.
	# 'min_df': range(0, 6), # int: absolute count of documents; float: proportion of documents.  is ignored if vocabulary is not None.
	# # 'max_features':[None], # maybe it is useful. is ignored if vocabulary is not None.
	# # 'vocabulary':[None],
	# 'binary':[True, False],
	# # 'use_idf':[True],
	# 'sublinear_tf':[False, True] #Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

	# Parametri per LinearSVC
	### 'tokenizer':[stemming_tokenizer], # notice: it applies only if analyazer='word'
	### 'stop_words':[stopwords.words('english')], # Only applies if analyzer == 'word'.
	### 'ngram_range':[(1,2)],
	# 'max_df':[1.0, 0.95], # is ignored if vocabulary is not None.
	### 'min_df':range(1,3), # int: absolute count of documents; float: proportion of documents.  is ignored if vocabulary is not None.
	# 'max_features':[None], # maybe it is useful. is ignored if vocabulary is not None.
	# 'vocabulary':[None],
	# 'binary':[True, False],
	# 'use_idf':[True, False],
	### 'sublinear_tf':[True] #Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

	# Parametri per kNN
	# 'tokenizer': [None, stemming_tokenizer], # [None, stemming_tokenizer],
	# 'analyzer': ['word', 'char'],
	# 'analyzer': ['char'],
	'stop_words': [stopwords.words('english')],
	'max_df': [0.2],
	# 'min_df': range(0, 3),
	'binary': [True],
	# 'use_idf': [True, False],
	'smooth_idf': [False],
	'sublinear_tf': [True],
	'ngram_range': [(1,4)],
	# 'ngram_range': [(3, x) for x in range(3, 10)],

}

max_k_neighbor = 10
min_k_neighbor = 3
kNN_params = {
	# 'algorithm':[‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’], # default is auto, let it free to chose
	# 'leaf_size':[30],
	# 'metric_params':[None],
	# 'n_jobs':[1]
	'n_neighbors': range(min_k_neighbor, max_k_neighbor, 2),
	'weights': ['uniform', 'distance'],
	# 'metric': ['minkowski', 'euclidean'], # probably it is not so relevant
	# 'p':[2], # the same for metric.

}

mnbc_params = {
	'alpha': [0.78, 0.775, 0.785],
	'fit_prior': [False]
}


svc_params = {
	# 'shrinking':[True],
	# 'probability':[False],
	# 'cache_size':[1000],
	# 'class_weight':[None],
	# 'verbose':[False],
	# 'max_iter':[-1],
	# 'decision_function_shape':[None],
	# 'random_state':[None]
	# 'tol':[0.001], # Tolerance for stopping criterion.

	# See details on kernel at: http://scikit-learn.org/stable/modules/svm.html#kernel-functions
	# In general, all of the following kernel perform well. (on avg. 97.5%)
	# the 'linear' one is the simplest, and is good.

	# For linear kernel: Only C matters
	'kernel': ['linear'],
	'C': [2**c for c in range(-2,5,1)],

	# For 'poly' kernel: C, gamma, degree and coef0. 'linear' is a special case of 'poly'
	# 'kernel': ['poly'],  # linear is very good! the alternatives 'rbf', 'sigmoid' not very good
	# 'C': [2**c for c in range(-2,5,2)],
	# 'gamma': [2**c for c in range(-3,4,2)],
	# 'coef0' : [1.0, .5, 0.0],
	# 'degree':[1, 2, 3],

	# For 'rbf' kernel: C, gamma
	# 'kernel': ['rbf'],  # linear is very good! the alternatives 'rbf', 'sigmoid' not very good
	# 'C': [2**c for c in range(-2,10,2)],
	# 'gamma': [2**c for c in range(-7,4,2)],

	# For 'sigmoid' kernel: C, gamma, coef0
	# 'kernel': ['rbf'],  # linear is very good! the alternatives 'rbf', 'sigmoid' not very good
	# 'C': [2**c for c in range(-2,10,2)],
	# 'gamma': [2**c for c in range(-7,4,2)],
	# 'coef0' : [1.0, .5, 0.0]

}


linearsvc_params = {
	# 'fit_intercept':[True],
	# 'intercept_scaling':[1],
	# 'class_weight':[None],
	# 'verbose':[1],
	# 'random_state':[None],
	# 'max_iter':[1000]
	# 'multi_class':['ovr'],

	# 'penalty':['l2'],
	# 'loss':['squared_hinge'],
	# 'dual':[True],
	# 'tol':[0.0001],
	'C':[0.8, 0.9, 1.0, 1.1], # the lower, the better the parameter generalize or avoid overfitting or tolerate some misclassified training example

}

sgdclassifier_params = {
	'loss':['hinge', 'perceptron', 'squared_hinge'],
	'penalty':['l2'],
	'alpha':[0.0001],
	'l1_ratio':[0.15],
	'fit_intercept':[True],
	'n_iter':[5],
	'shuffle':[True],
	'verbose':[0],
	'epsilon':[0.1, 0.01, 0.001],
	'n_jobs':[1],
	'random_state':[None],
	'learning_rate':['optimal'],
	'eta0':[0.0],
	'power_t':[0.5],
	'class_weight':[None],
	'warm_start':[False],
	'average':[False]
}
