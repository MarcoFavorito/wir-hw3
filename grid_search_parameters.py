import utils
from utils import stemming_tokenizer
from nltk.corpus import stopwords
from utils import stemming_tokenizer, porter_stemming_tokenizer, lancaster_stemming_tokenizer
# generate ngrams
min_start = 1
max_start = 5
min_end = 1
max_end = 5
ngrams = [(x, y) for x in range(min_start, max_start) for y in range(min_end, max_end) if x <= y]

# parameter of the vectorizer for each model
vectorizer_params = {
	'kNN': {
		'analyzer' : ['word'],
		'tokenizer':[stemming_tokenizer],
		'ngram_range': [(1,2)],
		'stop_words':[stopwords.words('english')],
		'max_df':[1.0],
		'min_df':[1, 2],
		'binary':[False],
		'use_idf':[True],
		'smooth_idf': [True],
		'sublinear_tf':[True]
	},
	'MultinomialNB':{
		'analyzer' : ['word'],
		'tokenizer':[stemming_tokenizer],
		'ngram_range': [(1,2)],
		'stop_words':[stopwords.words('english')],
		'max_df':[0.175],
		'min_df':[1],
		'binary':[False],
		'use_idf':[True],
		'smooth_idf':[True],
		'sublinear_tf':[False],
	},

	'LinearSVC': {
		'analyzer': ['word'],
		'tokenizer': [stemming_tokenizer],
		'ngram_range': [(1, 2)],
		'stop_words': [stopwords.words('english')],  # Only applies if analyzer == 'word'.
		'max_df': [1.0],
		'min_df': [2],
		'binary': [False],
		'use_idf': [True],
		'smooth_idf': [True],
		'sublinear_tf': [True],
	}

}

# TruncatedSVD
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
svd_params = {
	'kNN': {
		'n_components': range(2, 10),
	},
	'MultinomialNB':{
		# 'svd__n_components': [4],
	},

	'LinearSVC': {
		# 'svd__n_components': [4],
	}

}


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
	# 'norm': ['l1', 'l2'],  # not so relevant, I think
	# 'analyzer':['word', 'char'], # analysis by word or by char
	'tokenizer':[stemming_tokenizer], # notice: it applies only if analyazer='word'
	'stop_words':[stopwords.words('english')], # Only applies if analyzer == 'word'.
	'ngram_range':[(1,2)],
	# 'max_df':[1.0, 0.5, 0.4, 0.3, 0.2, 0.1], # is ignored if vocabulary is not None.
	'min_df':[2], # int: absolute count of documents; float: proportion of documents.  is ignored if vocabulary is not None.
	# 'max_features':[None], # maybe it is useful. is ignored if vocabulary is not None.
	# 'vocabulary':[None],
	# 'binary':[False],
	# 'use_idf':[True, False],
	'sublinear_tf':[True] #Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).


}

max_k_neighbor = 8
min_k_neighbor = 3
kNN_params = {
	# 'algorithm':[‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’], # default is auto, let it free to chose
	# 'leaf_size':[30],
	# 'metric_params':[None],
	# 'n_jobs':[1]
	'n_neighbors': range(min_k_neighbor, max_k_neighbor, 2),
	# 'weights': ['uniform', 'distance'],
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
	# 'kernel': ['poly'],
	# 'C': [2**c for c in range(-2,5,2)],
	# 'gamma': [2**c for c in range(-3,4,2)],
	# 'coef0' : [1.0, .5, 0.0],
	# 'degree':[1, 2, 3],

	# For 'rbf' kernel: C, gamma
	# 'kernel': ['rbf'],
	# 'C': [2**c for c in range(-2,10,2)],
	# 'gamma': [2**c for c in range(-7,4,2)],

	# For 'sigmoid' kernel: C, gamma, coef0
	# 'kernel': ['rbf'],
	# 'C': [2**c for c in range(-2,10,2)],
	# 'gamma': [2**c for c in range(-7,4,2)],
	# 'coef0' : [1.0, .5, 0.0]

}


linearsvc_params = {
	# 'fit_intercept':[False],
	# 'intercept_scaling':[1],
	# 'class_weight':[None],
	# 'verbose':[1],
	# 'random_state':[None],
	# 'max_iter':[1000],
	# 'multi_class':['ovr'],
	# 'dual':[True],

	# 'penalty':['l2', 'l1'],
	'loss':['squared_hinge'],

	'tol':[0.0000001],
	# 'C': [0.125, 0.5, 0.7, 1.0, 2, 4, 8], # the lower, the better the parameter generalize or avoid overfitting or tolerate some misclassified training example
	'C': [0.7]

}

sgdclassifier_params = {
	# 'eta0':[0.0],
	# 'power_t':[0.5],
	# 'class_weight':[None],
	# 'warm_start':[False],
	# 'average':[False],
	# 'fit_intercept': [True],
	# 'shuffle':[True],
	# 'verbose':[0],
	# 'n_jobs':[1],
	# 'random_state':[None],
	# 'learning_rate':['optimal'],
	'loss':['hinge'],
	'penalty':['l1', 'l2'], #'l2', 'l1',
	'alpha':[0.001, 0.0001],
	# 'l1_ratio':[0.15, 0.30], # only if penalty: 'elasticnet'
	'n_iter':[5],
	'epsilon':[0.1, 0.01],


}