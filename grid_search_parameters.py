import utils
from utils import stemming_tokenizer
from nltk.corpus import stopwords

# generate ngrams
min_start = 1
max_start = 2
min_end = 1
max_end = 5
ngrams = [(x, y) for x in range(min_start, max_start) for y in range(min_end, max_end) if x < y]



tfIdf_params = {
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

	# 'smooth_idf':[True], # prevents zero divisions
	# 'norm':['l1', 'l2'], # not so relevant, I think
	# 'tokenizer':[None, stemming_tokenizer], # notice: it applies only if analyazer='word'
	# 'analyzer':['word', 'char'], # analysis by word or by char
	'stop_words': [stopwords.words('english')], # Only applies if analyzer == 'word'.
	'ngram_range': [(1, x) for x in range(1, 3)],
	'max_df': [0.175], # is ignored if vocabulary is not None.
	'min_df': range(0, 6), # int: absolute count of documents; float: proportion of documents.  is ignored if vocabulary is not None.
	# 'max_features':[None], # maybe it is useful. is ignored if vocabulary is not None.
	# 'vocabulary':[None],
	'binary':[True, False],
	# 'use_idf':[True],
	'sublinear_tf':[False, True] #Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
}

max_k_neighbor = 10
min_k_neighbor = 1
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


svc_params = {
	'C':[1.0],
	'kernel':['linear'], # very good! the alternatives 'rbf', 'sigmoid' not very good
	'degree':[3],
	'gamma':['auto'],
	'coef0':[0.0],
	'shrinking':[True],
	'probability':[False],
	'tol':[0.001],
	'cache_size':[200],
	'class_weight':[None],
	'verbose':[False],
	'max_iter':[-1],
	'decision_function_shape':[None],
	'random_state':[None]
}

linearsvc_params = {
	'penalty':['l2'],
	'loss':['squared_hinge'],
	'dual':[True],
	'tol':[0.0001],
	'C':[1.0],
	'multi_class':['ovr'],
	'fit_intercept':[True],
	'intercept_scaling':[1],
	'class_weight':[None],
	'verbose':[0],
	'random_state':[None],
	'max_iter':[1000]
}

mnbc_params = {
	'alpha': [0.78, 0.775, 0.785],
	'fit_prior': [False]
}
