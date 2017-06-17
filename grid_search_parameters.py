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

# Every set of parameters, in the following lines of code,
# are set to the obtained optimal value.
# If you want to check the full grid search,
# You should toggle by comment one of the two version of parameters.


# parameter of the vectorizer for each model
vectorizer_params = {
	'kNN': {
		# optimal values
		'analyzer': ['word'],
		'tokenizer': [stemming_tokenizer],
		'ngram_range': [(1, 3)],
		'stop_words':[stopwords.words('english')],
		'max_df': [0.2],
		'min_df': [1],
		'binary': [True],
		'use_idf': [True],
		'smooth_idf': [False],
		'sublinear_tf': [True],

		# full grid search
		# 'analyzer' : ['word', 'char'],
		# 'tokenizer': [None, stemming_tokenizer, porter_stemming_tokenizer, lancaster_stemming_tokenizer],
		# 'ngram_range': [(x, y) for x in range(1, 6) for y in range(1, 6) if x <= y],
		# 'stop_words':[None, stopwords.words('english')],
		# 'max_df':[0.1*k for k in range(1, 11)],
		# 'min_df': range(1, 11),
		# 'binary':[True, False],
		# 'use_idf':[True, False],
		# 'smooth_idf': [True, False],
		# 'sublinear_tf':[True, False]

	},
	'MultinomialNB':{
		# optimal values
		'analyzer': ['word'],
		'tokenizer': [stemming_tokenizer],
		'ngram_range': [(1,1), (1, 2), (1, 3)],
		'stop_words':[stopwords.words('english')],
		'max_df': [1.0],
		'min_df': [1],
		'binary': [False],
		'use_idf': [True],
		'smooth_idf': [True],
		'sublinear_tf': [False],

		# full grid search
		# 'analyzer' : ['word', 'char'],
		# 'tokenizer': [None, stemming_tokenizer, porter_stemming_tokenizer, lancaster_stemming_tokenizer],
		# 'ngram_range': [(x, y) for x in range(1, 2) for y in range(1, 6) if x <= y],
		# 'stop_words':[None, stopwords.words('english')],
		# 'max_df':[0.1*k for k in range(1, 11)] + [0.15, 0.175],
		# 'min_df': range(1, 11),
		# 'binary':[True, False],
		# 'use_idf':[True, False],
		# 'smooth_idf': [True, False],
		# 'sublinear_tf':[True, False]
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


kNN_params = {
	# optimal values
	'n_neighbors':  [7],
	'weights': ['uniform'],
	'p': [2],

	# full grid search
	# 'n_neighbors':  range(3, 11),
	# 'weights': ['uniform', 'distance'],
	# 'p':[1, 2],
}

mnbc_params = {
	# optimal values
	'alpha': [0.78],
	'fit_prior': [False]

	# full grid search
	# 'alpha': [0.1*k for k in range(7, 11)] + [0.75, 0.78, 0.79],
	# 'fit_prior': [True, False]
}

linearsvc_params = {
	'loss':['squared_hinge'],
	'C': [0.7],
	# it is set to a low number
	# in order to have better convergence
	'tol':[0.0000001],


}

# 'fit_intercept':[False],
# 'intercept_scaling':[1],
# 'class_weight':[None],
# 'verbose':[1],
# 'random_state':[None],
# 'max_iter':[1000],
# 'multi_class':['ovr'],
# 'dual':[True],

# 'penalty':['l2', 'l1'],