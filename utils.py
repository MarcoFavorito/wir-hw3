import configurations as conf
from nltk.stem.snowball import EnglishStemmer
from nltk import word_tokenize

VERBOSITY_KEY = "verbosity"
DEFAULT_VERBOSITY = 1
stemmer = EnglishStemmer()

def debug_print(*args, **kwargs):
	verbosity = kwargs[VERBOSITY_KEY] if VERBOSITY_KEY in kwargs else DEFAULT_VERBOSITY
	if verbosity <= DEFAULT_VERBOSITY:
		print(*args, **kwargs)


def stemming_tokenizer(text):
	stemmed_text = [stemmer.stem(word) for word in word_tokenize(text, language='english')]
	return stemmed_text