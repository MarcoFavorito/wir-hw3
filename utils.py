import configurations as conf

VERBOSITY_KEY = "verbosity"
DEFAULT_VERBOSITY = 1

def debug_print(*args, **kwargs):
	verbosity = kwargs[VERBOSITY_KEY] if VERBOSITY_KEY in kwargs else DEFAULT_VERBOSITY
	if verbosity <= DEFAULT_VERBOSITY:
		print(*args, **kwargs)