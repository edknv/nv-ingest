import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    import cudf
