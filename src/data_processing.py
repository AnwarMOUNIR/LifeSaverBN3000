import pandas as pd


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize the memory usage of a pandas DataFrame.

    This function:
    - Calculates and prints the initial memory usage in megabytes.
    - Downcasts numeric columns to more memory-efficient dtypes:
      * Float-like columns are converted using ``pd.to_numeric(..., downcast="float")``.
      * Integer-like columns are converted using ``pd.to_numeric(..., downcast="integer")``.
    - Converts object columns to the ``category`` dtype when they have relatively low
      cardinality (unique values / total rows < 0.5), which can significantly reduce memory.
    - Calculates and prints the final memory usage and the percentage reduction.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to optimize. It is not modified in-place; a copy is returned.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with more memory-efficient dtypes where possible.
    """
    df_optimized = df.copy()

    start_mem = df_optimized.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"Initial memory usage: {start_mem:.2f} MB")

    for col in df_optimized.columns:
        col_series = df_optimized[col]
        col_dtype = col_series.dtype

        # Handle numeric columns: ints and floats
        if pd.api.types.is_numeric_dtype(col_dtype):
            if pd.api.types.is_integer_dtype(col_dtype):
                df_optimized[col] = pd.to_numeric(col_series, downcast="integer")
            elif pd.api.types.is_float_dtype(col_dtype):
                df_optimized[col] = pd.to_numeric(col_series, downcast="float")

        # Handle object columns with low cardinality -> category
        elif pd.api.types.is_object_dtype(col_dtype):
            if len(df_optimized) > 0:
                unique_ratio = col_series.nunique(dropna=False) / len(df_optimized)
                if unique_ratio < 0.5:
                    df_optimized[col] = col_series.astype("category")

    end_mem = df_optimized.memory_usage(deep=True).sum() / (1024 ** 2)
    reduction = ((start_mem - end_mem) / start_mem * 100) if start_mem > 0 else 0.0

    print(f"Final memory usage: {end_mem:.2f} MB")
    print(f"Memory reduction: {reduction:.2f}%")

    return df_optimized
