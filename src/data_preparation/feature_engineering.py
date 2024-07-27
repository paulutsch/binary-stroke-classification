from itertools import combinations_with_replacement
from typing import List, Tuple

import numpy as np
import pandas as pd
from loguru import logger





def create_polynomials(
    df: pd.DataFrame, columns: List[str], degree: int = 2
) -> pd.DataFrame:
    new_df = df.copy()

    for deg in range(2, degree + 1):
        for combination in combinations_with_replacement(columns, deg):
            new_col_name = "^".join(combination)
            new_df[new_col_name] = df[list(combination)].prod(axis=1)

    return new_df
