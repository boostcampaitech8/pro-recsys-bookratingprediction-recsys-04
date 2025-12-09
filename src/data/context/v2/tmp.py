import pandas as pd
import numpy as np
base_path = "/data/ephemeral/home/data/features/v2/"


books = pd.read_parquet(base_path + "books/book_features.parquet")



print(books.columns)