import numpy as np

assert df.shape == (5, 5), "Unexpected matrix dimensions"
assert np.allclose(df.values, df.values.T), "Matrix is not symmetric"
assert np.allclose(np.diag(df.values), 0), "Diagonal is not zero"
assert not df.isna().any().any(), "Matrix contains missing values"

print("URIEL+ syntactic matrix validation passed.")