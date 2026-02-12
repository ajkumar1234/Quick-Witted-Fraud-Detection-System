import numpy as np

def add_behavioral_features(X):
    """
    Example placeholder for advanced features.
    In real-time systems, this would include:
    - transaction velocity
    - amount deviation
    """
    transaction_sum = np.sum(X, axis=1).reshape(-1, 1)
    X_enhanced = np.hstack((X, transaction_sum))
    return X_enhanced
