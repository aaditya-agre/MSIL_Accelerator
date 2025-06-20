from sklearn.model_selection import train_test_split

def train_test_split_data(X, y, test_size=0.2, random_state=42, stratify=None):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
