# Core libraries
import pandas as pd
import numpy as np
import random
import re

# Statistical tools
from scipy import stats
from scipy.stats import zscore
from scipy.cluster.hierarchy import linkage, fcluster

# Imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer

# Encoding
from sklearn.preprocessing import (
    LabelEncoder,
    OrdinalEncoder,
    OneHotEncoder,
    Binarizer,
    KBinsDiscretizer,
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    Normalizer,
    PowerTransformer,
    FunctionTransformer
)
import category_encoders as ce

# Discretization (supervised)
from sklearn.tree import DecisionTreeClassifier

# Feature selection & extraction
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE


class EDA:

    def __init__(self):
        pass
    
    # 1. Data Overview
    def describe(self, data: pd.DataFrame):
        return{
            'cols': data.columns.to_list(),
            'shape': data.shape,
            'describe': data.describe(include = 'all').replace({np.nan: None}).to_dict(),
            'duplicates': int(data.duplicated().sum()),
            'null values': data.isna().sum().replace({np.nan: None}).to_dict(),
            'types': data.dtypes.apply(lambda x: str(x)).to_dict()
        }
    
    # 2. Univariate Analysis
    def univariate_analysis(self, data, type1, type2, column):
            if type1 == 'categorical':
                if type2 == 'countplot':
                    value_counts = data[column].value_counts().to_dict()
                    return {'type': 'countplot', 'data': value_counts}

                elif type2 == 'piechart':
                    value_counts = data[column].value_counts(normalize=True) * 100
                    value_counts = value_counts.round(2).to_dict()
                    return {'type': 'piechart', 'data': value_counts}

            else:  # numerical data
                if type2 == 'histogram':
                    counts, bins = np.histogram(data[column].dropna())
                    return {
                        'type': 'histogram',
                        'bins': bins.tolist(),
                        'counts': counts.tolist()
                    }

                elif type2 == 'distplot':
                    sorted_values = sorted(data[column].dropna().tolist())
                    return {'type': 'distplot', 'data': sorted_values}

                elif type2 == 'boxplot':
                    desc = data[column].describe()
                    q1 = desc['25%']
                    q3 = desc['75%']
                    iqr = q3 - q1
                    lower_whisker = max(data[column].min(), q1 - 1.5 * iqr)
                    upper_whisker = min(data[column].max(), q3 + 1.5 * iqr)
                    outliers = data[(data[column] < lower_whisker) | (data[column] > upper_whisker)][column].tolist()

                    return {
                        'type': 'boxplot',
                        'min': desc['min'],
                        'q1': q1,
                        'median': desc['50%'],
                        'q3': q3,
                        'max': desc['max'],
                        'outliers': outliers
                    }

            return {'error': 'Invalid type or plot option'}
    

    # 3. Multivariate Analysis
    def multivariate_analysis(self,df, no_of_col_to_do_analysis, type1, type2, type3, chosen_cols):
            result = {}

            # Pairplot: comparing multiple numerical columns
            if type3 == 'pairplot':
                cols = chosen_cols.get("cols", [])[:no_of_col_to_do_analysis]
                result = {
                    "type": "pairplot",
                    "cols": cols,
                    "rows": df[cols].dropna().to_dict(orient="records")
                }

            # Numerical vs Numerical
            elif type1 == 'numerical' and type2 == 'numerical':
                x = chosen_cols.get("x")
                y = chosen_cols.get("y")

                if type3 == 'scatterplot':
                    result = {
                        "type": "scatterplot",
                        "x": df[x].dropna().tolist(),
                        "y": df[y].dropna().tolist()
                    }

                elif type3 == 'lineplot':
                    result = {
                        "type": "lineplot",
                        "x": df[x].dropna().tolist(),
                        "y": df[y].dropna().tolist()
                    }

            # Numerical vs Categorical
            elif (type1 == 'numerical' and type2 == 'categorical') or (type1 == 'categorical' and type2 == 'numerical'):
                num_col = chosen_cols.get("x")
                cat_col = chosen_cols.get("y")

                if type3 == 'barplot':
                    grouped = df.groupby(cat_col)[num_col].mean()
                    result = {
                        "type": "barplot",
                        "labels": grouped.index.tolist(),
                        "values": grouped.values.tolist()
                    }

            # Categorical vs Categorical
            elif type1 == 'categorical' and type2 == 'categorical':
                x = chosen_cols.get("x")
                y = chosen_cols.get("y")
                crosstab = pd.crosstab(df[x], df[y])

                if type3 == 'heatmap':
                    result = {
                        "type": "heatmap",
                        "xLabels": crosstab.columns.tolist(),
                        "yLabels": crosstab.index.tolist(),
                        "matrix": crosstab.values.tolist()
                    }

                elif type3 == 'clustermap':
                    scaled = StandardScaler(with_mean=False).fit_transform(crosstab.values)
                    linkage_matrix = linkage(scaled, method='ward')
                    clusters = fcluster(linkage_matrix, t=2, criterion='maxclust')
                    result = {
                        "type": "clustermap",
                        "xLabels": crosstab.columns.tolist(),
                        "yLabels": crosstab.index.tolist(),
                        "matrix": crosstab.values.tolist(),
                        "rowClusters": clusters.tolist()
                    }

            else:
                result = {"error": "Invalid combination or unsupported plot type."}

            return result
    
    
    # 4. Missing Values (should display for those columns having missing values)
    def fill_missing_values(self, df, column, ctype, method):
        '''
        Impute missing values for a given column based on type and method.
        
        Numerical methods: drop, mean, median, arbitrary, random, end_of_distribution, knn, iterative  
        Categorical methods: drop, mode, missing, ffill, bfill, random, missing_indicator, knn, iterative
        '''
        
        if method == 'drop':
            df = df.dropna(subset=[column])
            return df  # early return since no need to fill
        
        if ctype == 'numerical':
            if method == 'mean':
                df[column] = df[column].fillna(df[column].mean())
            elif method == 'median':
                df[column] = df[column].fillna(df[column].median())
            elif method == 'arbitrary':
                df[column] = df[column].fillna(-1)
            elif method == 'random':
                non_null_vals = df[column].dropna().values
                df[column] = df[column].apply(lambda x: random.choice(non_null_vals) if pd.isnull(x) else x)
            elif method == 'end_of_distribution':
                extreme_val = df[column].mean() + 3 * df[column].std()
                df[column] = df[column].fillna(extreme_val)
            elif method == 'knn':
                imputer = KNNImputer(n_neighbors=5)
                df[[column]] = imputer.fit_transform(df[[column]])
            elif method == 'iterative':
                imputer = IterativeImputer()
                df[[column]] = imputer.fit_transform(df[[column]])

        elif ctype == 'categorical':
            if method == 'mode':
                df[column] = df[column].fillna(df[column].mode()[0])
            elif method == 'missing':
                df[column] = df[column].fillna("Missing")
            elif method == 'ffill':
                df[column] = df[column].fillna(method='ffill')
            elif method == 'bfill':
                df[column] = df[column].fillna(method='bfill')
            elif method == 'random':
                non_null_vals = df[column].dropna().values
                df[column] = df[column].apply(lambda x: random.choice(non_null_vals) if pd.isnull(x) else x)
            elif method == 'missing_indicator':
                df[column + "_missing"] = df[column].isnull()
            elif method == 'knn':
                df_temp = df.copy()
                df_temp[column] = df_temp[column].astype('category').cat.codes.replace(-1, np.nan)
                imputer = KNNImputer(n_neighbors=5)
                df_temp[[column]] = imputer.fit_transform(df_temp[[column]])
                df[column] = df_temp[column].astype(int).astype(str)
            elif method == 'iterative':
                df_temp = df.copy()
                df_temp[column] = df_temp[column].astype('category').cat.codes.replace(-1, np.nan)
                imputer = IterativeImputer()
                df_temp[[column]] = imputer.fit_transform(df_temp[[column]])
                df[column] = df_temp[column].astype(int).astype(str)

        else:
            raise ValueError("Invalid ctype. Must be 'numerical' or 'categorical'.")

        return df

    # 5. Outliers Handling

    def remove_outliers(self,df, column, method='zscore', strategy='trimming', threshold=3):
        """
        Removes or caps outliers from a numerical column.

        Parameters:
            df (pd.DataFrame): Input DataFrame.
            column (str): Column name to process.
            method (str): 'zscore', 'iqr', or 'percentile'.
            strategy (str): 'trimming' or 'capping'.
            threshold (float or tuple): 
                - For zscore: float (e.g., 3)
                - For iqr: multiplier (e.g., 1.5)
                - For percentile: tuple (e.g., (1, 99))

        Returns:
            pd.DataFrame: Modified DataFrame
        """
        if method == 'NA':
            return df
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            mask = z_scores > threshold
            if strategy == 'trimming':
                df = df.loc[~mask]
            elif strategy == 'capping':
                cap_value = df.loc[~mask, column].max()
                df.loc[mask, column] = cap_value

        elif method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            if strategy == 'trimming':
                df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            elif strategy == 'capping':
                df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
                df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])

        elif method == 'percentile':
            if not isinstance(threshold, tuple) or len(threshold) != 2:
                raise ValueError("Percentile threshold should be a tuple like (1, 99)")
            lower_percentile = np.percentile(df[column], threshold[0])
            upper_percentile = np.percentile(df[column], threshold[1])

            if strategy == 'trimming':
                df = df[(df[column] >= lower_percentile) & (df[column] <= upper_percentile)]
            elif strategy == 'capping':
                df[column] = np.where(df[column] < lower_percentile, lower_percentile, df[column])
                df[column] = np.where(df[column] > upper_percentile, upper_percentile, df[column])

        else:
            raise ValueError("Invalid method. Choose from 'zscore', 'iqr', or 'percentile'.")

        return df
 
    
    # 6. Feature Encoding

    def feature_encoding(self,df, column, ftype, method, sub_method=None, strategy=None, target_col=None, bins=5):
        """
        Generalized feature encoding for numerical and categorical features.

        Parameters:
            df (pd.DataFrame): The DataFrame to modify
            column (str): Column to encode
            ftype (str): 'numerical' or 'categorical'
            method (str): 
                - For numerical: 'discretization', 'binarization'
                - For categorical: 'ordinal_input', 'ordinal_output', 'nominal'
            sub_method (str): 'unsupervised' or 'supervised' (for discretization)
            strategy (str): For unsupervised: 'uniform', 'quantile', 'kmeans'
            target_col (str): Target column (for supervised binning or label encoding)
            bins (int): Number of bins for discretization

        Returns:
            pd.DataFrame
        """

        if ftype == 'numerical':
            if method == 'discretization':
                if sub_method == 'unsupervised':
                    if strategy not in ['uniform', 'quantile', 'kmeans']:
                        raise ValueError("strategy must be 'uniform', 'quantile', or 'kmeans'")
                    est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=strategy)
                    df[column + f"_{strategy}_bin"] = est.fit_transform(df[[column]])
                elif sub_method == 'supervised':
                    if target_col is None:
                        raise ValueError("target_col required for supervised binning")
                    tree = DecisionTreeClassifier(max_leaf_nodes=bins)
                    X = df[[column]].dropna()
                    y = df.loc[X.index, target_col]
                    tree.fit(X, y)
                    df[column + "_tree_bin"] = tree.apply(df[[column]])
                else:
                    raise ValueError("sub_method must be 'unsupervised' or 'supervised'")

            elif method == 'binarization':
                binarizer = Binarizer()
                df[column + "_bin"] = binarizer.fit_transform(df[[column]])

            else:
                raise ValueError("method must be 'discretization' or 'binarization' for numerical features")

        elif ftype == 'categorical':
            if method == 'ordinal_input':
                encoder = OrdinalEncoder()
                df[column + "_ord"] = encoder.fit_transform(df[[column]])
            elif method == 'ordinal_output':
                if target_col:
                    encoder = LabelEncoder()
                    df[column + "_lab"] = encoder.fit_transform(df[column])
                else:
                    raise ValueError("target_col is required for label encoding")
            elif method == 'nominal':
                encoder = OneHotEncoder(sparse=False, drop='first')
                encoded = encoder.fit_transform(df[[column]])
                ohe_df = pd.DataFrame(encoded, columns=[f"{column}_{cat}" for cat in encoder.categories_[0][1:]], index=df.index)
                df = df.drop(columns=[column])
                df = pd.concat([df, ohe_df], axis=1)
            else:
                raise ValueError("Invalid method for categorical feature")

        else:
            raise ValueError("ftype must be 'numerical' or 'categorical'")

        return df

    # 7. Feature Scaling
    def feature_scaling(self,df, column, method='standardization', strategy='zscore'):
        """
        Apply feature scaling to a numerical column.

        Parameters:
            df (pd.DataFrame): Input DataFrame.
            column (str): Column to scale.
            method (str): 'standardization' or 'normalization'
            strategy (str): 
                - If method = 'standardization' → 'zscore'
                - If method = 'normalization' → 'minmax', 'mean', 'max_abs', 'robust'

        Returns:
            pd.DataFrame: Modified DataFrame with scaled column.
        """
        if method == 'standardization':
            if strategy != 'zscore':
                raise ValueError("Standardization only supports 'zscore'")
            scaler = StandardScaler()
            df[column + "_zscore"] = scaler.fit_transform(df[[column]])

        elif method == 'normalization':
            if strategy == 'minmax':
                scaler = MinMaxScaler()
                df[column + "_minmax"] = scaler.fit_transform(df[[column]])
            elif strategy == 'mean':
                # mean norm applies across rows → not ideal for a single column
                norm = Normalizer(norm='l2')
                df[column + "_meanNorm"] = norm.fit_transform(df[[column]])
            elif strategy == 'max_abs':
                scaler = MaxAbsScaler()
                df[column + "_maxabs"] = scaler.fit_transform(df[[column]])
            elif strategy == 'robust':
                scaler = RobustScaler()
                df[column + "_robust"] = scaler.fit_transform(df[[column]])
            else:
                raise ValueError("Invalid normalization strategy. Choose from 'minmax', 'mean', 'max_abs', 'robust'.")

        else:
            raise ValueError("method must be 'standardization' or 'normalization'")

        return df
    
    # 8. Mixed Data Column
    def handle_mixed_data(self,df, column, mixed_type='type1'):
        """
        Handles columns with mixed categorical and numerical data.

        Parameters:
            df (pd.DataFrame): The input DataFrame
            column (str): Column to process
            mixed_type (str): 'type1' or 'type2'
                - 'type1': Single cell with mix, e.g., 'C45'
                - 'type2': Interleaved values in a column, e.g., A, 1, B, 2

        Returns:
            pd.DataFrame: DataFrame with separate categorical and numerical columns
        """
        if mixed_type == 'type1':
            # case like 'C45' or 'D32'
            def split_mixed(val):
                if pd.isnull(val):
                    return (np.nan, np.nan)
                match = re.match(r"([A-Za-z]+)?([0-9]+)?", str(val))
                if match:
                    cat = match.group(1) if match.group(1) else np.nan
                    num = int(match.group(2)) if match.group(2) else np.nan
                    return (cat, num)
                return (np.nan, np.nan)

            df[[f"{column}_cat", f"{column}_num"]] = df[column].apply(split_mixed).apply(pd.Series)

        elif mixed_type == 'type2':
            # case like ['A', 1, 6, 'B', 4, 2, 'G']
            new_rows = []
            for val in df[column]:
                if isinstance(val, str) and val.isalpha():
                    new_rows.append((val, np.nan))
                elif isinstance(val, (int, float)):
                    new_rows.append((np.nan, val))
                else:
                    new_rows.append((np.nan, np.nan))
            new_df = pd.DataFrame(new_rows, columns=[f"{column}_cat", f"{column}_num"])
            df = df.drop(columns=[column]).reset_index(drop=True)
            df = pd.concat([df, new_df], axis=1)

        else:
            raise ValueError("mixed_type must be 'type1' or 'type2'")

        return df
    
    def split_based_on_delimiter(self, df, column, delimiter='-'):
        """
        Splits a column based on a specified delimiter and creates new columns.

        Parameters:
            df (pd.DataFrame): Input DataFrame
            column (str): Column to split
            delimiter (str): Delimiter to use for splitting

        Returns:
            pd.DataFrame: Updated DataFrame with new columns created from the split
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

        # Split the column and expand into new columns
        split_cols = df[column].str.split(delimiter, expand=True)
        split_cols.columns = [f"{column}_part{i+1}" for i in range(split_cols.shape[1])]

        # Drop the original column and concatenate the new columns
        df = df.drop(columns=[column])
        df = pd.concat([df, split_cols], axis=1)

        return df
    
    # 9. Feature Tranformation

    def feature_transformation(self, df, column, type1='function', type2='log'):
        """
        Applies in-place feature transformation on a given column.

        Parameters:
            df (pd.DataFrame): Input DataFrame
            column (str): Column to transform
            type1 (str): Main type - 'function' or 'power'
            type2 (str): Subtype method (see below)
                - if 'function': 'log', 'reciprocal', 'square', 'sqrt'
                - if 'power': 'boxcox', 'yeojohnson'

        Returns:
            pd.DataFrame: Updated DataFrame with transformed column (in-place)
        """
        col_data = df[column].copy()

        try:
            if type1 == 'function':
                if type2 == 'log':
                    df[column] = np.log1p(col_data)
                elif type2 == 'reciprocal':
                    df[column] = 1 / (col_data.replace(0, np.nan))
                elif type2 == 'square':
                    df[column] = col_data ** 2
                elif type2 == 'sqrt':
                    df[column] = np.sqrt(col_data)
                else:
                    raise ValueError("Invalid type2 for function. Choose from: log, reciprocal, square, sqrt")

            elif type1 == 'power':
                if type2 == 'boxcox':
                    positive_data = col_data[col_data > 0].values.reshape(-1, 1)
                    transformer = PowerTransformer(method='box-cox')
                    transformed = transformer.fit_transform(positive_data)
                    df.loc[col_data > 0, column] = transformed.flatten()
                elif type2 == 'yeojohnson':
                    transformer = PowerTransformer(method='yeo-johnson')
                    df[column] = transformer.fit_transform(col_data.values.reshape(-1, 1)).flatten()
                else:
                    raise ValueError("Invalid type2 for power. Choose from: boxcox, yeojohnson")

            else:
                raise ValueError("Invalid type1. Choose 'function' or 'power'.")

            return df

        except Exception as e:
            print(f"Transformation failed: {e}")
            return df

    # 10. Manual Feature Selection

    def manual_feature_selection(self,df, target, selected_features):
        """
        Manually selects features from the dataset.

        Parameters:
            df (pd.DataFrame): Input DataFrame
            target (str): Name of the target column
            selected_features (list): List of feature column names to keep

        Returns:
            pd.DataFrame: DataFrame with selected features and target
        """
        if target not in df.columns:
            raise ValueError("Target column not found in DataFrame.")

        missing = [f for f in selected_features if f not in df.columns]
        if missing:
            raise ValueError(f"Selected features not in DataFrame: {missing}")

        return df[selected_features + [target]]
    
    # 11. Feature Selection & Extraction

    def feature_selection_extraction(self,df, target, method_type='selection', method=None, n_features=5):
        """
        Applies feature selection or feature extraction.

        Parameters:
            df (pd.DataFrame): Input DataFrame with features and target
            target (str): Target column name
            method_type (str): 'selection' or 'extraction'
            method (str): Specific method under each type
                - selection: 'forward', 'backward'
                - extraction: 'pca', 'lda', 'tsne'
            n_features (int): Number of features/components to keep

        Returns:
            Transformed feature DataFrame (X_new), target series (y)
        """

        X = df.drop(columns=[target])
        y = df[target]

        if method_type == 'selection':
            model = LogisticRegression(max_iter=1000)
            if method == 'forward':
                selector = SequentialFeatureSelector(model, n_features_to_select=n_features, direction='forward')
                selector.fit(X, y)
                X_new = selector.transform(X)
                selected_cols = X.columns[selector.get_support()]
                X_new = pd.DataFrame(X_new, columns=selected_cols)
            elif method == 'backward':
                selector = SequentialFeatureSelector(model, n_features_to_select=n_features, direction='backward')
                selector.fit(X, y)
                X_new = selector.transform(X)
                selected_cols = X.columns[selector.get_support()]
                X_new = pd.DataFrame(X_new, columns=selected_cols)
            else:
                raise ValueError("Invalid method for selection. Choose 'forward' or 'backward'.")

        elif method_type == 'extraction':
            if method == 'pca':
                pca = PCA(n_components=n_features)
                X_new = pca.fit_transform(X)
                X_new = pd.DataFrame(X_new, columns=[f'pca_{i+1}' for i in range(n_features)])
            elif method == 'lda':
                lda = LDA(n_components=min(n_features, len(np.unique(y)) - 1))  # LDA: max components = classes - 1
                X_new = lda.fit_transform(X, y)
                X_new = pd.DataFrame(X_new, columns=[f'lda_{i+1}' for i in range(X_new.shape[1])])
            elif method == 'tsne':
                tsne = TSNE(n_components=n_features)
                X_new = tsne.fit_transform(X)
                X_new = pd.DataFrame(X_new, columns=[f'tsne_{i+1}' for i in range(n_features)])
            else:
                raise ValueError("Invalid method for extraction. Choose 'pca', 'lda', or 'tsne'.")

        else:
            raise ValueError("Invalid method_type. Choose 'selection' or 'extraction'.")

        return X_new, y
