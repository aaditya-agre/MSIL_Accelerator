from flask import Flask, redirect, url_for, request, Response, send_file
from eda import EDA
import pandas as pd
import numpy as np
from Regressor.utils.metrics import regression_metrics
from Regressor.utils.data_split import train_test_split_data
import Regressor.models.trainer as RegressionModel
import Classification.models.trainer as ClassificationModel
from Regressor.fe.feature_engineering import feature_engineering
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import Regressor.tuning.hyperparam as RegressionTuning
import Classification.tuning.hyperparam as ClassificationTuning
from Classification.utils.metrics import classification_metrics
import io
from requests_toolbelt import MultipartEncoder
import json
from Clustering.clustering.models.trainer import get_model as get_cluster_model
from Clustering.clustering.utils.metrics import clustering_metrics
import Clustering.clustering.tuning.hyperparam as ClusteringHyperparameters


df = {'edasl': None, 'edaul': None, 'classification': None, 'regression': None, 'timeseries': None, 'clustering': None}
models = {'regression': None, 'classification': None}
count = {'regression': 0, 'classification': 0, 'clustering': 0}

eda = EDA()
app = Flask(__name__)
         
@app.get('/') 
def index():
    return redirect(url_for('static', filename = 'index.html'))

@app.post('/upload/<model>')
def uploadFile(model):
    global df, eda
    try:
        df[model] = pd.read_csv(request.files['dataset'])
    except KeyError:
        print("No such model!")
        return Response(status = 404)
    if model == 'edasl':
        return eda.describe(df['edasl'])
    return {'cols': df[model].columns.to_list()}

@app.get('/eda/<model>')
def useEDA(model):
    global df
    df[model] = df['edasl']
    return {'cols': df[model].columns.to_list()}

@app.get('/eda/newcols')
def getNewCols():
    global df
    return {'cols': df['edasl'].columns.to_list()}

@app.post('/train/<model>')
def trainModel(model):
    global count, df, models

    if model == 'regression':
        data = request.get_json(force = True)
        X = df['regression'].drop(data['target'], axis=1)
        y = df['regression'][data['target']]

        X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2, random_state=42)

        model = RegressionModel.get_model('ridge', task='regression')

        param_grid = RegressionTuning.hyperparams(str(model))

        def tune_model(model, param_grid, X_train, y_train, scoring='r2', operation = 'GridSeachCV',cv=5):
            if operation == 'GridSearchCV':
                grid = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=-1, verbose=1)
                grid.fit(X_train, y_train)
                return grid.best_estimator_, grid.best_params_, grid
            if operation == 'RandomizedSearchCV':
                random_search = RandomizedSearchCV(
                    model,
                    param_grid,
                    n_iter=20,               
                    scoring=scoring,            
                    cv=cv,                     
                    verbose=1,
                    random_state=42,
                    n_jobs=-1
                    )
                random_search.fit(X_train, y_train)
                return random_search.best_estimator_, random_search.best_params_, random_search
            if operation == 'None':
                return model, model.get_params(), None

        tuned_model, best_params, tuned_obj = tune_model(model, param_grid, X_train, y_train, operation = data['tuning'])
        
        buffer = io.BytesIO()
        RegressionTuning.visualize_results(tuned_obj, param_name = 'alpha', buffer = buffer)
        buffer.seek(0)
        # RegressionTuning.logger(str(tuned_model), best_params)
        model = tuned_model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = regression_metrics(y_test, y_pred)

        models['regression'] = tuned_model

        count['regression'] += 1
        metrics['num'] = count['regression']
        
        m = MultipartEncoder(fields = {'metrics': json.dumps(metrics), 'graph': ('regGraph.png', buffer, 'image/png')})

        return Response(m.to_string(), mimetype = m.content_type)
    
    elif model == 'classification':
        data = request.get_json(force = True)
        X = df['classification'].drop(data['target'], axis=1)
        y = df['classification'][data['target']]

        X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2, random_state=42, stratify=y)

        model = ClassificationModel.get_model(data['model'], task='classification')

        param_grid = ClassificationTuning.hyperparams(str(model))

        def tune_model(model, param_grid, X_train, y_train, scoring='accuracy', operation='GridSearchCV', cv=5):
            if operation == 'GridSearchCV':
                grid = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=-1, verbose=1)
                grid.fit(X_train, y_train)
                return grid.best_estimator_, grid.best_params_, grid
            
            elif operation == 'RandomizedSearchCV':
                random_search = RandomizedSearchCV(
                    model,
                    param_grid,
                    n_iter=20,
                    scoring=scoring,
                    cv=cv,
                    verbose=1,
                    random_state=42,
                    n_jobs=-1
                )
                random_search.fit(X_train, y_train)
                return random_search.best_estimator_, random_search.best_params_, random_search
            
            elif operation == 'None':
                return model, model.get_params(), None
            
        tuned_model, best_params, tuned_obj = tune_model(model, param_grid, X_train, y_train, operation = data['tuning'])
        
        buffer = io.BytesIO()
        ClassificationTuning.visualize_results(tuned_obj, param_name='n_estimators', buffer = buffer)  # Change this param based on model
        buffer.seek(0)

        # ClassificationTuning.logger(str(tuned_model), best_params)

        model = tuned_model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = classification_metrics(y_test, y_pred, average='macro')

        models['classification'] = tuned_model

        count['classification'] += 1
        metrics['num'] = count['classification']

        m = MultipartEncoder(fields = {'metrics': json.dumps(metrics), 'graph': ('classGraph.png', buffer, 'image/png')})

        return Response(m.to_string(), mimetype = m.content_type)
    
    elif model == 'clustering':
        
        data = request.get_json(force = True)
        X_cluster = df['clustering']

        model_name = data['model']
        model = get_cluster_model(model_name)
        params = ClusteringHyperparameters.hyperparams(model_name)

        tuned_model, best_params, tuned_obj, metric_logs = ClusteringHyperparameters.tune_model(model, params, X_cluster, operation = data['tuning'])

        buffer = io.BytesIO()
        ClusteringHyperparameters.visualize_cluster_evaluation(model_name, X_cluster, buffer, k_range=(2, 11))
        buffer.seek(0)

        tuned_model.fit(X_cluster)
        cluster_labels = tuned_model.labels_

        metrics = clustering_metrics(X_cluster, cluster_labels)

        X_cluster['assigned_cluster'] = cluster_labels
        csv_buffer = io.BytesIO()
        X_cluster.to_csv(csv_buffer, index = False)
        csv_buffer.seek(0)

        count['clustering'] += 1
        metrics['num'] = count['clustering']

        # cluster_logger(model_name=str(model.__class__.__name__), best_params=params, best_score=score)
        m = MultipartEncoder(fields = {'metrics': json.dumps(metrics), 'output': ('clusters.csv', csv_buffer, 'text/csv'), 'graph': ('clusterGraph.png', buffer, 'image/png')})

        return Response(m.to_string(), mimetype = m.content_type)

@app.post('/run/<model>')
def runModel(model):
    if model == 'regression':
        input = pd.read_csv(request.files['input'])
        input['Predicted Values'] = models['regression'].predict(input)
        csv_buffer = io.BytesIO()
        input.to_csv(csv_buffer, index = False)
        csv_buffer.seek(0)
        return send_file(csv_buffer, download_name = 'predicted.csv', as_attachment = True, mimetype = 'text/csv')
    
    elif model == 'classification':
        input = pd.read_csv(request.files['input'])
        input['Predicted Values'] = models['classification'].predict(input)
        csv_buffer = io.BytesIO()
        input.to_csv(csv_buffer, index = False)
        csv_buffer.seek(0)
        return send_file(csv_buffer, download_name = 'predicted.csv', as_attachment = True, mimetype = 'text/csv')
    
@app.post('/univariate')
def univariate():
    global df, eda
    req = request.get_json()
    column = req.get("column")
    type1 = req.get("type1")
    type2 = req.get("type2")

    if df['edasl'] is None or column not in df['edasl'].columns:
        return {"error": "Invalid column or no data uploaded"}, 400
    #print(eda.univariate_analysis(df, type1, type2, column))
    return eda.univariate_analysis(df['edasl'], type1, type2, column)


@app.post('/multivariate')
def multivariate():
    global df, eda
    req = request.get_json()
    no_of_col_to_do_analysis = req.get('no_of_col_to_do_analysis')
    type1 = req.get('type1')
    type2 = req.get('type2')
    type3 = req.get('type3')
    chosen_cols = req.get('chosen_cols')

    if df['edasl'] is None:
        return {'error': 'No dataset uploaded'}, 400

    try:
        return eda.multivariate_analysis(df['edasl'], no_of_col_to_do_analysis, type1, type2, type3, chosen_cols)
    except Exception as e:
        print("Error in /multivariate:", e)
        return {"error": str(e)}, 500
    
@app.route('/handle-missing', methods=['POST'])
def handle_missing():
    global df, eda

    data = request.json
    imputations = data.get("impute", [])
    if not imputations:
        return {"error": "No imputations provided"}, 400

    try:
        for item in imputations:
            column = item.get("column")
            ctype = item.get("type")
            method = item.get("method")
            
            if column and ctype and method:
                df['edasl'] = eda.fill_missing_values(df['edasl'], column, ctype, method)
        print(df['edasl'].info())
        print(df['edasl'].isnull().sum())

        return {"message": "Missing values handled successfully.",
                'columns': df['edasl'].columns.tolist()}

    except Exception as e:
        return {"error": str(e)}, 500
    

@app.route('/remove-outliers', methods=['POST'])
def remove_outliers_route():
    global df, eda
    try:
        data = request.get_json()
        print(data)
        if not data or 'outliers' not in data:
            return {'error': 'Invalid request. "outliers" key missing.'}, 400

        for item in data['outliers']:
            column = item.get('column')
            method = item.get('method')
            strategy = item.get('strategy')
            threshold = item.get('threshold')

            if column not in df['edasl'].columns:
                continue  # Skip unknown columns

            # Convert list to tuple if it's percentile threshold
            if method == 'percentile' and isinstance(threshold, list):
                threshold = tuple(threshold)

            df['edasl'] = eda.remove_outliers(df['edasl'], column, method, strategy, threshold)

        return {
            'message': 'Outlier handling completed.',
            'rows_remaining': len(df['edasl']),
            'columns': list(df['edasl'].columns)
        }, 200

    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/feature-encoding', methods=['POST'])
def feature_encoding_route():
    global df, eda

    try:
        configs = request.get_json()
        for config in configs:
            column = config['column']
            ftype = config['ftype']
            method = config['method']
            sub_method = config.get('sub_method')
            strategy = config.get('strategy')
            target_col = config.get('is_target')
            bins = config.get('bins', 5)

            
            

            
            df['edasl'] = eda.feature_encoding(
                df=df['edasl'],
                column=column,
                ftype=ftype,
                method=method,
                sub_method=sub_method,
                strategy=strategy,
                target_col=target_col,
                bins=bins
            )

        return {'message': 'Feature encoding applied successfully'}, 200

    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/feature-scaling', methods=['POST'])
def feature_scaling():
    global df, eda
    try:
        payload = request.get_json()
        print(payload)
        if not payload or not isinstance(payload, list):
            return {"error": "Invalid payload format"}, 400

        for config in payload:
            column = config.get('column')
            method = config.get('method')
            strategy = config.get('strategy')

            if not column or not method or not strategy:
                return {"error": f"Missing required field in config: {config}"}, 400

            # Apply scaling
            df['edasl'] = eda.feature_scaling(df['edasl'], column, method, strategy)

        return {"message": "Feature scaling applied successfully."}, 200

    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/mixed-data', methods=['POST'])
def handle_mixed_data_route():
    global df, eda
    try:
        data = request.get_json()
        column = data.get('column')
        mixed_type = data.get('mixed_type')

        if not column or not mixed_type:
            return {'error': 'Both "column" and "mixed_type" are required.'}, 400

        df['edasl'] = eda.handle_mixed_data(df['edasl'], column, mixed_type)
        return {'message': 'Mixed data handled successfully.', 'columns': df['edasl'].columns.tolist()}

    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/split_column', methods=['POST'])
def split_column():
    global df, eda
    try:
        data = request.get_json()
        column = data['column']
        delimiter = data.get('delimiter', '-')

        # Apply the EDA function
        df['edasl'] = eda.split_based_on_delimiter(df['edasl'], column, delimiter)

        return {
            "message": "Column split successfully.",
            "cols": df['edasl'].columns.tolist(),
            "shape": df['edasl'].shape
        },200
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/feature-transform', methods=['POST'])
def feature_transform():
    global df, eda

    if df['edasl'].empty:
        return {"error": "No data uploaded yet"}, 400

    try:
        config = request.get_json()
        required_keys = ['column', 'type1', 'type2']

        for key in required_keys:
            if key not in config:
                return {"error": f"Missing key: {key}"}, 400

        column = config['column']
        type1 = config['type1']
        type2 = config['type2']

        df['edasl'] = eda.feature_transformation(df['edasl'], column, type1, type2)
        return {"status": "success", "new_columns": df['edasl'].columns.tolist()},200
    except Exception as e:
        return {"error": str(e)}, 500
    
@app.route('/manual-feature-selection', methods=['POST'])
def manual_feature_selection():
    global df, eda

    if df['edasl'].empty:
        return {"error": "No data uploaded yet"}, 400

    try:
        data = request.get_json()
        target = data['target']
        selected_features = data['selected_features']

        df['edasl'] = eda.manual_feature_selection(df['edasl'], target, selected_features)
        return {"status": "success", "columns": df['edasl'].columns.tolist()}, 200   
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/feature-select-extract', methods=['POST'])
def feature_select_extract():
    global df, eda
    try:
        data = request.get_json()

        target = data.get('target')
        method_type = data.get('method_type')  # 'selection' or 'extraction'
        method = data.get('method')           # 'forward', 'backward', 'pca', 'lda', 'tsne'
        n_features = data.get('n_features')

        if not target or not method_type or not method or n_features is None:
            return {"error": "Missing one or more required fields"}, 400

        # Perform the operation using your EDA class
        df['edasl'] = eda.feature_selection_extraction(df['edasl'], target=target, method_type=method_type, method=method, n_features=n_features)

        return {"message": "Feature operation completed successfully!"}

    except Exception as e:
        return {"error": str(e)}, 500
    
@app.route('/download', methods=['GET'])
def download_file():
    global df, eda
    if df['edasl'] is None or df['edasl'].empty:
        return {'error': 'No data available for download.'}, 400

    # Convert DataFrame to CSV in memory
    buffer = io.StringIO()
    df['edasl'].to_csv(buffer, index=False)
    buffer.seek(0)

    return send_file(
        io.BytesIO(buffer.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='preprocessed_dataset.csv'
    )