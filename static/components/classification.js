const {useState} = React;

function Classification() {
    const [cols, setCols] = useState(null);
    const [metrics, setMetrics] = useState([]);
    const [graph, setGraph] = useState(null);

    let colOptions = null
    if (cols) {
        colOptions = cols.map(col => { 
            return <option key = {col}>{col}</option>; 
        });
    }

    async function useEDA() {
        await fetch('/eda/classification').then(response => response.json()).then(result => setCols(result['cols']));
    }

    async function uploadFile() {
        const data = new FormData();
        const f = document.getElementById('classificationTrainFile');
        data.append('dataset', f.files[0]);
        await fetch('/upload/classification', {
            method: 'POST',
            body: data
        }).then(response => response.json()).then(result => setCols(result['cols']));
    }

    function changeSplit() {
        document.getElementById('classSplitDisplay').innerText = document.getElementById('classSplit').value;
    }

    async function trainModel() {
        let data = {'model' : document.getElementById('classModel').value,
                'split': document.getElementById('classSplit').value,
                'target': document.getElementById('classTargetVar').value, 
                'tuning': document.getElementById('classTuning').value
        }
        
        await fetch('/train/classification', {
            method: 'POST',
            body: JSON.stringify(data)
        }).then(response => response.formData()).then(formData => {
            let result = JSON.parse(formData.get('metrics'));
            setMetrics([...metrics, [result['Accuracy'], result['Precision'], result['Recall'], result['F1 Score'], data['model'], data['split'], result['num']]]);
            setGraph(URL.createObjectURL(formData.get('graph')));
        });
    }

    async function runModel() {
        let data = new FormData();
        data.append('input', document.getElementById('classificationOutputFile').files[0]);
        await fetch('/run/classification', {
            method: 'POST',
            body: data
        }).then(response => response.blob()).then(blob => saveAs(blob, 'prediction.csv'));
    }

    return(
        <form>
            <input type = "file" id = "classificationTrainFile"/>
            <input type = "button" value = "Upload" onClick = {uploadFile}/><br /><br />
            <label>Use data from EDA: </label><input type = "checkbox" onChange = {useEDA}/><br /><br />
            <label>Model: </label>
            <select id = "classModel">
                <option value = "logistic">Logistic Regression</option>
                <option value = "decision_tree">Decision Tree Classifier</option>
                <option value = "random_forest">Random Forest Classifier</option>
                <option value = "gradient_boosting">Gradient Boosting Classifier</option>
                <option value = "xgb">XGBoost Classifier</option>
                <option value = "gaussian_nb">Gaussian Naive Bayes</option>
                <option value = "multinomial_nb">Multinomial Naive Bayes</option>
                <option value = "bernoulii_nb">Bernoulii Naive Bayes</option>
                <option value = "linear_svm">Linear Support Vector Classifier</option>
                <option value = "kernel_svm">Kernel Support Vector Classifier</option>
                <option value = "voting">Voting Classifier</option>
                <option value = "bagging">Bagging Classifier</option>
                <option value = "adaboost">AdaBoost Classifier</option>
                <option value = "ann">MLP Classifier</option>
            </select><br /><br />
            <label>Hyperparameter Tuning: </label>
            <select id = "classTuning"> 
                <option>GridSearchCV</option>
                <option>RandomizedSearchCV</option>
            </select><br /><br />
            <label>Train - test split (20 - 60): </label><input type = "range" min = "20" max = "60" id = "classSplit" onChange = {changeSplit}/><span id = "classSplitDisplay"></span><br /><br />
            {cols && <label>Target Variable: </label>}
            {cols && <><select id = "classTargetVar">{colOptions}</select><br /><br /></>}
            <input type = "button" value = "Train" onClick = {trainModel}/><br /><br />
            {metrics.length != 0 && <><table>
                <thead>
                    <tr>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
                        <th>Model</th>
                        <th>Split</th>
                    </tr>
                </thead>
                <tbody>
                    {metrics.map(metric => {
                        return <tr key = {metric[5]}>
                            <td>{metric[0]}</td>
                            <td>{metric[1]}</td>
                            <td>{metric[2]}</td>
                            <td>{metric[3]}</td>
                            <td>{metric[4]}</td>
                            <td>{metric[5]}</td>
                        </tr>
                    })}
                </tbody>
            </table><br /><br /></>}
            {metrics.length != 0  && <input type = "file" id = "classificationOutputFile" />}
            {metrics.length != 0  && <input type = "button" value = "Run" onClick = {runModel}/>}
            {graph && <img src = {graph} alt = "graph"/>}
        </form>
    );
}