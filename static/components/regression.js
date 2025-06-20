const {useState} = React;

function Regression() {
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
        await fetch('/eda/regression').then(response => response.json()).then(result => setCols(result['cols']));
    }

    async function uploadFile() {
        const data = new FormData();
        const f = document.getElementById('regressionTrainFile');
        data.append('dataset', f.files[0]);
        await fetch('/upload/regression', {
            method: 'POST',
            body: data
        }).then(response => response.json()).then(result => setCols(result['cols']));
    }

    function changeSplit() {
        document.getElementById('regSplitDisplay').innerText = document.getElementById('regSplit').value;
    }

    async function trainModel() {
        let data = {'model' : document.getElementById('regModel').value,
                'split': document.getElementById('regSplit').value,
                'target': document.getElementById('regTargetVar').value, 
                'tuning': document.getElementById('regTuning').value
        }
        
        await fetch('/train/regression', {
            method: 'POST',
            body: JSON.stringify(data)
        }).then(response => response.formData()).then(formData => {
            let result = JSON.parse(formData.get('metrics'));
            setMetrics([...metrics, [result['MAE'], result['MSE'], result['R2'],  data['model'], data['split'], result['num']]]);
            setGraph(URL.createObjectURL(formData.get('graph')));
        });
    }

    async function runModel() {
        let data = new FormData();
        data.append('input', document.getElementById('regressionOutputFile').files[0]);
        await fetch('/run/regression', {
            method: 'POST',
            body: data
        }).then(response => response.blob()).then(blob => saveAs(blob, 'prediction.csv'));
    }

    return(
        <form>
            <input type = "file" id = "regressionTrainFile"/>
            <input type = "button" value = "Upload" onClick = {uploadFile}/><br /><br />
            <label>Use data from EDA: </label><input type = "checkbox" onChange = {useEDA}/><br /><br />
            <label>Model: </label>
            <select id = "regModel">
                <option value = "linear">Linear Regression</option>
                <option value = "ridge">Ridge Regression</option>
                <option value = "lasso">Lasso Regression</option>
                <option value = "decision_tree">Decision Tree Regression</option>
                <option value = "random_forest">Random Forest Regression</option>
                <option value = "svr">Support Vector Regression</option>
                <option value = "xgb">XGBoost Regression</option>
            </select><br /><br />
            <label>Hyperparameter Tuning: </label>
            <select id = "regTuning"> 
                <option>GridSearchCV</option>
                <option>RandomizedSearchCV</option>
            </select><br /><br />
            <label>Train - test split (20 - 60): </label><input type = "range" min = "20" max = "60" id = "regSplit" onChange = {changeSplit}/><span id="regSplitDisplay"></span><br /><br />
            {cols && <label>Target Variable: </label>}
            {cols && <><select id = "regTargetVar">{colOptions}</select><br /><br /></>}
            <input type = "button" value = "Train" onClick = {trainModel}/><br /><br />
            {metrics.length != 0 && <><table>
                <thead>
                    <tr>
                        <th>MAE</th>
                        <th>MSE</th>
                        <th>R2</th>
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
                        </tr>
                    })}
                </tbody>
            </table><br /><br /></>}
            {metrics.length != 0  && <input type = "file" id = "regressionOutputFile" />}
            {metrics.length != 0  && <input type = "button" value = "Run" onClick = {runModel}/>}
            {graph && <img src = {graph} alt = "graph"/>}
        </form>
    );
}