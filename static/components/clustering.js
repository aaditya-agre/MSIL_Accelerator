const {useState} = React;

function Clustering() {
    const [cols, setCols] = useState(null);
    const [metrics, setMetrics] = useState([]);
    const [graph, setGraph] = useState(null);

    async function useEDA() {
        await fetch('/eda/clustering').then(response => response.json()).then(result => setCols(result['cols']));
    }

    async function uploadFile() {
        const data = new FormData();
        const f = document.getElementById('clusteringTrainFile');
        data.append('dataset', f.files[0]);
        await fetch('/upload/clustering', {
            method: 'POST',
            body: data
        }).then(response => response.json()).then(result => setCols(result['cols']));
    }

    async function trainModel() {
        let data = {'model' : document.getElementById('clusterModel').value,
                'tuning': document.getElementById('clusterTuning').value
        }

        await fetch('/train/clustering', {
            method: 'POST',
            body: JSON.stringify(data)
        }).then(response => response.formData()).then(formData => {
            let result = JSON.parse(formData.get('metrics'));
            setMetrics([...metrics, [result['silhouette_score'], result['calinski_harabasz_score'], result['davies_bouldin_score'], result['num_clusters_found'], data['model'], result['num']]]);
            setGraph(URL.createObjectURL(formData.get('graph')));
            saveAs(formData.get('output'), 'clusters.csv');
        });
    }

    return(
        <form>
            <input type = "file" id = "clusteringTrainFile"/>
            <input type = "button" value = "Upload" onClick = {uploadFile}/><br /><br />
            <label>Use data from EDA: </label><input type = "checkbox" onChange = {useEDA}/><br /><br />
            <label>Model: </label>
            <select id = "clusterModel">
                <option value = "kmeans">KMeans</option>
                <option value = "dbscan">DBScan</option>
                <option value = "gmm">Gaussian Mixture</option>
                <option value = "birch">Birch</option>
            </select><br /><br />
            <label>Hyperparameter Tuning: </label>
            <select id = "clusterTuning"> 
                <option>GridSearchCV</option>
                <option>RandomizedSearchCV</option>
            </select><br /><br />
            {cols && <><input type = "button" value = "Train" onClick = {trainModel}/><br /><br /></>}
            {metrics.length != 0 && <><table>
                <thead>
                    <tr>
                        <th>Silhoutte Score</th>
                        <th>Calinski Harabasz Score</th>
                        <th>Davies Bouldin Score</th>
                        <th>Number of clusters found</th>
                        <th>Model</th>
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
            {graph && <img src = {graph} alt = "graph"/>}
        </form>
    );
}