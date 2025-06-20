const {useState} = React;

function EDASL() {
    const [information, informer] = React.useState(null);
    const [selectedColumn, setSelectedColumn] = React.useState('');
    const [dataType, setDataType] = React.useState('categorical');
    const [plotType, setPlotType] = React.useState('countplot');
    const [plotData, setPlotData] = React.useState(null);

    // Upload CSV and get descriptive statistics
    async function uploadFile() {
        const data = new FormData();
        const f = document.getElementById('EDASLUploadFile');
        data.append('dataset', f.files[0]);
        const response = await fetch('/upload/edasl', {
            method: 'POST',
            body: data
        });
        const result = await response.json();
        informer(result);
    }

    // Generate Univariate Plot
    async function generateUnivariatePlot() {
        const response = await fetch('/univariate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                column: selectedColumn,
                type1: dataType,
                type2: plotType
            })
        });

        const result = await response.json();
        setPlotData(result);
    }

    // Render chart after receiving data
    React.useEffect(() => {
        if (!plotData || !plotData.type) return;

        const layout = {
            title: `${plotData.type} of ${selectedColumn}`,
            margin: { t: 40 },
        };

        if (plotData.type === 'countplot' || plotData.type === 'piechart') {
            const labels = Object.keys(plotData.data);
            const values = Object.values(plotData.data);

            if (plotData.type === 'countplot') {
                Plotly.newPlot("plotDiv", [{
                    x: labels,
                    y: values,
                    type: 'bar',
                    marker: { color: '#003366' }
                }], layout);
            } else {
                Plotly.newPlot("plotDiv", [{
                    labels,
                    values,
                    type: 'pie'
                }], layout);
            }

        } else if (plotData.type === 'histogram') {
            Plotly.newPlot("plotDiv", [{
                x: plotData.bins,
                y: plotData.counts,
                type: 'bar',
                marker: { color: '#0066cc' }
            }], layout);

        } else if (plotData.type === 'distplot') {
            Plotly.newPlot("plotDiv", [{
                x: plotData.data,
                type: 'scatter',
                mode: 'lines',
                line: { color: '#28a745' }
            }], layout);

        } else if (plotData.type === 'boxplot') {
            Plotly.newPlot("plotDiv", [{
                y: [...plotData.outliers, plotData.min, plotData.q1, plotData.median, plotData.q3, plotData.max],
                type: 'box',
                boxpoints: 'outliers',
                marker: { color: '#dc3545' }
            }], layout);
        }
    }, [plotData]);



    return (
        <form className="eda-form space-y-6 p-4">
            <div className="upload-section">
                <input type="file" id="EDASLUploadFile" />
                <input type="button" value="Upload" onClick={uploadFile} className="ml-2 px-4 py-1 bg-blue-600 text-white rounded" />
            </div>

            {information && <Details info={information} />}

            {/* === Univariate Analysis Section === */}
            {information &&
                <div className="variate-ui">
                    <h3 className="font-semibold text-lg">Univariate Analysis</h3>

                    <select value={selectedColumn} onChange={(e) => setSelectedColumn(e.target.value)}>
                        <option value="">-- Select Column --</option>
                        {information.cols.map(col => (
                            <option key={col} value={col}>{col}</option>
                        ))}
                    </select>

                    <select value={dataType} onChange={(e) => {
                        setDataType(e.target.value);
                        setPlotType(e.target.value === 'categorical' ? 'countplot' : 'histogram');
                    }}>
                        <option value="categorical">Categorical</option>
                        <option value="numerical">Numerical</option>
                    </select>

                    <select value={plotType} onChange={(e) => setPlotType(e.target.value)}>
                        {dataType === 'categorical' ? (
                            <>
                                <option value="countplot">Count Plot</option>
                                <option value="piechart">Pie Chart</option>
                            </>
                        ) : (
                            <>
                                <option value="histogram">Histogram</option>
                                <option value="distplot">Dist Plot</option>
                                <option value="boxplot">Box Plot</option>
                            </>
                        )}
                    </select>

                    <button type="button" onClick={generateUnivariatePlot} className="ml-4 px-3 py-1 bg-green-600 text-white rounded">
                        Generate Plot
                    </button>

                    <div id="plotDiv" style={{ marginTop: '30px' }}></div>
                </div>
            }

            {information && <MultivariateTool columns={information.cols} />}
            {information && console.log("information:", information)}
            {information && <MissingValueHandler information={information} />}
            {information && <OutlierHandler information={information} />}
            {information && <FeatureEncoder information={information} />}
            {information && <FeatureScaler information={information} /> }
            {information && <MixedDataHandler information={information} />}
            {information && <SplitBasedOnDelimiter information={information} />}
            {information && <FeatureTransformer information={information} />}
            {information && <ManualFeatureSelector />}
            {information && <AutomatedFeatureSelectorExtractor information={information} />}
            {information &&
    <a
        href="/download"
        className="download-button"
        download
    >
        Download Preprocessed CSV
    </a>
}
        </form>
    );
}


function Details({ info }) {
    const rows = info['cols'].map(col => {
        const colinfo = info['describe'][col];
        return (
            <tr key={col}>
                <td>{col}</td>
                <td>{colinfo['25%']}</td>
                <td>{colinfo['50%']}</td>
                <td>{colinfo['75%']}</td>
                <td>{colinfo['count']}</td>
                <td>{colinfo['freq']}</td>
                <td>{colinfo['max']}</td>
                <td>{colinfo['mean']}</td>
                <td>{colinfo['min']}</td>
                <td>{colinfo['std']}</td>
                <td>{colinfo['top']}</td>
                <td>{colinfo['unique']}</td>
                <td>{info['null values'][col]}</td>
            </tr>
        );
    });

    return (
        <div className="eda-details">
            <h3><strong>Data Overview</strong></h3>
            <p><strong>{info['shape'][0]}</strong> rows × <strong>{info['shape'][1]}</strong> columns</p>
            <p><strong>{info['duplicates']}</strong> duplicate(s) were found.</p>
            <div className="table-container">
                <table className="eda-table">
                    <thead>
                        <tr>
                            <th>Column</th>
                            <th>25%</th>
                            <th>50%</th>
                            <th>75%</th>
                            <th>Count</th>
                            <th>Frequency</th>
                            <th>Max</th>
                            <th>Mean</th>
                            <th>Min</th>
                            <th>Standard Deviation</th>
                            <th>Top</th>
                            <th>Unique</th>
                            <th>Null Values</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
        </div>
    );
}


function MultivariateTool({ columns }) {
    const [type1, setType1] = React.useState('numerical');
    const [type2, setType2] = React.useState('numerical');
    const [type3, setType3] = React.useState('scatterplot');
    const [x, setX] = React.useState('');
    const [y, setY] = React.useState('');
    const [cols, setCols] = React.useState([]);
    const [plotData, setPlotData] = React.useState(null);

    async function fetchMultivariate() {
        const payload = {
            type3,
            no_of_col_to_do_analysis: cols.length,
            chosen_cols: {}
        };

        if (type3 === 'pairplot') {
            payload.chosen_cols.cols = cols;
        } else {
            payload.type1 = type1;
            payload.type2 = type2;
            payload.chosen_cols.x = x;
            payload.chosen_cols.y = y;
        }

        const response = await fetch('/multivariate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();
        setPlotData(result);
    }

    React.useEffect(() => {
        if (!plotData || !plotData.type) return;
        const container = document.getElementById("multiPlotDiv");
        container.innerHTML = "";

        const layout = { title: `${plotData.type}` };

        if (plotData.type === 'scatterplot' || plotData.type === 'lineplot') {
            Plotly.newPlot("multiPlotDiv", [{
                x: plotData.x,
                y: plotData.y,
                type: 'scatter',
                mode: plotData.type === 'scatterplot' ? 'markers' : 'lines+markers',
                marker: { color: '#007bff' }
            }], layout);

        } else if (plotData.type === 'barplot') {
            Plotly.newPlot("multiPlotDiv", [{
                x: plotData.labels,
                y: plotData.values,
                type: 'bar',
                marker: { color: '#28a745' }
            }], layout);

        } else if (plotData.type === 'pairplot') {
            plotData.cols.forEach(colX => {
                plotData.cols.forEach(colY => {
                    const xData = plotData.rows.map(row => row[colX]);
                    const yData = plotData.rows.map(row => row[colY]);

                    const plotId = `${colX}_${colY}_plot`;
                    const plotDiv = document.createElement("div");
                    plotDiv.id = plotId;
                    plotDiv.style.marginBottom = "40px";
                    container.appendChild(plotDiv);

                    Plotly.newPlot(plotId, [{
                        x: xData,
                        y: yData,
                        mode: 'markers',
                        type: 'scatter',
                        name: `${colX} vs ${colY}`,
                        marker: { color: '#17a2b8' }
                    }], {
                        title: `${colX} vs ${colY}`,
                        height: 400,
                        width: 400
                    });
                });
            });

        } else if (plotData.type === 'heatmap') {
            Plotly.newPlot("multiPlotDiv", [{
                z: plotData.matrix,
                x: plotData.xLabels,
                y: plotData.yLabels,
                type: 'heatmap',
                colorscale: 'Viridis'
            }], layout);

        } else if (plotData.type === 'clustermap') {
            Plotly.newPlot("multiPlotDiv", [{
                z: plotData.matrix,
                x: plotData.xLabels,
                y: plotData.yLabels,
                type: 'heatmap',
                colorscale: 'Cividis'
            }], layout);
        }

    }, [plotData]);

    return (
        <div className="variate-ui">
            <h3>Bivariate Analysis</h3>

            <label>Type 3 (Plot Type):</label>
            <select value={type3} onChange={(e) => {
                setType3(e.target.value);
                setX('');
                setY('');
                setCols([]);
            }}>
                <option value="scatterplot">Scatterplot</option>
                <option value="lineplot">Lineplot</option>
                <option value="barplot">Barplot</option>
                <option value="boxplot">Boxplot</option>
                <option value="displot">Displot</option>
                <option value="pairplot">Pairplot</option>
                <option value="heatmap">Heatmap</option>
                <option value="clustermap">Clustermap</option>
            </select>

            {type3 !== 'pairplot' && (
                <>
                    <label>Type 1:</label>
                    <select value={type1} onChange={(e) => setType1(e.target.value)}>
                        <option value="numerical">Numerical</option>
                        <option value="categorical">Categorical</option>
                    </select>

                    <label>Type 2:</label>
                    <select value={type2} onChange={(e) => setType2(e.target.value)}>
                        <option value="numerical">Numerical</option>
                        <option value="categorical">Categorical</option>
                    </select>

                    <label>X Axis:</label>
                    <select value={x} onChange={(e) => setX(e.target.value)}>
                        <option value="">Select X</option>
                        {columns.map(col => <option key={col}>{col}</option>)}
                    </select>

                    <label>Y Axis:</label>
                    <select value={y} onChange={(e) => setY(e.target.value)}>
                        <option value="">Select Y</option>
                        {columns.map(col => <option key={col}>{col}</option>)}
                    </select>
                </>
            )}

            {type3 === 'pairplot' && (
                <>
                    <label>Select Columns for Pairplot:</label>
                    <select multiple onChange={(e) => {
                        const selected = Array.from(e.target.selectedOptions, o => o.value);
                        setCols(selected);
                    }}>
                        {columns.map(col => <option key={col}>{col}</option>)}
                    </select>
                </>
            )}

            <button
                type="button"
                onClick={(e) => {
                    e.preventDefault();
                    fetchMultivariate();
                }}
            >
                Generate Multivariate Plot
            </button>

            <div id="multiPlotDiv" style={{ marginTop: '30px' }}></div>
        </div>
    );
}


function MissingValueHandler({ information }) {
    if (!information || !information.cols || !information['null values'] || !information.types)
        return null;

    const [imputationMethods, setImputationMethods] = useState({});
    const [submitting, setSubmitting] = useState(false);

    const categoricalMethods = [
        'drop', 'mode', 'missing', 'ffill', 'bfill',
        'random', 'missing_indicator', 'knn', 'iterative'
    ];
    const numericalMethods = [
        'drop', 'mean', 'median', 'arbitrary', 'random',
        'end_of_distribution', 'knn', 'iterative'
    ];

    const inferType = (dtype) => {
        const numTypes = ['int64', 'float64', 'int32', 'float32', 'int', 'float'];
        return numTypes.includes(dtype.toLowerCase()) ? 'numerical' : 'categorical';
    };

    const handleMethodChange = (col, method) => {
        setImputationMethods(prev => ({
            ...prev,
            [col]: method
        }));
    };

    const handleSubmit = async () => {
        setSubmitting(true);
        const payload = [];

        for (let col of information.cols) {
            const nullCount = information['null values'][col] || 0;
            const dtype = information.types[col];
            const inferredType = inferType(dtype);

            if (nullCount > 0) {
                payload.push({
                    column: col,
                    type: inferredType,
                    method: imputationMethods[col] || 'drop'
                });
            }
        }

        try {
            const res = await fetch('/handle-missing', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ impute: payload })
            });
            const result = await res.json();
            console.log("Imputation result:", result);
            alert("Imputation submitted successfully!");
        } catch (error) {
            console.error("Error submitting imputation:", error);
            alert("Error during submission. Check console.");
        } finally {
            setSubmitting(false);
        }
    };

    return (
        <div className="mt-8 variate-ui">
            <h3 className="text-xl font-semibold mb-4">Handle Missing Values</h3>
            <table className="w-full table-auto border border-gray-300">
                <thead className="bg-gray-100">
                    <tr>
                        <th className="border px-6 py-2 text-left">Column Name</th>
                        <th className="border px-6 py-2 text-left">Inferred Type</th>
                        <th className="border px-6 py-2 text-left">Null Exists</th>
                        <th className="border px-6 py-2 text-left">Imputation Method</th>
                    </tr>
                </thead>
                <tbody>
                    {information.cols.map(col => {
                        const nullCount = information['null values'][col] || 0;
                        const dtype = information.types[col];
                        const inferredType = inferType(dtype);
                        const methods = inferredType === 'numerical' ? numericalMethods : categoricalMethods;

                        return (
                            <tr key={col}>
                                <td className="border px-4 py-2">{col}</td>
                                <td className="border px-4 py-2 capitalize">{inferredType}</td>
                                <td className="border px-4 py-2">{nullCount > 0 ? 'Yes' : 'No'}</td>
                                <td className="border px-4 py-2">
                                    {nullCount > 0 ? (
                                        <select
                                            className="border rounded px-2 py-1"
                                            value={imputationMethods[col] || 'drop'}
                                            onChange={(e) => handleMethodChange(col, e.target.value)}
                                        >
                                            {methods.map(method => (
                                                <option key={method} value={method}>{method}</option>
                                            ))}
                                        </select>
                                    ) : (
                                        <span className="text-gray-400 italic">No action needed</span>
                                    )}
                                </td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>

            <div className="mt-4">
                <button
                    onClick={handleSubmit}
                    disabled={submitting}
                    className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
                >
                    {submitting ? 'Submitting...' : 'Submit'}
                </button>
            </div>
        </div>
    );
}

function OutlierHandler({ information }) {
    if (!information || !information.cols || !information.types) return null;

    const [config, setConfig] = useState({});
    const [submitting, setSubmitting] = useState(false);

    const numericalCols = information.cols.filter(col => {
        const dtype = information.types[col];
        return ['int64', 'float64', 'int32', 'float32', 'int', 'float'].includes(dtype?.toLowerCase());
    });

    const handleChange = (col, field, value) => {
        setConfig(prev => ({
            ...prev,
            [col]: {
                ...prev[col],
                [field]: value
            }
        }));
    };

    const handleSubmit = async () => {
        const payload = [];

        for (let col of numericalCols) {
            const method = config[col]?.method || 'zscore';
            if (method === 'na') continue; // Skip columns marked as NA

            const strategy = config[col]?.strategy || 'trimming';
            const thresholdInput = config[col]?.threshold || '3';
            const threshold = method === 'percentile'
                ? thresholdInput.split(',').map(v => parseFloat(v.trim()))
                : parseFloat(thresholdInput);

            payload.push({ column: col, method, strategy, threshold });
        }

        try {
            setSubmitting(true);
            const res = await fetch('/remove-outliers', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ outliers: payload })
            });
            const result = await res.json();
            console.log('Outlier handling result:', result);
            alert("Outlier handling submitted successfully!");
        } catch (err) {
            console.error('Submission error:', err);
            alert("Error during submission.");
        } finally {
            setSubmitting(false);
        }
    };

    return (
        <div className="mt-8 variate-ui">
            <h3 className="text-xl font-semibold mb-4">Handle Outliers</h3>
            <table className="w-full table-auto border border-gray-300">
                <thead className="bg-gray-100">
                    <tr>
                        <th className="border px-4 py-2 text-left">Column</th>
                        <th className="border px-4 py-2 text-left">Method</th>
                        <th className="border px-4 py-2 text-left">Strategy</th>
                        <th className="border px-4 py-2 text-left">Threshold</th>
                    </tr>
                </thead>
                <tbody>
                    {numericalCols.map(col => {
                        const method = config[col]?.method || 'zscore';
                        const isDisabled = method === 'na';

                        return (
                            <tr key={col}>
                                <td className="border px-4 py-2">{col}</td>
                                <td className="border px-4 py-2">
                                    <select
                                        className="border rounded px-2 py-1"
                                        value={method}
                                        onChange={(e) => handleChange(col, 'method', e.target.value)}
                                    >
                                        <option value="zscore">zscore</option>
                                        <option value="iqr">iqr</option>
                                        <option value="percentile">percentile</option>
                                        <option value="na">NA</option>
                                    </select>
                                </td>
                                <td className="border px-4 py-2">
                                    <select
                                        className="border rounded px-2 py-1"
                                        value={config[col]?.strategy || 'trimming'}
                                        onChange={(e) => handleChange(col, 'strategy', e.target.value)}
                                        disabled={isDisabled}
                                    >
                                        <option value="trimming">trimming</option>
                                        <option value="capping">capping</option>
                                    </select>
                                </td>
                                <td className="border px-4 py-2">
                                    <input
                                        className="border px-2 py-1 w-full"
                                        type="text"
                                        placeholder={method === 'percentile' ? 'e.g. 1,99' : 'e.g. 3'}
                                        value={config[col]?.threshold || ''}
                                        onChange={(e) => handleChange(col, 'threshold', e.target.value)}
                                        disabled={isDisabled}
                                    />
                                </td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>

            <div className="mt-4">
                <button
                    onClick={handleSubmit}
                    disabled={submitting}
                    className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700"
                >
                    {submitting ? 'Submitting...' : 'Submit Outlier Handling'}
                </button>
            </div>
        </div>
    );
}


function FeatureEncoder({ information }) {
    const [configs, setConfigs] = useState({});
    const [submitting, setSubmitting] = useState(false);

    if (!information || !information.cols || !information.types) return null;

    const currentTargetCol = Object.entries(configs).find(([_, cfg]) => cfg?.is_target)?.[0];

    const handleChange = (col, field, value) => {
        setConfigs(prev => {
            const updated = { ...prev };

            // If one is marked as target, unmark others
            if (field === 'is_target' && value === true) {
                Object.keys(updated).forEach(key => {
                    if (!updated[key]) updated[key] = {};
                    updated[key].is_target = false;
                });
            }

            if (!updated[col]) updated[col] = {};
            updated[col][field] = value;
            return updated;
        });
    };

    
    const handleSubmit = async () => {
    try {
        setSubmitting(true);
        const payload = Object.entries(configs).map(([col, config]) => {
            const type = ['int', 'float', 'int64', 'float64'].includes(information.types[col]?.toLowerCase())
                ? 'numerical'
                : 'categorical';

            return {
                column: col,
                ftype: type,          // Include ftype here
                ...config
            };
        });

        const response = await fetch('/feature-encoding', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();
        console.log('Encoding result:', result);
        alert('Feature encoding submitted successfully!');
    } catch (err) {
        console.error('Error submitting encoding:', err);
        alert('Failed to encode features.');
    } finally {
        setSubmitting(false);
    }
};


    return (
        <div className="mt-8 variate-ui">
            <h3 className="text-xl font-semibold mb-4">Feature Encoding</h3>

            <table className="w-full border">
                <thead>
                    <tr className="bg-gray-200">
                        <th className="p-2 border">Column</th>
                        <th className="p-2 border">Type</th>
                        <th className="p-2 border">Target (Y/N)</th>
                        <th className="p-2 border">Method</th>
                        <th className="p-2 border">Sub-method</th>
                        <th className="p-2 border">Strategy</th>
                        <th className="p-2 border">Bins</th>
                    </tr>
                </thead>
                <tbody>
                    {information.cols.map(col => {
                        const type = ['int', 'float', 'int64', 'float64'].includes(information.types[col]?.toLowerCase())
                            ? 'numerical'
                            : 'categorical';
                        const config = configs[col] || {};

                        return (
                            <tr key={col}>
                                <td className="p-2 border">{col}</td>
                                <td className="p-2 border text-center">{type}</td>
                                <td className="p-2 border text-center">
                                    <input
                                        type="checkbox"
                                        checked={!!config.is_target}
                                        onChange={e => handleChange(col, 'is_target', e.target.checked)}
                                        disabled={!!currentTargetCol && currentTargetCol !== col}
                                    />
                                </td>
                                <td className="p-2 border">
                                    <select
                                        className="w-full p-1 border rounded"
                                        onChange={e => handleChange(col, 'method', e.target.value)}
                                        value={config.method || ''}
                                    >
                                        <option value="">Select</option>
                                        {type === 'numerical' && (
                                            <>
                                                <option value="discretization">Discretization</option>
                                                <option value="binarization">Binarization</option>
                                            </>
                                        )}
                                        {type === 'categorical' && (
                                            <>
                                                <option value="ordinal_input">Ordinal Input</option>
                                                <option value="ordinal_output">Ordinal Output</option>
                                                <option value="nominal">Nominal (One-Hot)</option>
                                            </>
                                        )}
                                    </select>
                                </td>

                                <td className="p-2 border">
                                    {type === 'numerical' && config.method === 'discretization' && (
                                        <select
                                            className="w-full p-1 border rounded"
                                            onChange={e => handleChange(col, 'sub_method', e.target.value)}
                                            value={config.sub_method || ''}
                                        >
                                            <option value="">Select</option>
                                            <option value="unsupervised">Unsupervised</option>
                                            <option value="supervised">Supervised</option>
                                        </select>
                                    )}
                                </td>

                                <td className="p-2 border">
                                    {type === 'numerical' && config.sub_method === 'unsupervised' && (
                                        <select
                                            className="w-full p-1 border rounded"
                                            onChange={e => handleChange(col, 'strategy', e.target.value)}
                                            value={config.strategy || ''}
                                        >
                                            <option value="">Select</option>
                                            <option value="uniform">Uniform</option>
                                            <option value="quantile">Quantile</option>
                                            <option value="kmeans">KMeans</option>
                                        </select>
                                    )}
                                </td>

                                <td className="p-2 border text-center">
                                    {type === 'numerical' && config.method === 'discretization' && (
                                        <input
                                            type="number"
                                            min={2}
                                            value={config.bins || 5}
                                            onChange={e => handleChange(col, 'bins', parseInt(e.target.value))}
                                            className="w-full p-1 border rounded"
                                        />
                                    )}
                                </td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>

            <button
                className="mt-6 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
                onClick={handleSubmit}
                disabled={submitting}
            >
                {submitting ? 'Submitting...' : 'Submit Feature Encoding'}
            </button>
        </div>
    );
}



function FeatureScaler({ information }) {
    const [configs, setConfigs] = useState({});
    const [submitting, setSubmitting] = useState(false);

    if (!information || !information.cols || !information.types) return null;

    const handleChange = (col, field, value) => {
    setConfigs(prev => {
        const updated = { ...prev };
        if (!updated[col]) updated[col] = {};

        updated[col][field] = value;

        // Auto set zscore if method is standardization
        if (field === 'method' && value === 'standardization') {
            updated[col].strategy = 'zscore';
        }

        return updated;
    });
};


    const handleSubmit = async () => {
        try {
            setSubmitting(true);

            const payload = Object.entries(configs)
    .filter(([_, config]) => config.method) // Only send configured
    .map(([col, config]) => ({
        column: col,
        ...config
    }));


            const response = await fetch('/feature-scaling', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await response.json();
            console.log('Scaling result:', result);
            alert('Feature scaling applied successfully!');
        } catch (err) {
            console.error('Error applying scaling:', err);
            alert('Failed to scale features.');
        } finally {
            setSubmitting(false);
        }
    };

    return (
        <div className="mt-10 variate-ui">
            <h3 className="text-xl font-semibold mb-4">Feature Scaling</h3>

            <table className="w-full border">
                <thead>
                    <tr className="bg-gray-200">
                        <th className="p-2 border">Column</th>
                        <th className="p-2 border">Method</th>
                        <th className="p-2 border">Strategy</th>
                    </tr>
                </thead>
                <tbody>
                    {information.cols.map(col => {
                        const isNumerical = ['int', 'float', 'int64', 'float64'].includes(information.types[col]?.toLowerCase());
                        if (!isNumerical) return null;

                        const config = configs[col] || {};
                        return (
                            <tr key={col}>
                                <td className="p-2 border">{col}</td>
                                <td className="p-2 border">
                                    <select
                                        className="w-full p-1 border rounded"
                                        onChange={e => handleChange(col, 'method', e.target.value)}
                                        value={config.method || ''}
                                    >
                                        <option value="">Select</option>
                                        <option value="standardization">Standardization</option>
                                        <option value="normalization">Normalization</option>
                                    </select>
                                </td>
                                <td className="p-2 border">
    {config.method === 'standardization' && (
        <div className="text-center">Z-score</div>
    )}

                                    {config.method === 'normalization' && (
                                        <select
                                            className="w-full p-1 border rounded"
                                            value={config.strategy || ''}
                                            onChange={e => handleChange(col, 'strategy', e.target.value)}
                                        >
                                            <option value="">Select</option>
                                            <option value="minmax">Min-Max</option>
                                            <option value="mean">Mean Norm (L2)</option>
                                            <option value="max_abs">Max Abs</option>
                                            <option value="robust">Robust</option>
                                        </select>
                                    )}
                                </td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>

            <button
                className="mt-6 bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700"
                onClick={handleSubmit}
                disabled={submitting}
            >
                {submitting ? 'Submitting...' : 'Apply Scaling'}
            </button>
        </div>
    );
}

function MixedDataHandler({ information }) {
    const [selectedColumn, setSelectedColumn] = useState('');
    const [mixedType, setMixedType] = useState('');
    const [submitting, setSubmitting] = useState(false);
    const [columns, setColumns] = useState(information.cols);
    if (!information || !information.cols) return null;

    const handleSubmit = async () => {
        if (!selectedColumn || !mixedType) {
            alert('Please select both column and type.');
            return;
        }

        try {
            setSubmitting(true);

            const payload = {
                column: selectedColumn,
                mixed_type: mixedType
            };

            const response = await fetch('/mixed-data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await response.json();
            
            console.log('Mixed data result:', result);
            alert('Mixed data processed successfully!');
        } catch (err) {
            console.error('Error handling mixed data:', err);
            alert('Failed to handle mixed data.');
        } finally {
            setSubmitting(false);
        }
    };

    return (
        <div className="mt-10  variate-ui">
            <h3 className="text-xl font-semibold mb-4">Handle Mixed Data Column</h3>

            <div className="mb-4">
                <label className="block mb-1 font-medium">Select Column:</label>
                <select
                    className="w-full p-2 border rounded"
                    value={selectedColumn}
                    onChange={e => setSelectedColumn(e.target.value)}
                >
                    <option value="">Select a column</option>
                    {information.cols.map(col => (
                        <option key={col} value={col}>
                            {col}
                        </option>
                    ))}
                </select>
            </div>

            <div className="mb-4">
                <label className="block mb-1 font-medium">Mixed Data Type:</label>
                <select
                    className="w-full p-2 border rounded"
                    value={mixedType}
                    onChange={e => setMixedType(e.target.value)}
                >
                    <option value="">Select a type</option>
                    <option value="type1">Type 1 - 'C45', 'D32'</option>
                    <option value="type2">Type 2 - Interleaved values (A, 1, B, 2)</option>
                </select>
            </div>

            <button
                className="bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700"
                onClick={handleSubmit}
                disabled={submitting}
            >
                {submitting ? 'Submitting...' : 'Handle Mixed Data'}
            </button>
        </div>
    );
}

function SplitBasedOnDelimiter({ information }) {
    const [selectedColumn, setSelectedColumn] = React.useState('');
    const [delimiter, setDelimiter] = React.useState('');
    const [submitting, setSubmitting] = React.useState(false);

    if (!information || !information.cols) return null;

    const handleSubmit = async () => {
        if (!selectedColumn || !delimiter) {
            alert('Please select a column and enter a delimiter.');
            return;
        }

        try {
            setSubmitting(true);

            const payload = {
                column: selectedColumn,
                delimiter: delimiter
            };

            const response = await fetch('/split_column', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await response.json();
            console.log('Split column result:', result);
            alert('Column split successfully!');
        } catch (error) {
            console.error('Error splitting column:', error);
            alert('Failed to split column.');
        } finally {
            setSubmitting(false);
        }
    };

    return (
        <div className="mt-10 variate-ui">
            <h3 className="text-xl font-semibold mb-4">Split Column Based on Delimiter</h3>

            <div className="mb-4">
                <label className="block mb-1 font-medium">Select Column:</label>
                <select
                    className="w-full p-2 border rounded"
                    value={selectedColumn}
                    onChange={e => setSelectedColumn(e.target.value)}
                >
                    <option value="">Select a column</option>
                    {information.cols.map(col => (
                        <option key={col} value={col}>
                            {col}
                        </option>
                    ))}
                </select>
            </div>

            <div className="mb-4">
                <label className="block mb-1 font-medium">Enter Delimiter:</label>
                <input
                    type="text"
                    className="w-full p-2 border rounded"
                    placeholder="e.g. - , ; |"
                    value={delimiter}
                    onChange={e => setDelimiter(e.target.value)}
                />
            </div>

            <button
                className="bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700"
                onClick={handleSubmit}
                disabled={submitting}
            >
                {submitting ? 'Splitting...' : 'Split Column'}
            </button>
        </div>
    );
}




function FeatureTransformer({ information }) {
    const [selectedColumn, setSelectedColumn] = useState('');
    const [type1, setType1] = useState('');
    const [type2, setType2] = useState('');
    const [submitting, setSubmitting] = useState(false);

    if (!information || !information.cols || !information.types) return null;

    const numericCols = information.cols.filter(
        col => ['int', 'float', 'int64', 'float64'].includes(information.types[col]?.toLowerCase())
    );

    const handleSubmit = async () => {
        if (!selectedColumn || !type1 || !type2) {
            alert('Please select all fields.');
            return;
        }

        try {
            setSubmitting(true);
            const payload = {
                column: selectedColumn,
                type1,
                type2
            };

            const response = await fetch('/feature-transform', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await response.json();
            console.log('Feature transformation result:', result);
            alert('Feature transformation successful!');
        } catch (err) {
            console.error('Error transforming feature:', err);
            alert('Failed to transform feature.');
        } finally {
            setSubmitting(false);
        }
    };

    return (
        <div className="mt-10 variate-ui">
            <h3 className="text-xl font-semibold mb-4">Feature Transformation</h3>

            <div className="mb-4">
                <label className="block font-medium mb-1">Select Column</label>
                <select
                    className="w-full p-2 border rounded"
                    value={selectedColumn}
                    onChange={e => setSelectedColumn(e.target.value)}
                >
                    <option value="">Select column</option>
                    {numericCols.map(col => (
                        <option key={col} value={col}>
                            {col}
                        </option>
                    ))}
                </select>
            </div>

            <div className="mb-4">
                <label className="block font-medium mb-1">Type of Transformation</label>
                <select
                    className="w-full p-2 border rounded"
                    value={type1}
                    onChange={e => {
                        setType1(e.target.value);
                        setType2(''); // reset type2 on type1 change
                    }}
                >
                    <option value="">Select type</option>
                    <option value="function">Function-Based</option>
                    <option value="power">Power-Based</option>
                </select>
            </div>

            {type1 && (
                <div className="mb-4">
                    <label className="block font-medium mb-1">Transformation Method</label>
                    <select
                        className="w-full p-2 border rounded"
                        value={type2}
                        onChange={e => setType2(e.target.value)}
                    >
                        <option value="">Select method</option>
                        {type1 === 'function' && (
                            <>
                                <option value="log">Log</option>
                                <option value="reciprocal">Reciprocal</option>
                                <option value="square">Square</option>
                                <option value="sqrt">Square Root</option>
                            </>
                        )}
                        {type1 === 'power' && (
                            <>
                                <option value="boxcox">Box-Cox</option>
                                <option value="yeojohnson">Yeo-Johnson</option>
                            </>
                        )}
                    </select>
                </div>
            )}

            <button
                className="mt-4 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
                onClick={handleSubmit}
                disabled={submitting}
            >
                {submitting ? 'Submitting...' : 'Apply Transformation'}
            </button>
        </div>
    );
}

function ManualFeatureSelector() {
    const [selectedTarget, setSelectedTarget] = useState('');
    const [selectedFeatures, setSelectedFeatures] = useState([]);
    const [submitting, setSubmitting] = useState(false);
    const [newCols, setNewCols] = useState(null);

    async function getNewCols() {
        await fetch('/eda/newcols').then(response => response.json()).then(result => setNewCols(result['cols']));
    }

    const handleFeatureToggle = (col) => {
        setSelectedFeatures(prev =>
            prev.includes(col)
                ? prev.filter(item => item !== col)
                : [...prev, col]
        );
    };

    const handleSubmit = async () => {
        if (!selectedTarget || selectedFeatures.length === 0) {
            alert("Please select a target and at least one feature.");
            return;
        }

        try {
            setSubmitting(true);
            const response = await fetch('/manual-feature-selection', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    target: selectedTarget,
                    selected_features: selectedFeatures
                })
            });

            const result = await response.json();
            if (result.error) throw new Error(result.error);
            alert("Manual feature selection applied successfully.");
        } catch (error) {
            console.error("Manual Feature Selection Error:", error);
            alert("Failed to apply manual feature selection.");
        } finally {
            setSubmitting(false);
        }
    };

    const handleTargetChange = (e) => {
        setSelectedTarget(e.target.value);
    }

    return (<> {newCols ?
        <div className="mt-10 variate-ui">
            <h3 className="text-xl font-semibold mb-4">Manual Feature Selection</h3>

            <div className="mb-4">
                <label className="block mb-2 font-medium">Select Target Column:</label>
                <select
                    className="w-full border p-2 rounded"
                    value={selectedTarget}
                    onChange={handleTargetChange}
                >
                    <option value="">-- Select Target --</option>
                    {newCols.map(col => (
                        <option key={col} value={col}>{col}</option>
                    ))}
                </select>
            </div>

            <div className="mb-4">
                <label className="block mb-2 font-medium">Select Feature Columns:</label>
                {newCols.map(col => (
                    col !== selectedTarget && (
                        <div key={col} className="flex items-center mb-1">
                            <input
                                type="checkbox"
                                checked={selectedFeatures.includes(col)}
                                onChange={() => handleFeatureToggle(col)}
                                className="mr-2"
                            />
                            <label>{col}</label>
                        </div>
                    )
                ))}
            </div>

            <button
                className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
                onClick={handleSubmit}
                disabled={submitting}
            >
                {submitting ? "Submitting..." : "Apply Feature Selection"}
            </button>
        </div> : <button type = "button" onClick = {getNewCols}>Select Features</button> }</>
    );
}


function AutomatedFeatureSelectorExtractor({ information }) {
    const [target, setTarget] = useState('');
    const [methodType, setMethodType] = useState('');
    const [method, setMethod] = useState('');
    const [nFeatures, setNFeatures] = useState(3);
    const [loading, setLoading] = useState(false);

    if (!information || !information.cols || !information.types) return null;

    const handleSubmit = async () => {
        if (!target || !methodType || !method || !nFeatures) {
            alert('Please fill in all fields.');
            return;
        }

        try {
            setLoading(true);
            const response = await fetch('/feature-select-extract', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    target,
                    method_type: methodType,
                    method,
                    n_features: parseInt(nFeatures)
                })
            });

            const result = await response.json();
            console.log('Feature selection/extraction result:', result);
            alert('Feature operation successful!');
        } catch (err) {
            console.error('Error during feature selection/extraction:', err);
            alert('Failed to perform feature operation.');
        } finally {
            setLoading(false);
        }
    };

    const availableMethods = {
        selection: ['forward', 'backward'],
        extraction: ['pca', 'lda', 'tsne']
    };

    return (
        <div className="mt-10 variate-ui">
            <h3 className="text-xl font-semibold mb-4">Feature Selection / Extraction</h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                    <label className="block mb-1">Target Column:</label>
                    <select
                        className="w-full p-2 border rounded"
                        value={target}
                        onChange={e => setTarget(e.target.value)}
                    >
                        <option value="">Select Target</option>
                        {information.cols.map(col => (
                            <option key={col} value={col}>
                                {col}
                            </option>
                        ))}
                    </select>
                </div>

                <div>
                    <label className="block mb-1">Method Type:</label>
                    <select
                        className="w-full p-2 border rounded"
                        value={methodType}
                        onChange={e => {
                            setMethodType(e.target.value);
                            setMethod('');
                        }}
                    >
                        <option value="">Select</option>
                        <option value="selection">Selection</option>
                        <option value="extraction">Extraction</option>
                    </select>
                </div>

                <div>
                    <label className="block mb-1">Method:</label>
                    <select
                        className="w-full p-2 border rounded"
                        value={method}
                        onChange={e => setMethod(e.target.value)}
                        disabled={!methodType}
                    >
                        <option value="">Select Method</option>
                        {(availableMethods[methodType] || []).map(m => (
                            <option key={m} value={m}>{m}</option>
                        ))}
                    </select>
                </div>

                <div>
                    <label className="block mb-1"># of Features to Keep:</label>
                    <input
                        type="number"
                        min="1"
                        className="w-full p-2 border rounded"
                        value={nFeatures}
                        onChange={e => setNFeatures(e.target.value)}
                    />
                </div>
            </div>

            <button
                className="mt-6 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
                onClick={handleSubmit}
                disabled={loading}
            >
                {loading ? 'Processing...' : 'Apply Feature Operation'}
            </button>
        </div>
    );
}


