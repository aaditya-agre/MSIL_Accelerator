function App() {
    let active = "edasl";

    function setNav(element) {
        document.getElementById(active).style.display = 'none';
        document.getElementById(element).style.display = 'block';
        active = element;
    }

    return(
        <>
            <header>MSIL Accelerator</header>
            <div id = "mainContainer">
                <aside>
                    <ul>
                        <li onClick={() => setNav('edasl')}>Exploratory Data Analysis Supervised Learning</li>
                        <li onClick={() => setNav('classification')}>Classification</li>
                        <li onClick={() => setNav('regression')}>Regression</li>
                        <li onClick={() => setNav('timeseries')}>Time Series</li>
                        <li onClick={() => setNav('clustering')}>Clustering</li>
                    </ul>
                </aside>
                <main>
                    <div id = "edasl" style = {{display: 'none'}}><EDASL/></div>
                    <div id = "regression" style = {{display: 'none'}}><Regression/></div>
                    <div id = "classification" style = {{display: 'none'}}><Classification/></div>
                    <div id = "timeseries" style = {{display: 'none'}}><TimeSeries/></div>
                    <div id = "clustering" style = {{display: 'none'}}><Clustering/></div>
                </main>
            </div>
        </>
    );
}

let rootNode = document.getElementById('root');
let root = ReactDOM.createRoot(rootNode);
root.render(<App />);