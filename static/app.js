function App() {
    let active = "edaul"

    function setNav(element, x) {
        document.getElementById(x).hidden = true;
        document.getElementById(element).hidden = false;
        active = element;
    }

    return(
        <>
            <header>MSIL Accelerator</header>
            <div id = "mainContainer">
                <aside>
                    <Navigation navigator = {setNav} active = {active}/>
                </aside>
                <main>
                    <EDASL id = "edasl" />
                    <EDAUL id = "edaul"/>
                    <Regression id = "regression" />
                    <Classification id = "classification" />
                    <TimeSeries id = "timeseries" />
                    <Clustering id = "clustering" />
                </main>
            </div>
        </>
    );
}

let rootNode = document.getElementById('root');
let root = ReactDOM.createRoot(rootNode);
root.render(<App />);