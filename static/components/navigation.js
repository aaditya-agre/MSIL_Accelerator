function Navigation({navigator, active}) {
    
    return (
        <ul>
            <li onClick={() => navigator('edasl', active)}>Exploratory Data Analysis Supervised Learning</li>
            <li onClick={() => navigator('edaul', active)}>Exploratory Data Analysis Unsupervised Learning</li>
            <li onClick={() => navigator('classification', active)}>Classification</li>
            <li onClick={() => navigator('regression', active)}>Regression</li>
            <li onClick={() => navigator('timeseries', active)}>Time Series</li>
            <li onClick={() => navigator('clustering', active)}>Clustering</li>
        </ul>
    );
}