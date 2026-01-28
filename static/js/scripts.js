async function cargarDatos() {
    const semilla = document.getElementById('semilla').value;
    const porcentaje = document.getElementById('porcentaje').value;
    
    const respuesta = await fetch('/cargar_datos', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ semilla, porcentaje })
    });
    
    const datos = await respuesta.json();
    document.getElementById('mensaje-carga').innerHTML = 
        `<p>${datos.mensaje}. Entrenamiento: ${datos.filas_entrenamiento} filas, Prueba: ${datos.filas_prueba} filas.</p>`;
}

async function entrenarModelo() {
    const respuesta = await fetch('/entrenar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    });
    
    const resultados = await respuesta.json();
    
    // Mostrar accuracy
    document.getElementById('resultados').innerHTML = `
        <h3>Accuracy: ${resultados.accuracy.toFixed(4)}</h3>
        <div id="grafica-matriz"></div>
        <div id="grafica-dist"></div>
    `;
    
    // Renderizar gráficas Plotly
    const matrizConf = JSON.parse(resultados.matriz_confusion);
    Plotly.newPlot('grafica-matriz', matrizConf.data, matrizConf.layout);
    
    const graficaDist = JSON.parse(resultados.grafica_distribucion);
    Plotly.newPlot('grafica-dist', graficaDist.data, graficaDist.layout);
}

async function predecir() {
    // Recolectar datos del formulario
    const datos = {
        feature1: document.getElementById('feature1').value
        // Añadir más características según tu dataset
    };
    
    const respuesta = await fetch('/predecir', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(datos)
    });
    
    const resultado = await respuesta.json();
    document.getElementById('resultado-prediccion').innerHTML = 
        `<p>Predicción: ${resultado.prediccion}</p>`;
}