// ============================================================================
// FUNCIONES GLOBALES
// ============================================================================

const API_BASE = '';

function mostrarMensaje(elemento_id, mensaje, tipo = 'info') {
    const elemento = document.getElementById(elemento_id);
    if (elemento) {
        const color = tipo === 'error' ? '#dc3545' : '#28a745';
        elemento.innerHTML = `<p style="color: ${color}; font-weight: bold;">${mensaje}</p>`;
        elemento.style.display = 'block';
    }
}

// ============================================================================
// 1. CONFIGURAMOS LA MUESTRA
// ============================================================================

async function configurar() {
    const semilla = document.getElementById('semilla').value;
    const porcentaje = document.getElementById('porcentaje').value;
    const split = document.getElementById('split').value;
    
    console.log('📤 Enviando configuración:', {semilla, porcentaje, split});
    
    mostrarMensaje('mensaje-config', '⏳ Procesando configuración...', 'info');
    
    try {
        const response = await fetch(`${API_BASE}/configurar`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                semilla: parseInt(semilla),
                porcentaje: parseFloat(porcentaje),
                split: parseFloat(split)
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Error desconocido');
        }
        
        let mensaje = `✅ ${data.mensaje}<br>`;
        mensaje += `📊 Muestra total: <strong>${data.muestra_filas}</strong> filas<br>`;
        mensaje += `🎓 Entrenamiento: <strong>${data.entrenamiento_filas}</strong> filas<br>`;
        mensaje += `✔️ Prueba: <strong>${data.prueba_filas}</strong> filas`;
        
        mostrarMensaje('mensaje-config', mensaje, 'success');
        document.getElementById('mensaje-config').innerHTML = mensaje;
        document.getElementById('mensaje-config').style.display = 'block';
        
        console.log('✅ Configuración completada');
        
    } catch (error) {
        console.error('❌ Error:', error);
        mostrarMensaje('mensaje-config', `❌ Error: ${error.message}`, 'error');
        document.getElementById('mensaje-config').style.display = 'block';
    }
}

// ============================================================================
// 2. EVALUAMOS EL  MODELO
// ============================================================================

async function evaluar() {
    console.log('📊 Iniciando evaluación...');
    
    // Limpiamos secciones anteriores
    document.getElementById('metricas').style.display = 'none';
    document.getElementById('grafica-matriz').style.display = 'none';
    document.getElementById('grafica-dist').style.display = 'none';
    document.getElementById('grafica-imp').style.display = 'none';
    document.getElementById('grafica-rend').style.display = 'none';
    
    mostrarMensaje('metricas', '⏳ Evaluando modelo y generando gráficas...', 'info');
    document.getElementById('metricas').style.display = 'block';
    
    try {
        const response = await fetch(`${API_BASE}/evaluar`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Error en evaluación');
        }
        
        console.log('✅ Datos de evaluación recibidos');
        
        // Mostrar métricas
        if (data.metricas) {
            mostrarMetricas(data.metricas);
        }
        
        // Mostramos gráficas
        if (data.matriz_confusion) {
            console.log('📊 Mostrando matriz de confusión...');
            mostrarImagen('grafica-matriz', data.matriz_confusion, 'Matriz de Confusión');
        }

        if (data.grafica_distribucion) {
            console.log('📈 Mostrando gráfica de distribución...');
            mostrarImagen('grafica-dist', data.grafica_distribucion, 'Distribución de Variables');
        }

        if (data.grafica_rendimiento) {
            console.log('📊 Mostrando gráfica de rendimiento...');
            mostrarImagen('grafica-rend', data.grafica_rendimiento, 'Rendimiento por Clase');
        }

        if (data.grafica_importancia) {
            console.log('🎯 Mostrando gráfica de importancia...');
            mostrarImagen('grafica-imp', data.grafica_importancia, 'Importancia de Características');
        }
        
        console.log('✅ Evaluación completada');
        
    } catch (error) {
        console.error('❌ Error:', error);
        mostrarMensaje('metricas', `❌ ${error.message}`, 'error');
        document.getElementById('metricas').style.display = 'block';
    }
}

// Mostramos métricas en tabla
function mostrarMetricas(metricas) {
    let html = '<h3>📊 Métricas de Evaluación</h3>';
    html += '<table style="border-collapse: collapse; width: 100%;">';
    html += '<tr style="background: #667eea; color: white;">';
    html += '<th style="border: 1px solid #ddd; padding: 10px; text-align: left;">Métrica</th>';
    html += '<th style="border: 1px solid #ddd; padding: 10px; text-align: right;">Valor</th>';
    html += '</tr>';
    
    const metricas_a_mostrar = ['accuracy', 'precision', 'recall', 'f1'];
    
    for (let metrica of metricas_a_mostrar) {
        if (metricas[metrica] !== undefined) {
            const valor = (metricas[metrica] * 100).toFixed(2);
            html += '<tr style="background: #f9f9f9;">';
            html += `<td style="border: 1px solid #ddd; padding: 10px;"><strong>${metrica.toUpperCase()}</strong></td>`;
            html += `<td style="border: 1px solid #ddd; padding: 10px; text-align: right;"><strong>${valor}%</strong></td>`;
            html += '</tr>';
        }
    }
    
    html += '</table>';
    document.getElementById('metricas').innerHTML = html;
    document.getElementById('metricas').style.display = 'block';
}

// Función para mostrar imágenes base64
function mostrarImagen(elemento_id, imagen_base64, titulo = '') {
    const elemento = document.getElementById(elemento_id);
    if (elemento) {
        console.log(`🖼️ ${titulo}: Longitud de base64: ${imagen_base64 ? imagen_base64.length : 0}`);
        
        if (imagen_base64 && imagen_base64.startsWith('data:image/')) {
            elemento.innerHTML = `
                <h4>${titulo}</h4>
                <img src="${imagen_base64}" style="max-width: 100%; height: auto; border: 1px solid #ddd; padding: 10px; border-radius: 4px;">
            `;
        } else if (imagen_base64 && imagen_base64.length > 100) {
            // Si no tiene el prefijo, agregarlo
            elemento.innerHTML = `
                <h4>${titulo}</h4>
                <img src="data:image/png;base64,${imagen_base64}" style="max-width: 100%; height: auto; border: 1px solid #ddd; padding: 10px; border-radius: 4px;">
            `;
        } else {
            elemento.innerHTML = `
                <h4>${titulo}</h4>
                <p style="color: red;">No se pudo cargar la gráfica.</p>
            `;
        }
        elemento.style.display = 'block';
    }
}

// Luego, en la función evaluar, cambia las llamadas:
async function evaluar() {
    console.log('📊 Iniciando evaluación...');
    
    // Limpiar secciones anteriores
    document.getElementById('metricas').style.display = 'none';
    document.getElementById('grafica-matriz').style.display = 'none';
    document.getElementById('grafica-dist').style.display = 'none';
    document.getElementById('grafica-rend').style.display = 'none';
    
    mostrarMensaje('metricas', '⏳ Evaluando modelo y generando gráficas...', 'info');
    document.getElementById('metricas').style.display = 'block';
    
    try {
        const response = await fetch(`${API_BASE}/evaluar`, {
            method: 'GET',
            headers: {'Content-Type': 'application/json'}
        });
        
        const data = await response.json();
        
        if (!response.ok) throw new Error(data.error || 'Error en evaluación');
        
        console.log('✅ Datos de evaluación recibidos');
        
        // Mostrar métricas
        if (data.metricas) mostrarMetricas(data.metricas);
        
        // Mostrar gráficas
        if (data.matriz_confusion) {
            console.log('📊 Mostrando matriz de confusión...');
            mostrarImagen('grafica-matriz', data.matriz_confusion, 'Matriz de Confusión');
        }
        
        if (data.grafica_distribucion) {
            console.log('📈 Mostrando gráfica de distribución...');
            mostrarImagen('grafica-dist', data.grafica_distribucion, 'Distribución de Variables');
        }
        
        if (data.grafica_rendimiento) {
            console.log('📊 Mostrando gráfica de rendimiento...');
            mostrarImagen('grafica-rend', data.grafica_rendimiento, 'Rendimiento por Clase');
        }
        
        console.log('✅ Evaluación completada');
        
    } catch (error) {
        console.error('❌ Error:', error);
        mostrarMensaje('metricas', `❌ ${error.message}`, 'error');
        document.getElementById('metricas').style.display = 'block';
    }
}

// ============================================================================
// 3. FORMULARIO DE PREDICCIÓN
// ============================================================================

async function generarFormulario() {
    try {
        const response = await fetch(`${API_BASE}/info_dataset`);
        const info = await response.json();
        
        const formulario = document.getElementById('formulario-prediccion');
        let html = '<h3>Ingresa los datos del vehículo:</h3>';
        html += '<form id="form-prediccion" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">';
        
        // Campos numéricos
        if (info.columnas_numericas && Array.isArray(info.columnas_numericas)) {
            for (let col of info.columnas_numericas) {
                const valor = window.ejemploFila?.[col] || 0;
                html += `<div>`;
                html += `<label for="${col}" style="display: block; font-weight: 600; margin-bottom: 5px;">${col}:</label>`;
                html += `<input type="number" step="0.01" name="${col}" id="${col}" value="${valor}" required style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">`;
                html += `</div>`;
            }
        }
        
        // Campos categóricos
        if (info.columnas_categoricas && Array.isArray(info.columnas_categoricas)) {
            for (let col of info.columnas_categoricas) {
                const valor = window.ejemploFila?.[col] || '';
                html += `<div>`;
                html += `<label for="${col}" style="display: block; font-weight: 600; margin-bottom: 5px;">${col}:</label>`;
                html += `<input type="text" name="${col}" id="${col}" value="${valor}" placeholder="Ingrese valor" required style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">`;
                html += `</div>`;
            }
        }
        
        html += `<button type="button" onclick="predecir()" style="grid-column: 1 / -1; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-weight: 600;">🔮 Realizar Predicción</button>`;
        html += '</form>';
        
        formulario.innerHTML = html;
        
    } catch (error) {
        console.error('Error al generar formulario:', error);
        document.getElementById('formulario-prediccion').innerHTML = 
            '<p style="color: red;">Error al cargar el formulario</p>';
    }
}

// Realizamos predicción
async function predecir() {
    const form = document.getElementById('form-prediccion');
    const formData = new FormData(form);
    const datos = Object.fromEntries(formData);
    
    console.log('🔍 Enviando predicción:', datos);
    
    mostrarMensaje('resultado-prediccion', '⏳ Realizando predicción...', 'info');
    document.getElementById('resultado-prediccion').style.display = 'block';
    
    try {
        const response = await fetch(`${API_BASE}/predecir_manual`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(datos)
        });
        
        const resultado = await response.json();
        
        if (!response.ok) {
            throw new Error(resultado.error || 'Error en predicción');
        }
        
        let html = '<h3>🎯 Resultado de la Predicción</h3>';
        html += `<p style="font-size: 1.2em;"><strong>Predicción:</strong> <span style="color: #667eea; font-weight: bold;">${resultado.prediccion}</span></p>`;
        html += `<p><strong>Confianza:</strong> <span style="color: #28a745; font-weight: bold;">${(resultado.confianza * 100).toFixed(2)}%</span></p>`;
        
        if (resultado.probabilidades) {
            html += '<h4>Probabilidades por Clase:</h4>';
            html += '<table style="border-collapse: collapse; width: 100%;">';
            html += '<tr style="background: #28a745; color: white;">';
            html += '<th style="border: 1px solid #ddd; padding: 10px; text-align: left;">Clase</th>';
            html += '<th style="border: 1px solid #ddd; padding: 10px; text-align: right;">Probabilidad</th>';
            html += '</tr>';
            
            for (let [clase, prob] of Object.entries(resultado.probabilidades)) {
                const porcentaje = (prob * 100).toFixed(2);
                const color = prob === Math.max(...Object.values(resultado.probabilidades)) ? '#e8f5e9' : '#f9f9f9';
                html += `<tr style="background: ${color};">`;
                html += `<td style="border: 1px solid #ddd; padding: 10px;"><strong>${clase}</strong></td>`;
                html += `<td style="border: 1px solid #ddd; padding: 10px; text-align: right;"><strong>${porcentaje}%</strong></td>`;
                html += '</tr>';
            }
            
            html += '</table>';
        }
        
        document.getElementById('resultado-prediccion').innerHTML = html;
        console.log('✅ Predicción completada');
        
    } catch (error) {
        console.error('❌ Error:', error);
        mostrarMensaje('resultado-prediccion', `❌ ${error.message}`, 'error');
        document.getElementById('resultado-prediccion').style.display = 'block';
    }
}

// ============================================================================
// INICIALIZAR AL CARGAR LA PÁGINA
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('✅ Página cargada. Inicializando...');
    
    // Generamos formulario de predicción
    generarFormulario();
    
    console.log('✅ Inicialización completada');
});
