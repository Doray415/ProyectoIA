from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
# Cargar el modelo
model = joblib.load('Analisis_Riesgo_Educacion_UdeA.pkl')
# Inicializar la aplicación Flas
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del cuerpo de la petición
    sexo = float(request.form['sexo'])
    cod_prog = float(request.form['cod_prog'])
    dpto_nace = float(request.form['dpto_nace'])
    mpio_nace = float(request.form['mpio_nace'])
    estrato = float(request.form['estrato'])
    mod_ingreso = float(request.form['mod_ingreso'])
    t_colegio  = float(request.form['t_colegio'])
    ult_prog = float(request.form['ult_prog'])
    p_ingreso = float(request.form['p_ingreso'])
    edad_categoria = float(request.form['edad_categoria'])

    pred_prob = model.predict_proba([[sexo, 
                                      cod_prog, 
                                      dpto_nace,
                                      mpio_nace,
                                      estrato, 
                                      mod_ingreso,
                                      t_colegio,
                                      ult_prog,
                                      p_ingreso,
                                      edad_categoria]])
    # Esto deberia desde modelo
    class_names = model.classes_ 
    # Esto hay que cambiarlo
    class_names = ("Probabilidad con riesgo", "Probabilidad sin riesgo")
    mensaje = ""

    for i, class_name in enumerate(class_names):
        prob = pred_prob[0, i] * 100
        mensaje += f"Probabilidad de {class_name}: {round(prob, 2)}% <br/>"
    app.logger.info("fin")
    return render_template('result.html', pred=mensaje)

if __name__ == '__main__':
    app.run()