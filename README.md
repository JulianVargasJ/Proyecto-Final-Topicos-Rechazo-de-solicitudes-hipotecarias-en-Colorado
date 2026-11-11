# Proyecto Final Topicos | Rechazo de Solicitudes Hipotecarias en Colorado
Trabajo Final de Tópicos Aplicados en Estadística
Integrantes: Lucas González, Álvaro Morales, Celeste Romero y Julián Vargas




## Reproducibilidad del Proyecto
Seguir los siguientes pasos para reproducir el entrenamiento y las métricas de prueba:

### 1. Clonar el repositorio y acceder a la carpeta
```bash
git clone [https://github.com/tu-usuario/hmda-mortgage-risk.git](https://github.com/tu-usuario/hmda-mortgage-risk.git)
cd Proyecto-Final-Topicos-Rechazo-de-solicitudes-hipotecarias-en-Colorado
```

### 2. Crear y activar un entorno virtual
```bash
# En Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# En Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar dependencias exactas
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Descargar los datos
Descarga `solicitudes_hmda.csv` e inserat en la carpeta `data/`.

### 5. Ejecutar el pipeline completo
```bash
python main.py
```
Gráficos y tablas de resultados se guardarán en `outputs/`.*
