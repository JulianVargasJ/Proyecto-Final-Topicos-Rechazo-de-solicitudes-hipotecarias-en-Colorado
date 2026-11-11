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
### 2. Abre una terminal desde esa carpeta

### 3. Crear y activar un entorno virtual
```bash
# En Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# En Windows:
python -m venv venv
.\venv\Scripts\activate

# Para solucionar este bloqueo de seguridad predeterminado en Windows PowerShell y activar tu entorno virtual, debes habilitar # la ejecución de scripts:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Ahora vuelve a ejecutar este codigo
.\venv\Scripts\activate

# Deberia aparecer (venv) en verde (el enviroment fue creado correctamente)
```

### 4. Instalar dependencias exactas
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Descargar los datos
Descarga `solicitudes_hmda.csv` e inserar en la carpeta `data/`.

### 6. Ejecutar el pipeline completo
```bash
python main.py
```
Tablas se mostrarán de análisis estadístico en el CMD y los gráficos de resultados se guardarán en `outputs/`.*
