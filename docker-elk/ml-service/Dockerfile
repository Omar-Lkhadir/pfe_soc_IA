FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY commun ./commun
COPY hist_gradient_boosting ./hist_gradient_boosting
COPY random_forest ./random_forest

# Sans ca, chaque worker uvicorn charge son propre pool de threads BLAS
# (= nb de coeurs de l'hote par defaut) pour l'inference sklearn -- avec
# plusieurs workers, ca sur-souscrit largement les coeurs disponibles et
# recree la meme contention CPU que le fix "--workers" est cense resoudre.
# Le parallelisme est deja assure par les workers uvicorn + les workers
# Logstash cote appelant, donc 1 thread BLAS par worker suffit.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

EXPOSE 8000

CMD ["uvicorn", "commun.service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
