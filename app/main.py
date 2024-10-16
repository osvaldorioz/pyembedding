from fastapi import FastAPI
import transformer_module
import time
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Definir el modelo para la matriz
class Matrix(BaseModel):
    matrix: List[List[float]]

class Vector(BaseModel):
    vector: List[float]

@app.post("/embedding")
async def tt(embedding: Vector, weights: Matrix):
    start = time.time()
    
    wwdata = weights.matrix
    embdata = embedding.vector

    # Procesar el embedding utilizando el transformer
    output = transformer_module.transformer(embdata, wwdata)
    
    end = time.time()

    var1 = 'Time taken in seconds: '
    var2 = end - start

    str = f'{var1}{var2}\n'.format(var1=var1, var2=var2)
    str1= f"Salida del Transformer Embedding: {output} "
    
    return str + str1