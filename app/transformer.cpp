#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>

//c++ -O3 -Ofast -Wall -shared -std=c++20 -fPIC $(python3.12 -m pybind11 --includes) transformer.cpp -o transformer_module$(python3.12-config --extension-suffix)

namespace py = pybind11;

// Simulación de la multiplicación por una matriz de pesos
std::vector<double> linear_transform(const std::vector<double>& embedding, const std::vector<std::vector<double>>& weights) {
    std::vector<double> transformed(embedding.size(), 0.0);

    for (size_t i = 0; i < embedding.size(); ++i) {
        for (size_t j = 0; j < embedding.size(); ++j) {
            transformed[i] += embedding[j] * weights[i][j];
        }
    }
    return transformed;
}

// Simulación de la función de activación (ReLU)
std::vector<double> relu(const std::vector<double>& input) {
    std::vector<double> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::max(0.0, input[i]);
    }
    return output;
}

// Simulación de atención
std::vector<double> attention(const std::vector<double>& query, const std::vector<double>& key, const std::vector<double>& value) {
    double dot_product = 0.0;
    for (size_t i = 0; i < query.size(); ++i) {
        dot_product += query[i] * key[i];
    }
    
    // Normalización simple de la atención (softmax básico)
    double attention_score = std::exp(dot_product) / (std::exp(dot_product) + 1.0);
    
    std::vector<double> attended_value(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        attended_value[i] = attention_score * value[i];
    }
    return attended_value;
}

// Función principal del transformer
std::vector<double> transformer(const std::vector<double>& embedding, const std::vector<std::vector<double>>& weights) {
    // Paso 1: Transformación lineal (capa densa)
    std::vector<double> transformed = linear_transform(embedding, weights);
    
    // Paso 2: Activación ReLU
    std::vector<double> activated = relu(transformed);
    
    // Paso 3: Simulación de un mecanismo de atención
    return attention(activated, embedding, transformed);
}

// Enlace con Pybind11
PYBIND11_MODULE(transformer_module, m) {
    m.def("transformer", &transformer, "Un transformer para procesar embeddings",
          py::arg("embedding"), py::arg("weights"));
}
