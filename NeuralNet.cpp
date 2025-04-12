#include <iostream>
#include <cmath>
#include <vector>
#include <string>

using namespace std;

vector<double> MatMult(const vector<vector<double>>& a, const vector<double>& b) {
    vector<double> product(a.size(), 0.0);
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            product[i] += a[i][j] * b[j];
        }
    }
    return product;
}

vector<double> VecAddition(const vector<double>& a, const vector<double>& b) {
    vector<double> result;
    for (size_t i = 0; i < a.size(); ++i) {
        result.push_back(a[i] + b[i]);
    }
    return result;
}

vector<double> VecSubtraction(const vector<double>& a, const vector<double>& b) {
    vector<double> result;
    for (size_t i = 0; i < a.size(); ++i) {
        result.push_back(a[i] - b[i]);
    }
    return result;
}

vector<vector<double>> Transpose(const vector<vector<double>>& mat) {
    vector<vector<double>> trans(mat[0].size(), vector<double>(mat.size()));
    for (size_t i = 0; i < mat.size(); ++i)
        for (size_t j = 0; j < mat[0].size(); ++j)
            trans[j][i] = mat[i][j];
    return trans;
}

vector<vector<double>> OuterProduct(const vector<double>& a, const vector<double>& b) {
    vector<vector<double>> result(a.size(), vector<double>(b.size()));
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
            result[i][j] = a[i] * b[j];
    return result;
}

vector<double> Hadamard(const vector<double>& a, const vector<double>& b) {
    vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        result[i] = a[i] * b[i];
    return result;
}

vector<vector<double>> ScalarMultMatrix(const vector<vector<double>>& mat, double scalar) {
    vector<vector<double>> result = mat;
    for (auto& row : result)
        for (auto& val : row)
            val *= scalar;
    return result;
}

vector<double> ScalarMultVector(const vector<double>& vec, double scalar) {
    vector<double> result = vec;
    for (auto& val : result)
        val *= scalar;
    return result;
}

vector<vector<double>> SubtractMatrix(const vector<vector<double>>& a, const vector<vector<double>>& b) {
    vector<vector<double>> result = a;
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < a[0].size(); ++j)
            result[i][j] -= b[i][j];
    return result;
}

vector<double> SubtractVector(const vector<double>& a, const vector<double>& b) {
    vector<double> result = a;
    for (size_t i = 0; i < a.size(); ++i)
        result[i] -= b[i];
    return result;
}

class NeuralNet {
public:
    vector<vector<double>> w1, w2;
    vector<double> b1, b2;
    vector<double> z1, a1, z2, output;
    double lr;

    NeuralNet(vector<vector<double>> weight1, vector<vector<double>> weight2,
              vector<double> bias1, vector<double> bias2, double learningRate)
        : w1(weight1), w2(weight2), b1(bias1), b2(bias2), lr(learningRate) {}

    vector<double> activation(const vector<double>& a) {
        vector<double> result;
        for (double val : a)
            result.push_back(max(0.0, val));
        return result;
    }

    vector<double> activation_deriv(const vector<double>& a) {
        vector<double> result;
        for (double val : a)
            result.push_back(val > 0.0 ? 1.0 : 0.0);
        return result;
    }

    vector<double> softmax(const vector<double>& input) {
        double sum = 0.0;
        for (double val : input)
            sum += exp(val);
        vector<double> result;
        for (double val : input)
            result.push_back(exp(val) / sum);
        return result;
    }

    vector<double> forward_prop(const vector<double>& X) {
        z1 = VecAddition(MatMult(w1, X), b1);
        a1 = activation(z1);
        z2 = VecAddition(MatMult(w2, a1), b2);
        output = softmax(z2);
        return output;
    }

    double compute_loss(const vector<double>& y_true, const vector<double>& y_pred) {
        double sum = 0.0;
        for (size_t i = 0; i < y_true.size(); ++i)
            sum += pow(y_true[i] - y_pred[i], 2);
        return sum / y_true.size();
    }

    void backward(const vector<double>& X, const vector<double>& y_true) {
        vector<double> dZ2(output.size());
        for (size_t i = 0; i < output.size(); ++i)
            dZ2[i] = 2.0 * (output[i] - y_true[i]);

        vector<vector<double>> dW2 = OuterProduct(a1, dZ2);
        vector<double> db2 = dZ2;

        vector<double> dA1 = MatMult(Transpose(w2), dZ2);
        vector<double> dZ1 = Hadamard(dA1, activation_deriv(z1));

        vector<vector<double>> dW1 = OuterProduct(X, dZ1);
        vector<double> db1 = dZ1;

        w1 = SubtractMatrix(w1, ScalarMultMatrix(dW1, lr));
        b1 = SubtractVector(b1, ScalarMultVector(db1, lr));
        w2 = SubtractMatrix(w2, ScalarMultMatrix(dW2, lr));
        b2 = SubtractVector(b2, ScalarMultVector(db2, lr));
    }
};

int main() {
    
    return 0;
}
