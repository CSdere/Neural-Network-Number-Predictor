#include <iostream>
#include <cmath>
using namespace std;

double norm_cdf(double x) {
    return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}

int main() {
    double St;    // Spot price
    double K;     // Strike price
    double r;     // Risk-free rate
    double t;     // Time to maturity
    double sigma; // Volatility

    double C;     // Call option price
    double d1;
    double d2;

    cout << "Enter the spot price of the asset: ";
    cin >> St;

    cout << "Enter the strike price: ";
    cin >> K;

    cout << "Enter the risk-free rate : ";
    cin >> r;

    cout << "Enter the time to maturity in years : ";
    cin >> t;

    cout << "Enter the volatility of the asset : ";
    cin >> sigma;

    d1 = (log(St / K) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt(t));
    d2 = d1 - sigma * sqrt(t);

    C = norm_cdf(d1) * St - norm_cdf(d2) * K * exp(-r * t);

    cout << "The call price of the asset is: " << C << endl;

    return 0;
}
