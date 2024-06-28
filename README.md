# AI-Notes

- Deep Learning = Neural Networks

# Linear Regression

### Squared Cost Function
\( f_{\vec{w}, b} \)

$$
J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \vec{w} \cdot \vec{x}^{(i)} + b - y^{(i)} \right)^2
$$

# Logistic Regression

- Sigmoid Function:

$$
g(z) = \frac{1}{1+ e^{-z}} = g( \vec{w} . \vec{x} +b) = \frac{1}{1+ e^{-( \vec{w} . \vec{x} +b)}} \quad || \quad 0 < g(z) < 1
$$

