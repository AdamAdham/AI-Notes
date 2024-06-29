# AI-Notes

- Deep Learning = Neural Networks

# Definitions
$$
\vec{x}^{(i)} = \text{inputs for result set } i
$$

$$
y^{(i)} = \text{value of result set } i
$$

# Linear Regression

### Estimate Function
$$
f_{\vec{w}, b}(\vec{x}^{(i)} = \vec{w} \cdot \vec{x}^{(i)} + b 
$$

### Loss Function

$$
L(f_{\vec{w}, b}(\vec{x}^{(i)}),y^{(i)}) = f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)}
$$


### Squared Cost Function

$$
J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left(f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)} \right)^2 
$$

$$
J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left(L(f_{\vec{w}, b}(\vec{x}^{(i)}),y^{(i)}) \right)^2 
$$



# Logistic Regression

### Sigmoid Function:

$$
g(z) = \frac{1}{1+ e^{-z}} = g( \vec{w} . \vec{x} +b) = \frac{1}{1+ e^{-( \vec{w} . \vec{x} +b)}} \quad || \quad 0 < g(z) < 1
$$

### Loss Function

- **Why is the logistic regression loss function different from the linear regression loss function?**
  Because when we use the logistic regression loss function, such as the cross-entropy loss, it helps to avoid issues like getting stuck in local minima during optimization. The gradient descent process in logistic regression is designed to minimize this specific loss function, which is better suited for classification tasks where we deal with probabilities and discrete outcomes.


![Screenshot](./Comparison.png){: width="400px"}

$$
L(h_{\vec{w}, b}(\vec{x}^{(i)}), 1) = -\log(h_{\vec{w}, b}(\vec{x}^{(i)})) \quad\quad || \quad\quad \ y^{(i)} = 1
$$

$$
L(h_{\vec{w}, b}(\vec{x}^{(i)}), 0) = -\log(1 - h_{\vec{w}, b}(\vec{x}^{(i)}))  \quad\quad || \quad\quad \ y^{(i)} = 0
$$

$$
L(h_{\vec{w}, b}(\vec{x}^{(i)}), y^{(i)}) = -y^{(i)} \log(h_{\vec{w}, b}(\vec{x}^{(i)})) - (1 - y^{(i)}) \log(1 - h_{\vec{w}, b}(\vec{x}^{(i)}))
$$


