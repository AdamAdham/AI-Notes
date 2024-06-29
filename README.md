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
How well you are doing on **one** training example

$$
L(f_{\vec{w}, b}(\vec{x}^{(i)}),y^{(i)}) = f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)}
$$


### Squared Cost Function
How well you are doing on **all** the training examples

$$
J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left(f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)} \right)^2 
$$

$$
J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left(L(f_{\vec{w}, b}(\vec{x}^{(i)}),y^{(i)}) \right)^2 
$$

### Gradient Descent
### Gradient Descent Algorithm

1. **Initialize Parameters**: Choose initial values for \( \vec{w} \) and \( b \).
   
2. **Define the Learning Rate**: Select a learning rate \( \alpha \).

3. **Iterate Until Convergence**:
   - Compute gradients:
     \[
     \frac{\partial J(\vec{w}, b)}{\partial \vec{w}}, \quad \frac{\partial J(\vec{w}, b)}{\partial b}
     \]
   - Update parameters:
     \[
     \vec{w} := \vec{w} - \alpha \frac{\partial J(\vec{w}, b)}{\partial \vec{w}}, \quad b := b - \alpha \frac{\partial J(\vec{w}, b)}{\partial b}
     \]
   - Repeat until convergence criteria are met.



# Logistic Regression

### Sigmoid Function:

$$
g(z) = \frac{1}{1+ e^{-z}} = g( \vec{w} . \vec{x} +b) = \frac{1}{1+ e^{-( \vec{w} . \vec{x} +b)}} \quad || \quad 0 < g(z) < 1
$$

### Loss Function

$$
L(h_{\vec{w}, b}(\vec{x}^{(i)}), 1) = -\log(h_{\vec{w}, b}(\vec{x}^{(i)})) \quad\quad || \quad\quad \ y^{(i)} = 1
$$

$$
L(h_{\vec{w}, b}(\vec{x}^{(i)}), 0) = -\log(1 - h_{\vec{w}, b}(\vec{x}^{(i)}))  \quad\quad || \quad\quad \ y^{(i)} = 0
$$

$$
L(h_{\vec{w}, b}(\vec{x}^{(i)}), y^{(i)}) = -y^{(i)} \log(h_{\vec{w}, b}(\vec{x}^{(i)})) - (1 - y^{(i)}) \log(1 - h_{\vec{w}, b}(\vec{x}^{(i)}))
$$

Derivation is from principle called maximum likelyhood estimation
- **Why is the logistic regression loss function different from the linear regression loss function?** <br>
Because when we use previous cost function, there will exist many local minima which the gradient descent can be stuck at it.

<img src="./Comparison.png" alt="image" width="500" height="auto">
<img src="./y-equal-1.png" alt="image" width="500" height="auto">
<img src="./y-equal-0.png" alt="image" width="500" height="auto">
credit to Andrew Ng course of Stanford university.
