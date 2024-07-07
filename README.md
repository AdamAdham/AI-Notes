# AI-Notes

- Deep Learning = Neural Networks = Multilayer perceptron

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

$$
\vec{w}_j := \vec{w}_j - \alpha \frac{\partial J(\vec{w}, b)}{\partial \vec{w}}, \quad b := b - \alpha \frac{\partial J(\vec{w}, b)}{\partial b} \quad || \quad \alpha= \text {learning rate} \quad\quad j=1..n \text{ where n is number of features}
$$

   - Repeat until convergence criteria are met. **Simultaneously**

#### Derivitive

$$
\frac{\partial J(\vec{w}, b)}{\partial \vec{w}} = \frac{1}{m} \sum_{i=1}^{m} \left(f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)} \right)x^{(i)}
$$

$$
\frac{\partial J(\vec{w}, b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left(f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)} \right)
$$

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

## Regularization
Done to minimize **w** parameters to reduce overfitting by adding an extra term to the cost function

### Regularization Term

$$
\frac{\partial J(\vec{w}, b)}{\partial \vec{w}} = \frac{1}{m} \sum_{i=1}^{m} \left(f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)} \right)x^{(i)}
$$

### Final Cost Function

$$
J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left(f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)} \right)^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2 \quad|| \quad  \lambda = \text{Regularization parameter}
$$

### Derivative

Because according to this Wj all other Ws are constant so are zeroed

$$
\frac{\partial}{\partial w_j} \left( \sum_{j=1}^{m}\frac{\lambda}{2m} w_j^2 \right) = \lambda w_j   \quad || \quad m=\text{number of features}
$$

$$
\frac{\partial J(\vec{w}, b)}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \left(f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)} \right)x^{(i)} + \frac{\lambda}{m} w_j 
$$

# Deep Learning

- **Layer** : Group of neurons/units
- **Unit** : A processes the input and produces output
- **Activations** : Output from the layer

$$
x^n_m \quad || \quad \text{this is the variable x for neuron m at layer n}
$$

## TensorFlow

### Basics
- Row x Column
```python
x = np.array([[200, 17]])    # 1x2 row vector
x = np.array([[200], [209]]) # 2x1 column vector
x= np.array([200,17])        # 1D vector/array
```
<br>

- **Dense**: Type of layer

Without Vectorization
```python
# Simplified implementation
def dense (a_in, W, b):
units = W.shape[1] # Getting number of cols = number of outputs from layer
a_out = np.zeros(units)
for j in range(units): 0,1,2
   W = W[:, j]
   z = np.dot (w,a_in) + b[j]
   a_out [j] = g(z)
return a_out
```
Vectorization
```python
X = np.array([[200, 17]]) # 1x2 Made into 2D
W = np.array([[1, -3, 5],
               [-2, 4, 6]])
B = np.array([[-1, 1, 2]]) # 1x3 Made into 2D
def dense (A_in, W, B):
   Z = np. matmul (A_in, W) + B # matmul =  matrix multiplication
   A_out = g(Z)
   return A_out
```

``` python
x = np.array([[0.0,... 245,...240...0]])
layer_1 = Dense (units=25, activation= 'sigmoid') 
a1 = layer_1(x)
# a1 --> tf.Tensor([[0.2 0.7 0.3]], shape=(1, 3), dtype=float32)
a1.numpy()
# a1 --> array([[0.2, 0.7, 0.3]], dtype=float32)
```
<br>

- **Sequential**: A framework that strings together layers to create a neural network <br>
*predict()*: Carries out forward propagation to output the expected values

```python
def sequential(x):
   a1 = dense (x, W1, b1)
   a2 = dense (a1, W2, b2)
   a3 = dense (a2, W3, b3)
   a4 = dense (a3, W4, b4)
   f_x = a4
   return f_x
```

```python
model Sequential([
Dense (units=3, activation="sigmoid"),
Dense (units=1, activation="sigmoid")])
x = np.array([[200.0, 17.0],
[120.0, 5.0], 4 x 2
[425.0, 20.0],
[212.0, 18.0]])
y = np.array([1,0,0,1])
model.compile(...) more about this next week
model.fit(x,y)
model.predict(x_new) <
```
<br>

### Init Layers Basics

``` python
x = np.array([[0.0,... 245,...240...0]])
layer_1 = Dense (units=25, activation= 'sigmoid') 
a1 = layer_1(x)
# a1 --> tf.Tensor([[0.2 0.7 0.3]], shape=(1, 3), dtype=float32)
a1.numpy()
# a1 --> array([[0.2, 0.7, 0.3]], dtype=float32)

layer_2 = Dense (units=15, activation= 'sigmoid')
a2 = layer_2(a1)
layer 3 = Dense (units=1, activation=ʻsigmoid')

a3 = layer_3(a2)
if a3 >= 0.5:
yhat = 1
else:
yhat = 0
```

### Convention
- Using 2D arrays not 1D even if row/column vector (due to tensorflow being developed for extremely large data sets)
- Lowercase vars = Scalar/Vectors
- Uppercase vars = Matrix

## Linear Algebra

### Dot Product
The dot product (also known as the scalar product) of two vectors is an algebraic operation that takes two equal-length sequences of numbers (usually coordinate vectors) and returns a single number. The dot product of vectors \(\mathbf{a}\) and \(\mathbf{b}\) is defined as:

$$ \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} (a_i \cdot b_i) $$

Where $\(\mathbf{a}\)$ and $\(\mathbf{b}\)$ are vectors of length \(n\).

Example:
* $\(\mathbf{a} = [a_1, a_2, a_3]\)$
* $\(\mathbf{b} = [b_1, b_2, b_3]\)$

The dot product $\(\mathbf{a} \cdot \mathbf{b}\)$ is given by:

$$ a_1 \cdot b_1 + a_2 \cdot b_2 + a_3 \cdot b_3 $$

### Transpose
The transpose of a matrix is an operation that flips a matrix over its diagonal, switching the row and column indices of the matrix. The transpose of matrix $\(\mathbf{A}\)$ is denoted as $\(\mathbf{A}^T\)$.

If $\(\mathbf{A}\)$ is an $\(m \times n\)$ matrix, then the transpose $\(\mathbf{A}^T\)$ is an $\(n \times m\)$ matrix.

Example:

$$
\mathbf{A} = 
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}\
$$

$$
\mathbf{A}^T = 
\begin{bmatrix}
1 & 4 \\
2 & 5 \\
3 & 6
\end{bmatrix}\
$$

### Note

The dot product (also known as the scalar product) of two vectors can be expressed as the transpose of the first vector multiplied by the second vector. For vectors \(\mathbf{a}\) and \(\mathbf{b}\), it is defined as:

$$ \mathbf{a} \cdot \mathbf{b} = \mathbf{a}^T \mathbf{b} $$

Where:
- $\(\mathbf{a}^T\)$ denotes the transpose of vector $\(\mathbf{a}\)$.

#### Example:
If $\(\mathbf{a} = [a_1, a_2, a_3]\)$ and $\(\mathbf{b} = [b_1, b_2, b_3]\)$, <br>
Then $\(\mathbf{a} \cdot \mathbf{b} = \begin{bmatrix} a_1 & a_2 & a_3 \end{bmatrix} \begin{bmatrix} b_1 \\ b_2 \\ b_3 \end{bmatrix} = a_1 b_1 + a_2 b_2 + a_3 b_3 \)$
