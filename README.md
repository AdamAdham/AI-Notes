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

- **Dense**: Type of layer

``` python
x = np.array([[0.0,... 245,...240...0]])
layer_1 = Dense (units=25, activation= 'sigmoid') 
a1 = layer_1(x)
# a1 --> tf.Tensor([[0.2 0.7 0.3]], shape=(1, 3), dtype=float32)
a1.numpy()
# a1 --> array([[0.2, 0.7, 0.3]], dtype=float32)
```

- **Sequential**: A framework that strings together layers to create a neural network
*predict()*: Carries out forward propagation to output the expected values
```python
layer 1 Dense (units=3, activation="sigmoid")
layer_2 = Dense (units=1, activation="sigmoid")
model = Sequential([layer_1, layer_2])
x = np.array([[200.0, 17.0],
[120.0, 5.0], 4 x 2
[425.0, 20.0],
[212.0, 18.0]])
y = np.array([1,0,0,1])
model.compile(...) more about this next week
model.fit(x,y)
model.predict(x_new) <
```

### Init Layers Basics

- **Dense**: Type of layer

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
