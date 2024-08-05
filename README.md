# AI-Notes

# Definitions
- Deep Learning = Neural Networks = Multilayer perceptron
- Handwritten digit classification problem = Binary classification
- Information gain: Reduction of entropy
- **Variables**

$$
\vec{x}^{(i)} = \text{inputs for result set } i
$$

$$
y^{(i)} = \text{value of result set } i
$$

- **Cross Entropy function**

$$
L(f(x), y) = -y \log(f(x)) - (1 - y) \log(1 - f(x))
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

# Softmax Regression

- Classification but not binary (multiclass classification)
- If at output layer there are 10 units, then there are 10 classes

## Softmax Regression Function

Given a set of $n$ features $\(\vec{x} = [x_1, x_2, \ldots, x_n]\)$ and $k$ classes, the softmax function for the $j$-th class $(\( j = 1, 2, \ldots, k \))$ is defined as:

<img src="softmax function.png" alt="softmax function" width="500" height="auto">


where:
- $\vec{x}$ is the input feature vector.
- $\vec{w}_j$ is the weight vector for the $j$-th class.
- $b_j$ is the bias term for the $j$-th class.
- $k$ is the number of classes.

## Softmax Loss Function

<img src="softmax loss.png" alt="softmax function" width="500" height="auto">

## Feature Scaling

Feature scaling is a technique used to standardize the range of independent variables or features in a dataset. It is an important preprocessing step in many machine learning algorithms, especially those that calculate distances between data points, such as k-nearest neighbors (KNN) and support vector machines (SVM), or those that rely on gradient descent optimization, like neural networks.

### Why Feature Scaling is Important

1. **Different Ranges:** Features in a dataset can have different units and scales. For example, one feature might represent age (ranging from 0 to 100), while another represents income (ranging from hundreds to thousands). Without scaling, the feature with the larger range may dominate the model's learning process, leading to biased or suboptimal results.

2. **Improved Convergence:** In algorithms like gradient descent, feature scaling can improve the convergence speed. Scaled features can help the algorithm reach the optimal solution more efficiently.

3. **Distance-Based Algorithms:** Algorithms that compute distances (e.g., KNN, k-means clustering) can be sensitive to the scale of the data. Feature scaling ensures that each feature contributes equally to the distance calculations.

### Common Feature Scaling Techniques

1. **Min-Max Scaling (Normalization):**
   - **Description:** This technique rescales the data to a fixed range, usually [0, 1].
   - **Formula:** $X' = (X - X_min) / (X_max - X_min)$
   - **Use Case:** Useful when you know the bounds of your data.

2. **Standardization (Z-score Normalization):**
   - **Description:** This technique transforms the data to have a mean of 0 and a standard deviation of 1.
   - **Formula:** $X' = (X - μ) / σ$
   - **Use Case:** Useful when the data has outliers or is not bounded, as it retains the influence of outliers.

3. **MaxAbs Scaling:**
   - **Description:** This technique scales each feature by its maximum absolute value, ensuring that the data is within the range [-1, 1].
   - **Formula:** $X' = X / |X_max|$
   - **Use Case:** Useful when the data is sparse or contains negative values.

4. **Robust Scaling:**
   - **Description:** This technique uses the median and the interquartile range (IQR) for scaling.
   - **Formula:** $X' = (X - median) / IQR$
   - **Use Case:** Useful when the data contains outliers.

### How to Choose a Scaling Technique

The choice of scaling technique depends on the nature of your data and the specific requirements of the machine learning algorithm you're using. For example, standardization is often preferred for algorithms that assume normally distributed data, while normalization is useful when the data has a uniform distribution.


## Data Split

#### For Evaluating Performance
**When it is hard to plot the graph of the function**
_We can split the training set into two subsets, a larger part is the training set and another smaller set is the test set_ <br>
**This actually has an overly optimistic estimate of the generalization error on the training data since we choose a model, changing the function's degree of polynomial then check the $J_test$** <br>
_So we instead split the data set into **3** sets 1.Training set 2.Cross validation set(Development set) 3.Test set_
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

# Further split into train and validation sets
x_train, x_valid, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=42)
```
## Building the Model

```python
from tensorflow. keras.losses import BinaryCrossentropy
model.compile (loss=BinaryCrossentropy()) # BinaryCrossentropy() is specified as the loss function of binary classification
model.fit (x_train, y_train, epochs=100) # epochs is the number of steps in gradient descent
```

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
   # np. matmul (A_in, W) the same as A_in @ W
   Z = np. matmul (A_in, W) + B # matmul =  matrix multiplication
   A_out = g(Z) # g() activation function
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
from tensorflow. keras.losses import BinaryCrossentropy
model = Sequential([
Dense (units=3, activation="sigmoid"),
Dense (units=1, activation="sigmoid")])
x = np.array([[200.0, 17.0],
[120.0, 5.0], 4 x 2
[425.0, 20.0],
[212.0, 18.0]])
y = np.array([1,0,0,1])
model.compile (loss=BinaryCrossentropy()) # BinaryCrossentropy() is specified as the loss function
model. fit (X, Y, epochs=100) # epochs is the number of steps in gradient descent
model.predict(x_new)
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

### Layer Types
#### Dense
Each neuron output is a function of **all** the activation outputs of the previous layers.

#### Convolutional
Each neuron only looks at a **part** of the previous layer's ouputs (can be overlapping). <br>

_Why?_
- Faster computation
- Need less training data (less prone to overfitting)


### Convention
- Using 2D arrays not 1D even if row/column vector (due to tensorflow being developed for extremely large data sets)
- Lowercase vars = Scalar/Vectors
- Uppercase vars = Matrix

## Model Building
1. Import dependencies
2. Review features
3. Feature engineering
4. Feature scaling using [this](#feature-scaling)
5. Data split using [this](#data-split)
7. Creation of model
8. Training model
9. Review model
- If satisfactory -> DONE
- If not -> 6 or 7

### Number of layers
### Number of units
**Input Layer**: number of features
**Output Layer**: number of outputs

## Reviewing the model
### Why **loss** not just the accuracy:
#### Handling Class Imbalance
Adjusting for Imbalance: In cases of class imbalance, where one class is much more frequent than the other, the loss function can help mitigate the effects of imbalance. Loss functions like binary cross-entropy can be modified with class weights to give more importance to the minority class, improving model performance on imbalanced datasets.
```python
optimizer = Adam(learning_rate=0.01)  # Adjust learning rate
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
```

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
If $\mathbf{a} = [a_1, a_2, a_3]$ and $\mathbf{b} = [b_1, b_2, b_3]\$, <br>
Then 

$$
\mathbf{a}^T \mathbf{b} = \begin{bmatrix} a_1 & a_2 & a_3 \end{bmatrix} 
\begin{bmatrix} b_1 \\
b_2 \\
b_3 
\end{bmatrix} = a_1 b_1 + a_2 b_2 + a_3 b_3 
$$

### Matrix Multiplication

1. **Check Dimensions**: Ensure that the number of columns in $B$ matches the number of rows in $B$.

2. **Setup the Resulting Matrix**: Determine the dimensions of $C$, which will be $m \times p$ if $A$ is $m \times n$ and $B$ is $n \times p$.

3. **Compute Each Element of** $\(\mathbf{C}\)$:

$$
C_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj}\
$$

5. **Repeat for Each Element**: Compute each element $C_{ij}\$ by iterating over all possible values of $k$ from 1 to $n$.

6. **Write Out the Resulting Matrix** $C$: Assemble all computed elements into the resulting matrix $C$.


$$
C = \begin{bmatrix}
\vec{A_{1j}}.\vec{B_{i1}} & \vec{A_{1j}}.\vec{B_{i2}} \\
\vec{A_{2j}}.\vec{B_{i1}} & \vec{A_{2j}}.\vec{B_{i2}}
\end{bmatrix}
$$

#### Example:

$$
\mathbf{A} = 
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
\quad \quad
\mathbf{B} = 
\begin{bmatrix}
7 & 8 \\
9 & 10 \\
11 & 12
\end{bmatrix}
$$

$$
\quad \quad \quad [2x3] \quad \quad \quad \quad \quad \quad [3x2]
$$
<br>

$$
\mathbf{A} \times \mathbf{B} = 
\begin{bmatrix}
1 \cdot 7 + 2 \cdot 9 + 3 \cdot 11 & 1 \cdot 8 + 2 \cdot 10 + 3 \cdot 12 \\
4 \cdot 7 + 5 \cdot 9 + 6 \cdot 11 & 4 \cdot 8 + 5 \cdot 10 + 6 \cdot 12
\end{bmatrix}
$$

## Building a model steps

<img src="build model logistic vs neural network.png" alt="logistic regression vs neural network" width="500" height="auto">

### 1. Create a Model
Creating the structure of the model
```python
import tensorflow as tf
from tensorflow. keras import Sequential
from tensorflow. keras.layers import Dense
model = Sequential ([
Dense (units=25, activation='sigmoid'),
Dense (units=15, activation='sigmoid′),
Dense (units=1, activation='sigmoid')])
```

### 2. Loss and Cost Functions

- **Binary Classification**
```python
from tensorflow. keras. losses import BinaryCrossentropy
model.compile(loss= BinaryCrossentropy())
```

- **Linear Regression**
```python
from tensorflow. keras. losses import MeanSquaredError
model.compile (loss= Mean SquaredError())
```

### Minimize Cost (Optimization function)

#### Gradient Descent
[Gradient descent Algorithm](#gradient-descent) <br>
TensorFlow uses **back propagation** to compute the partial derivitives

#### ADAM Algorithm
Adaptive moment estimation
- Has a different learning rate $a$ for every $w$
- Dynamic learning rate $a$
- Increases learning rate $a$ if $w_j$ keeps moving towards the same direction
- Decreases learning rate $a$ if $w_j$ keeps oscillating

```python
# Same model
# learning_rate=1e-3 is the initial learning rate
model. compile (optimizers=tf.keras.optimizaers.Adam(learning_rate=1e-3),
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) )
```
## Activation Functions

### Linear Function

$$
g(z) = z
$$

<img src="linear graph.png" alt="sigmoid function" width="300" height="auto">

### Sigmoid Function

$$
g(z) = \frac{1}{1+ e^{-z}} = g( \vec{w} . \vec{x} +b) = \frac{1}{1+ e^{-( \vec{w} . \vec{x} +b)}} \quad || \quad 0 < g(z) < 1
$$

<img src="sigmoid function.png" alt="sigmoid function" width="300" height="auto">

### Relu Function
**Rectified linear Unit function**

$$
g(z) = max(0,z)
$$

<img src="relu graph.png" alt="relu function" width="300" height="auto">

### Softmax Function
- Multiclass classification
- Only function that leads to output of output layer $a_j$ to be in terms of $z_1$ to $z_n$ where $n$ is the number of classes

[Softmax Function](#softmax-regression) <br>

### Why use activation functions
**Why not use linear for all hidden**
Because if so, the neural network will just be a linear regression model.
And if we just make the output activation a sigmoid function then it will be logistic regression
<br>
<img src="act fn exp.png" alt="relu function" width="300" height="auto">

## Numeric Round Off Error
**Definition:**
Numeric round-off error, also known as floating-point round-off error, occurs when a computer approximates a real number with a finite precision representation. This is inherent in digital computation due to the limitations of storing infinitely precise numbers in a finite amount of memory.

**Why It Happens:**
1. **Finite Precision:** Computers use a fixed number of bits to represent numbers. For example, a common representation is the IEEE 754 standard for floating-point arithmetic, which uses 32 bits (single precision) or 64 bits (double precision).
2. **Binary Representation:** Not all decimal fractions can be represented exactly in binary form. For example, 0.1 in decimal is a repeating fraction in binary, leading to an approximation.
3. **Operations:** Arithmetic operations on floating-point numbers (addition, subtraction, multiplication, division) can introduce additional errors. The result of an operation might need to be rounded to fit into the designated number of bits.

**Consequences:**
1. **Accumulation of Error:** In iterative calculations, such as numerical methods or long-running simulations, small round-off errors can accumulate, leading to significant inaccuracies.
2. **Precision Limits:** Certain mathematical operations can exacerbate round-off errors, especially when dealing with very large or very small numbers.
3. **Comparisons:** Testing for equality between floating-point numbers can be problematic. Small errors can lead to unexpected results when comparing numbers that should theoretically be equal.

**Examples:**
1. **Simple Addition:**
    ```python
    a = 0.1
    b = 0.2
    c = a + b
    print(c)  # Outputs 0.30000000000000004 instead of 0.3
    ```

2. **Subtraction Leading to Loss of Precision:**
    ```python
    x = 1.0000001
    y = 1.0000000
    z = x - y
    print(z)

**Solution:**

### Logistic Regression

```python
model = Sequential ([10,000
Dense (units=25, activation='relu' ) ,
Dense (units=15, activation='relu' )
Dense (units=1, activation='linear' )
])
# We changed the output activation function instead of sigmoid
model. compile (loss=BinaryCrossEntropy (from_logits=True) )
```
<img src="error logistic.png" alt="relu function" width="300" height="auto">
<img src="error softmax.png" alt="relu function" width="300" height="auto">

## Multi-label Classification
**Input data leads to multiple outputs.** <br>
Output y is a **vector**

_Example:_
**Input:** Image
**Output:** Is there a car? How many pedestrians? Is there a bus?

### How
You can have 3 seperate neural networks.
Or you can have 1 neural network with the last layer having 3 units/neurons.

## Back Propagation

<img src="der 1.png" alt="derivitive part 1" width="500" height="auto">
<img src="der 2.png" alt="derivitive part 2" width="500" height="auto">
<br>
Right to left is back prop
<img src="back prop.png" alt="back propagation" width="500" height="auto">

### How to do it
1. Get $vec{w}$ and $b$ by using the training set to get lowest cost for this data set for a specific degree function $d$
2. Choose which degree of function (a specific model) $d$ according to lowest cost function for the development set $J_cv$
3. Estimate generalization error using test set with $J_test$

**Why is this better?** <br>
_Because in the process of choosing a specific model the test set is not used at all, so the generalization error is truly fair_
<img src="cv set.png" alt="back propagation" width="500" height="auto">

This can be done for any cost/loss function
<img src="training set.png" alt="back propagation" width="500" height="auto">
<br> 
Can do like the square cost can do for logistic cost (log) or can do this
<img src="fraction training set.png" alt="back propagation" width="500" height="auto">

# Diagnostics
**A test that you run to gain insight into what is/isn't working with a learning algorithm, to gain guidance into improving its performance**

**High Bias:** Underfit <br>
**High Variance:** Overfit <br>
<img src="bias variance reg parameter.png" alt="bias variance reg parameter" width="500" height="auto">
<img src="bias variance degree.png" alt="bias variance degree" width="500" height="auto">

# Precision & Recall

### Precision, Recall, and F1 Score

**Precision:**

$$
\text{Precision} = \frac{TP}{TP + FP} 
$$

**Recall:**

$$
\text{Recall} = \frac{TP}{TP + FN} 
$$

**F1 Score (Harmonic mean):**
An average that prioritizes the lower value 

$$
\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} 
$$

where:
- $TP$ = True Positives
- $FP$ = False Positives
- $FN$ = False Negatives

# Decision Trees

1. **Decision 1:** How to choose what feature to split on?
- Maximize Purity
3. **Decision 2:** When to stop splitting?
- When a node is 100% one class
- When splitting a node will result in the tree exceeding a maximum depth
- When improvements in purity score are below a threshold
- When number of examples in a node is below a threshold

## Entropy
It is a measure of impurity (can also use genie criteria)

**Definition:**
Entropy is a measure of the uncertainty or impurity in a dataset. In the context of classification, it quantifies the amount of disorder or unpredictability in a collection of classes. A lower entropy value indicates a more homogenous set, while a higher value indicates more diversity among the classes.

**Entropy Function:**
For a discrete random variable $Y$ that can take on $k$ different classes, the entropy $H(Y)$ is defined as:

$$
H(Y) = -\sum_{i=1}^{k} p(y_i) \log_b(p(y_i))
$$

where:
- $p(y_i)$ is the probability of class $y_i$
- $b$ is the base of the logarithm (commonly base 2 for bits, or base $e$ for nats),
- $k$ is the total number of classes.

<br>
High entropy is worse in large amounts of data therefore; <br>
We take a weighted average to have the larger data set have more influence on the entropy
<img src="bias variance degree.png" alt="bias variance degree" width="500" height="auto">

**Why not directly use information gain(reduction in entropy)?** <br>
_Because one of the criterias to stop splitting is if the reduction in entropy is below a certain threshold_
<img src="info gain split.png" alt="bias variance degree" width="500" height="auto">
<br>

### Information gain
<img src="info gain.png" alt="info gain" width="500" height="auto">
Where:

$p_n^{\text{root}}$: Fraction of examples that are positive classes in the root node

$p_n^{\text{left}}$: Fraction of examples that are positive classes in the left branch

$w_n^{\text{left}}$: Fraction of examples that were split into the left branch

# Machine Learning Notes

## Regression Trees

### Key Concepts
- **Regression Trees** are a type of decision tree used for predicting continuous values.
- They partition the data into subsets using a series of binary splits based on feature values.
- The goal is to minimize the variance within each subset.

### How It Works
1. **Splitting the Data**: The data is split into two groups at each node based on the feature that results in the greatest reduction in variance.
2. **Leaf Nodes**: When a stopping criterion is met (e.g., minimum number of samples per leaf or maximum tree depth), the leaf node is created, which contains the predicted value (typically the mean of the target values in that node).

### Advantages and Disadvantages
- **Advantages**:
  - Simple to understand and interpret.
  - Can handle both numerical and categorical data.
- **Disadvantages**:
  - Prone to overfitting.
  - Sensitive to small changes in the data.

## Sampling with Replacement

### Key Concepts
- **Sampling with Replacement**: A sampling method where each sample is chosen randomly from the data, and after selection, it is returned to the dataset, allowing it to be chosen again.
- This method is essential for techniques like **Bootstrap Aggregation (Bagging)**.

### Use in Machine Learning
- Used in ensemble methods to create multiple training sets from the original dataset.
- Helps in reducing variance and improving the stability of models.

## Random Forest Algorithm

### Key Concepts
- **Random Forest** is an ensemble learning method that builds multiple decision trees and merges them to get a more accurate and stable prediction.
- It uses both bagging (sampling with replacement) and feature randomness to create diverse trees.

### How It Works
1. **Bootstrap Samples**: Random subsets of the original data are created using sampling with replacement.
2. **Random Feature Selection**: At each node in a tree, a random subset of features is considered for splitting.
3. **Aggregation**: The predictions of individual trees are aggregated (averaged for regression, majority vote for classification) to produce the final prediction.

### Advantages and Disadvantages
- **Advantages**:
  - Reduces overfitting by averaging multiple models.
  - Handles large datasets and many features well.
  - Provides feature importance estimates.
- **Disadvantages**:
  - Can be computationally expensive.
  - Less interpretable than individual decision trees.
 
## Boosted Trees
An adjusted tree that actually chooses the subset of training examples **not** randomly with **1/m** probability but according to the previous classification of the ensamble tree, the training example with the least success rate of classification has the highest probability to be chosen 
<img src="./boosted-trees.png" alt="image" width="500" height="auto">

### Implementation (xgBoost)
<img src="./xgboost.png" alt="image" width="500" height="auto">

  

## Practical Considerations

- **Hyperparameters to Tune**:
  - Number of trees (`n_estimators`)
  - Maximum depth of the trees (`max_depth`)
  - Number of features to consider at each split (`max_features`)

- **Common Applications**:
  - Regression tasks (predicting a continuous target variable).
  - Classification tasks (predicting a categorical target variable).

## Summary
- Regression Trees and Random Forests are powerful tools in machine learning for both regression and classification tasks.
- Understanding how to use and tune these models can greatly improve model performance and predictive accuracy.

# Decision Trees vs Neural Networks
<img src="./Decision Trees vs Neural Networks.png" alt="Decision Trees vs Neural Networks" height="auto">


### One Hot Encoding
Changing a categorical feature with $k$ categories(possible values), to $k$ binary **features**. Only one of the features will have 1 (positive value).

**Example:** <br>
1 feature:<br>
position: defender, midfielder, forward 
3 features: <br>
isDefender, isMidfielder, isForward

```python
df_encoded = pd.get_dummies(df, columns=[‘Category’])
```
In this example, the get_dummies function is used to one-hot encode the 'Category' column in the DataFrame. The resulting DataFrame (df_encoded) will have new columns for each unique category in the original 'Category' column, with binary values (0 or 1) indicating the presence of each category.

# NLP 
**Natural Language Processing** <br>
[Github repo that trains transformer on any given text](https://github.com/karpathy/nanoGPT) <br>
One file creates the model and the other trains it. <br>
[Youtube video of the author rewriting the repo](https://youtu.be/kCc8FmEb1nY?si=U5k2hjLQXJsFpYJY)

### What Is the Bag of Words Model in NLP?
The bag of words model is a simple way to convert words to numerical representation in natural language processing. This model is a simple document embedding technique based on word frequency. Conceptually, we think of the whole document as a “bag” of words, rather than a sequence. We represent the document simply by the frequency of each word. Using this technique, we can embed a whole set of documents and feed them into a variety of different machine learning algorithms.<br>
For example, if we have a vocabulary of 1000 words, then the whole document will be represented by a 1000-dimensional vector, where the vector’s ith entry represents the frequency of the ith vocabulary word in the document.<br>
Some common uses of the bag of words method include spam filtering, sentiment analysis, and language identification.
# Transformers

## Main Paper
[Attention is all you need](Transformer AI (Attention is all you need))

