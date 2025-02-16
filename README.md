# Deep-Learning-Image-Classification

### 1. The Problem: Image Classification

In regular programming, you write rules and conditions in your code so the program knows what to do. This works great for many different problems.

But when it comes to image classification, getting a program to recognize and sort images it has never seen before.. in traditional programming is near impossible to solve it. How could a human could possibly write enough rules to correctly classify tons of different images?

### 2. The Solution: Deep Learning
Deep learning is great at recognizing patterns through trial and error. By training a deep neural network with sufficient data, and providing feedback on its performance via training, the network can identify, though a huge iteration, its own set of conditions by which it can act in the correct way.

### 3. Tensors
If a vector is a 1-dimensional array, and a matrix is a 2-dimensional array, a tensor is an n-dimensional array representing any number of dimensions. Most modern neural network frameworks are powerful tensor processing tools.

One example of a 3-dimensional tensor could be pixels on a computer screen. The different dimensions would be **width**, **height**, and **color** channel. Video games use matrix mathematics to calculate pixel values in a similar way to how neural networks calculate tensors. This is why GPUs are effective tensor processesing machines.

We'll convert our images into tensors so we process them with a neural network