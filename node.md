### 1 NN
Neural networks are compound model classes that divide classification into two or more stages.

Each stage uses a linear model to seperate the data. And then an activation function to reshape it.

<img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/72d8591c-1eba-4a34-9ca0-eaa0f8bad819" />
<img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/9f637297-2bab-4090-9939-fd70aeaf1ecc" />

Using ReLU activating function:

<img width="350" height="200" alt="image" src="https://github.com/user-attachments/assets/52118a13-a9d4-41e4-b715-dba1874b74fa" />
<img width="350" height="200" alt="image" src="https://github.com/user-attachments/assets/ae0c92bf-3fde-43dc-beb6-ecf6b0ec650b" />

Basically the right X's are thresholed to positive values and the other O's and X's are 0.

Since all the O's are now at the origin it is very easy to separate out the space:

<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/626e6b7a-7129-4f9b-b38d-f7cd3335322e" />

Mathematically, we can understand it this way:

<img width="600" height="125" alt="image" src="https://github.com/user-attachments/assets/b4ff97d7-79a8-4709-bc47-06239ac1a97b" />


### 2 Autodiff
- collect information about the computation path used within the function
- transform this information into a procedure for computing derivatives
#### (1) Scalar
number -> class Scalar

User cannot tell the difference between number and Scalar. 

But we utilize the extra information in Scalar to implement the operations we need.

Graphically, we can think of functions as little boxes: (Computation Graph)

<img width="640" height="180" alt="image" src="https://github.com/user-attachments/assets/e4471c7c-9447-4cc5-9aa1-e846a7a41675" />


override operators:
```out = Mul.apply(x, y) -> out = x * y```

#### (2) Autodifferentiation
- implement the derivative of each invidual function
- utilize the chain rule to compute a derivative for any scale value

for common function f - ```forward```

calculate f' in advance - ```backward```

<img width="770" height="145" alt="image" src="https://github.com/user-attachments/assets/803035d8-c4ea-42f9-82e4-474ed6825688" />

Chain Rule:

<img width="330" height="88" alt="image" src="https://github.com/user-attachments/assets/75dbad0a-cba8-4851-9435-62c79239dda1" />

particularly,

<img width="422" height="209" alt="image" src="https://github.com/user-attachments/assets/3e1ce628-9c84-49e5-98ac-b29a361e6bf9" />


### 3 Tensors
#### (1) Basic
- shape: ```(x, y, z)```

  some tricks: change external view without changing inner storage
  - permute: reorders the dimensions
  
    the ```order[i]```-th dimension -> the ```i```-th dimension
    
  - view: increase or decrease the dimension

    ```tensor.view(new_shape) //new_shape's size = shape's size```

- size: ```x * y * z```
  
- dims: ```2/3/.../n```
  
- index: ```tensor[i][j][k]```
  
- storage: flattens to 1 dimension
  - row-major / col-major:

  <img width="300" height="250" alt="image" src="https://github.com/user-attachments/assets/b544a630-2283-4949-9c19-48787c198c4e" />
  <img width="370" height="300" alt="image" src="https://github.com/user-attachments/assets/b13654b5-45b4-422c-af49-4d3467a558e7" />
- stride:
  
  e.g. ```stride=(2, 1)```: each row moves 2 steps in storage and each column moves 1 step

  - provides the mapping from index to the position in storage
    
    ```storage[s1 * index1 + s2 * index2 + s3 * index3 ... ]```

  - stride + shape: we can easily manipulate how we view the same underlying storage
    ```(s1, 1)``` -> row-major, s1 elements/row
    
    ```(1, s2)``` -> col-major, s2 elements/col

    
#### (2) Broadcast
- Number of dimensions does not match -> add dimensions of size 1 on the **left** side
- For the same dimension, size does not match -> turn size 1 to size n (view it as **be copied** n times)

Examples:

<img width="1554" height="228" alt="image" src="https://github.com/user-attachments/assets/c362d931-f7a6-4d5e-b770-263cb51d0420" />
<img width="1541" height="221" alt="image" src="https://github.com/user-attachments/assets/183e1aa8-ff64-4372-9ecc-07d10132f60c" />

#### (3) Operations
- map: for each element ```x``` -> ```fn(x)``` (independently)
  
  <img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/c95aadca-671c-4d20-b0f0-2456eee0edae" />

- zip
  
  <img width="600" height="200" alt="image" src="https://github.com/user-attachments/assets/b36d5fce-a349-40ab-b9d0-6ae436c60662" />

- reduce

  different ```reduce_dim```:

  <img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/befa4350-3267-41d7-9e17-1903131ec02a" />
  <img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/916f9635-8242-48c7-bfb7-31330a906591" />

