---
layout: single
title:  "bream_smelt 예제입니다. !!!"
categories: coding
tag: [python, blog, jekyll]
toc: true
---

```python
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
```


```python
fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
fish_target = [1]*35 + [0]*14
```


```python
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
```


```python
print(fish_data[4])
```

    [29.0, 430.0]
    


```python
print(fish_data[0:5])
```

    [[25.4, 242.0], [26.3, 290.0], [26.5, 340.0], [29.0, 363.0], [29.0, 430.0]]
    


```python
print(fish_data[:5])
```

    [[25.4, 242.0], [26.3, 290.0], [26.5, 340.0], [29.0, 363.0], [29.0, 430.0]]
    


```python
print(fish_data[44:])
```

    [[12.2, 12.2], [12.4, 13.4], [13.0, 12.2], [14.3, 19.7], [15.0, 19.9]]
    


```python
train_input = fish_data[:35]
train_target = fish_target[:35]
test_input = fish_data[35:]
test_target = fish_target[35:]
```


```python
kn = kn.fit(train_input, train_target)
kn.score(test_input, test_target)
```




    0.0




```python
import numpy as np
```


```python
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)
```


```python
print(input_arr)
```

    [[  25.4  242. ]
     [  26.3  290. ]
     [  26.5  340. ]
     [  29.   363. ]
     [  29.   430. ]
     [  29.7  450. ]
     [  29.7  500. ]
     [  30.   390. ]
     [  30.   450. ]
     [  30.7  500. ]
     [  31.   475. ]
     [  31.   500. ]
     [  31.5  500. ]
     [  32.   340. ]
     [  32.   600. ]
     [  32.   600. ]
     [  33.   700. ]
     [  33.   700. ]
     [  33.5  610. ]
     [  33.5  650. ]
     [  34.   575. ]
     [  34.   685. ]
     [  34.5  620. ]
     [  35.   680. ]
     [  35.   700. ]
     [  35.   725. ]
     [  35.   720. ]
     [  36.   714. ]
     [  36.   850. ]
     [  37.  1000. ]
     [  38.5  920. ]
     [  38.5  955. ]
     [  39.5  925. ]
     [  41.   975. ]
     [  41.   950. ]
     [   9.8    6.7]
     [  10.5    7.5]
     [  10.6    7. ]
     [  11.     9.7]
     [  11.2    9.8]
     [  11.3    8.7]
     [  11.8   10. ]
     [  11.8    9.9]
     [  12.     9.8]
     [  12.2   12.2]
     [  12.4   13.4]
     [  13.    12.2]
     [  14.3   19.7]
     [  15.    19.9]]
    


```python
np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)
```


```python
print(index)
```

    [13 45 47 44 17 27 26 25 31 19 12  4 34  8  3  6 40 41 46 15  9 16 24 33
     30  0 43 32  5 29 11 36  1 21  2 37 35 23 39 10 22 18 48 20  7 42 14 28
     38]
    


```python
print(input_arr[[1,3]])
```

    [[ 26.3 290. ]
     [ 29.  363. ]]
    


```python
train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]

```


```python
print(input_arr[13], train_input[0])
```

    [ 32. 340.] [ 32. 340.]
    


```python
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]
```


```python
import matplotlib.pyplot as plt
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```


    
![png](../images/output_18_0.png)
    



```python
kn = kn.fit(train_input, train_target)
```


```python
kn.score(test_input, test_target)
```




    1.0




```python
kn.predict([[25,150]])
```




    array([0])




```python
test_input
```




    array([[ 10.6,   7. ],
           [  9.8,   6.7],
           [ 35. , 680. ],
           [ 11.2,   9.8],
           [ 31. , 475. ],
           [ 34.5, 620. ],
           [ 33.5, 610. ],
           [ 15. ,  19.9],
           [ 34. , 575. ],
           [ 30. , 390. ],
           [ 11.8,   9.9],
           [ 32. , 600. ],
           [ 36. , 850. ],
           [ 11. ,   9.7]])


