
 <a href="https://www.bigdatauniversity.com"><img src = "https://ibm.box.com/shared/static/ugcqz6ohbvff804xp84y4kqnvvk3bq1g.png" width = 300, align = "center"></a>

<h1 align=center><font size = 5>Data Analysis with Python</font></h1>

#### Recommended, Python for Data Science click to start course:
  
  <a href="http://cocl.us/DA0101ENtoPY0101EN"><img src = "https://ibm.box.com/shared/static/jmtb4pgle2dsdlzfmyrgv755cnqw95wk.png" width = 300, align = "center"></a>

# Module 4: Model Development

In this section, we will develop several models that will predict the price of the car using the variables or features. This is just an estimate but should give us an objective idea of how much the car should cost.

Some questions we want to ask in this module
- How do I know if the dealer is offering fair value for my trade-in? 
- How do I know if I put a fair value on my car?
In Data Analytics, we often use **Model Development** to help us predict future observations from the data we have. 

A Model will help us understand the exact relationship between different variables and how these variables are used to predict the result.

#### Setup

 Import libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

load data and store in dataframe df:


```python
# path of data 
path = 'https://ibm.box.com/shared/static/q6iiqb1pd7wo8r3q28jvgsrprzezjqk3.csv'
df = pd.read_csv(path)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>symboling</th>
      <th>normalized-losses</th>
      <th>make</th>
      <th>aspiration</th>
      <th>num-of-doors</th>
      <th>body-style</th>
      <th>drive-wheels</th>
      <th>engine-location</th>
      <th>wheel-base</th>
      <th>...</th>
      <th>compression-ratio</th>
      <th>horsepower</th>
      <th>peak-rpm</th>
      <th>city-mpg</th>
      <th>highway-mpg</th>
      <th>price</th>
      <th>city-L/100km</th>
      <th>horsepower-binned</th>
      <th>diesel</th>
      <th>gas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>122</td>
      <td>alfa-romero</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>9.0</td>
      <td>111.0</td>
      <td>5000.0</td>
      <td>21</td>
      <td>27</td>
      <td>13495.0</td>
      <td>11.190476</td>
      <td>Medium</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>122</td>
      <td>alfa-romero</td>
      <td>std</td>
      <td>two</td>
      <td>convertible</td>
      <td>rwd</td>
      <td>front</td>
      <td>88.6</td>
      <td>...</td>
      <td>9.0</td>
      <td>111.0</td>
      <td>5000.0</td>
      <td>21</td>
      <td>27</td>
      <td>16500.0</td>
      <td>11.190476</td>
      <td>Medium</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>122</td>
      <td>alfa-romero</td>
      <td>std</td>
      <td>two</td>
      <td>hatchback</td>
      <td>rwd</td>
      <td>front</td>
      <td>94.5</td>
      <td>...</td>
      <td>9.0</td>
      <td>154.0</td>
      <td>5000.0</td>
      <td>19</td>
      <td>26</td>
      <td>16500.0</td>
      <td>12.368421</td>
      <td>Medium</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>fwd</td>
      <td>front</td>
      <td>99.8</td>
      <td>...</td>
      <td>10.0</td>
      <td>102.0</td>
      <td>5500.0</td>
      <td>24</td>
      <td>30</td>
      <td>13950.0</td>
      <td>9.791667</td>
      <td>Medium</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2</td>
      <td>164</td>
      <td>audi</td>
      <td>std</td>
      <td>four</td>
      <td>sedan</td>
      <td>4wd</td>
      <td>front</td>
      <td>99.4</td>
      <td>...</td>
      <td>8.0</td>
      <td>115.0</td>
      <td>5500.0</td>
      <td>18</td>
      <td>22</td>
      <td>17450.0</td>
      <td>13.055556</td>
      <td>Medium</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>



### 1. Linear Regression and Multiple Linear Regression 

####  Linear Regression


One example of a Data  Model that we will be using is 
 **Simple Linear Regression**.
Simple Linear Regression is a method to help us understand the relationship between two variables:
- The predictor/independent variable (X)
- The response/dependent variable (that we want to predict)(Y)


The result of Linear Regression is a **linear function** that predicts the response (dependent) variable as a function of the predictor (independent) variable. 



\begin{equation*}
 Y: Response \ Variable\\
 X :Predictor\ Variables
\end{equation*}


 **Linear function:**
\begin{equation*}
Yhat = a + b  X
\end{equation*}


- a refers to the **intercept** of the regression, in other words: the value of Y when X is 0 
- b refers to the **slope** of the regression line, in other words: the value with which Y changes when X increases by 1.





####  Lets load the modules for linear regression


```python
from sklearn.linear_model import LinearRegression
```

#### Create the linear regression object


```python
lm = LinearRegression()
lm
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



#### How could Highway-mpg help us predict car price?

For this example, we want to look at how highway-mpg can help us predict car price.
Using simple linear regression, we will create a linear function with "highway-mpg" as the predictor variable and the "price" as the response variable.


```python
X = df[['highway-mpg']]
Y = df['price']
```

Fit the linear model using highway-mpg.


```python
lm.fit(X,Y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



 We can output a prediction 


```python
Yhat=lm.predict(X)
Yhat[0:5]   
```




    array([16236.50464347, 16236.50464347, 17058.23802179, 13771.3045085 ,
           20345.17153508])



#### What is the value of the intercept (a) ?


```python
lm.intercept_
```




    38423.3058581574



#### What is the value of the Slope (b) ?


```python
lm.coef_
```




    array([-821.73337832])



### What is the final estimated linear model we get?

As we saw above, we should get a final linear model with the structure:

 \begin{equation*}
Yhat = a + b  X
\end{equation*}

Plugging in the actual values we get:

**price** = 38423.31 - 821.73 x  **highway-mpg**

 <div class="alert alert-danger alertdanger" style="margin-top: 20px">
<h1> Question  #1a): </h1>

<b>Create a linear regression object?</b>
</div>


```python
lm = LinearRegression()
lm
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



Double-click __here__ for the solution.

<!-- Your answer is below:

lm1 = LinearRegression()
lm1 

-->

<div class="alert alert-danger alertdanger" style="margin-top: 20px">
<h1> Question  #1b): </h1>

<b> Train the model using 'engine-size' as the independent variable and 'price' as the dependent variable?</b>
</div>


```python
X = df[['engine-size']]
Y = df['price']
lm.fit(X,Y)
Yhat=lm.predict(X)
Yhat[0:5]   
```




    array([13728.4631336 , 13728.4631336 , 17399.38347881, 10224.40280408,
           14729.62322775])



Double-click __here__ for the solution.

<!-- Your answer is below:

lm1.fit(df[['highway-mpg']], df[['price']])
lm1

-->

 <div class="alert alert-danger alertdanger" style="margin-top: 20px">
<h1> Question  #1c): </h1>

<b> Find the slope and intercept of the model?</b>
</div>

#### Slope 


```python
lm.coef_
```




    array([166.86001569])



#### Intercept


```python

lm.intercept_
```




    -7963.338906281042



Double-click __here__ for the solution.

<!-- Your answer is below:

# Slope 
lm1.coef_
# Intercept
lm1.intercept_

-->

<div class="alert alert-danger alertdanger" style="margin-top: 20px">
<h1> Question  #1d): </h1>

<b> What is the equation of the predicted line. You can use x and yhat or ''engine-size'  or  'price'?</b>
</div>

Double-click __here__ for the solution.

<!-- Your answer is below:

# using X and Y  
Yhat=38423.31-821.733*X

Price=38423.31-821.733*engine-size

-->

####  Multiple Linear Regression

 What if we want to predict car price using more than one variable? 

If we want to use more variables in our model to predict car price, we can use **Multiple Linear Regression**.
Multiple Linear Regression is very similar to Simple Linear Regression, but this method is used to explain the relationship between one continuous response (dependent) variable and *two or more* predictor (independent) variables.
Most of the real-world regression models involve multiple predictors. We illustrate the structure by using four predictor variables, but these results can generalize to any integer :


 \begin{equation*}
Y: Response \ Variable\\
X_1 :Predictor\ Variable \ 1\\
X_2: Predictor\ Variable \ 2\\
X_3: Predictor\ Variable \ 3\\
X_4: Predictor\ Variable \ 4\\
\end{equation*}


 \begin{equation*}
a: intercept\\
b_1 :coefficients \ of\ Variable \ 1\\
b_2: coefficients \ of\ Variable \ 2\\
b_3: coefficients \ of\ Variable \ 3\\
b_4: coefficients \ of\ Variable \ 4\\
\end{equation*}


 The equation is given by 

 \begin{equation*}
Yhat = a + b_1 X_1 + b_2 X_2 + b_3 X_3 + b_4 X_4
\end{equation*}



From the previous section  we know that other good predictors of price could be: 
- Horsepower
- Curb-weight
- Engine-size
- Highway-mpg

Let's develop a model using these variables as the predictor variables.


```python
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
```

Fit the linear model using the four above-mentioned variables.


```python
 lm.fit(Z, df['price'])

```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



 What is the value of the intercept(a)?


```python
lm.intercept_
```




    -15806.62462632922



 What are the values of the coefficients (b1, b2, b3, b4) ?


```python
lm.coef_
```




    array([53.49574423,  4.70770099, 81.53026382, 36.05748882])



 What is the final estimated linear model that we get?

As we saw above, we should get a final linear function with the structure:

 \begin{equation*}
Yhat = a + b_1 X_1 + b_2 X_2 + b_3 X_3 + b_4 X_4
\end{equation*}

What is the linear function we get in this example?

**Price** = -15678.742628061467 + 52.65851272 x **horsepower** + 4.69878948 x **curb-weight** + 81.95906216 x **engine-size** + 33.58258185 x **highway-mpg**

<div class="alert alert-danger alertdanger" style="margin-top: 20px">
<h1> Question  #2a): </h1>
Create and train a  Multiple Linear Regression model "lm2" where the response variable is price, and the predictor variable is  'normalized-losses' and  'highway-mpg'.
<b> ?</b>
</div>


```python
Z = df[['normalized-losses', 'highway-mpg']]
lm2= lm.fit(Z, df['price'])
lm2.coef_
```




    array([   1.49789586, -820.45434016])



Double-click __here__ for the solution.

<!-- Your answer is below:

lm2 = LinearRegression()
lm2.fit(df[['normalized-losses' , 'highway-mpg']],df['price'])

-->

<div class="alert alert-danger alertdanger" style="margin-top: 20px">
<h1> Question  #2a): </h1>
<b> Find the coefficient of the model?</b>
</div>

Double-click __here__ for the solution.

<!-- Your answer is below:

lm2.coef_

-->

### 2)  Model Evaluation using Visualization

Now that we've developed some models, how do we evaluate our models and how do we choose the best one? One way to do this is by using visualization.

import the visualization package: seaborn


```python
# import the visualization package: seaborn
import seaborn as sns
%matplotlib inline 
```

### Regression Plot

When it comes to simple linear regression, an excellent way to visualise the fit of our model is by using **regression plots**.

This plot will show a combination of a scattered data points (a **scatterplot**), as well as the fitted **linear regression** line going through the data. This will give us a reasonable estimate of the relationship between the two variables, the strength of the correlation, as well as the direction (positive or negative correlation).

 Let's visualize Horsepower as potential predictor variable of price:


```python
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
```




    (0, 48272.843591699864)




![output_79_1](https://user-images.githubusercontent.com/36623487/44046438-4fd69d14-9ee0-11e8-9f34-ee412679bca6.png)


We can see from this plot that price is negatively correlated to highway-mpg, since the regression slope is negative.
One thing to keep in mind when looking at a regression plot is to pay attention to how scattered the data points are around the regression line. This will give you a good indication of the variance of the data, and whether a linear model would be the best fit or not. If the data is too far off from the line, this linear model might not be the best model for this data. Let's compare this plot to the regression plot of "peak-rpm".


```python
plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)
```




    (0, 47422.919330307624)




![output_81_1](https://user-images.githubusercontent.com/36623487/44046439-501724c4-9ee0-11e8-9a75-0b3c08296aa6.png)


Comparing the regression plot of "peak-rpm" and "highway-mpg" We see that the points for "highway-mpg" are much closer to the generated line and on the average decrease. The points for "peak-rpm"  have more spread around the predicted line, and it is much harder to determine if the points are decreasing or increasing as the  "highway-mpg"  increases.

 <div class="alert alert-danger alertdanger" style="margin-top: 20px">
<h1> Question  #3: </h1>
<b> Given the regression plots above is "peak-rpm" or "highway-mpg"  more strongly correlated with "price". Use the method  ".corr()"  to verify your answer. </b>
</div>


```python
cor1= df[['peak-rpm','price']].corr()
cor2= df[['highway-mpg','price']].corr()
cor3= df[["peak-rpm","highway-mpg","price"]].corr()
print(cor1)
print("")
print(cor2)
print("")
print(cor3)
```

              peak-rpm     price
    peak-rpm  1.000000 -0.101616
    price    -0.101616  1.000000
    
                 highway-mpg     price
    highway-mpg     1.000000 -0.704692
    price          -0.704692  1.000000
    
                 peak-rpm  highway-mpg     price
    peak-rpm     1.000000    -0.058598 -0.101616
    highway-mpg -0.058598     1.000000 -0.704692
    price       -0.101616    -0.704692  1.000000


Double-click __here__ for the solution.

<!-- Your answer is below:

The variable "peak-rpm"  has a stronger correlation with "price", it is approximate -0.704692  compared to   "highway-mpg" which is approximate     -0.101616. You can verify it using the following command:
df[["peak-rpm","highway-mpg","price"]].corr()

-->

### Residual Plot

A good way to visualize the variance of the data is to use a residual plot.

What is a **residual**?

The difference between the observed value (y) and the predicted value (Yhat) is called the residual (e). When we look at a regression plot, the residual is the distance from the data point to the fitted regression line.

So what is a **residual plot**?

A residual plot is a graph that shows the residuals on the vertical y-axis and the independent variable on the horizontal x-axis.

What do we pay attention to when looking at a residual plot?

We look at the spread of the residuals:

- If the points in a residual plot are **randomly spread out around the x-axis**, then a **linear model is appropriate** for the data. Why is that? Randomly spread out residuals means that the variance is constant, and thus the linear model is a good fit for this data.


```python
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['highway-mpg'], df['price'])
plt.show()
```


![output_87_0](https://user-images.githubusercontent.com/36623487/44046440-504ab3f2-9ee0-11e8-9b4b-0ec82e71eae2.png)


*What is this plot telling us?*

We can see from this residual plot that the residuals are not randomly spread around the x-axis, which leads us to believe that maybe a non-linear model is more appropriate for this data.

###  Multiple Linear Regression

How do we visualise a model for Multiple Linear Regression? This gets a bit more complicated because you can't visualise it with regression or residual plot. 

One way to look at the fit of the model is by looking at the **distribution plot**: We can look at the distribution of the fitted values that result from the model and compare it to the distribution of the actual values.

 First lets make a prediction 


```python
Y_hat = lm.predict(Z)
```


```python
plt.figure(figsize=(width, height))


ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()
```


![output_93_0](https://user-images.githubusercontent.com/36623487/44046441-50801e16-9ee0-11e8-895f-559f875904dc.png)


We can see that the fitted values are reasonably close to the actual values, since the two distributions overlap a bit. However, there is definitely some room for improvement.

## Part 3: Polynomial Regression and Pipelines 

**Polynomial regression** is a particular case of the general linear regression model or multiple linear regression models. 
We get non-linear relationships by squaring or setting higher-order terms of the predictor variables.

There are different orders of polynomial regression:

<center>**Quadratic - 2nd order**</center>


 \begin{equation*}
Yhat = a + b_1 X^2 +b_2 X^2 
\\
\end{equation*}


 <center>**Cubic - 3rd order**</center>
 
 
 \begin{equation*}
Yhat = a + b_1 X^2 +b_2 X^2 +b_3 X^3\\
\end{equation*}

<center> **Higher order**:</center>


 \begin{equation*}
Y = a + b_1 X^2 +b_2 X^2 +b_3 X^3 ....\\
\end{equation*}

We saw earlier that a linear model did not provide the best fit while using highway-mpg as the predictor variable. Let's see if we can try fitting a polynomial model to the data instead.

 We will use the following function to plot the data:


```python
def PlotPolly(model,independent_variable,dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable,dependent_variabble,'.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_axis_bgcolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()
    
print("done")
```

    done


lets get the variables  


```python
x = df['highway-mpg']
y = df['price']
print("done")
```

    done


Let's fit the polynomial using the function **polyfit**, then use the function **poly1d** to display the polynomial function.


```python
# Here we use a polynomial of the 3rd order (cubic) 
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)
```

            3         2
    -1.557 x + 204.8 x - 8965 x + 1.379e+05


 Let's plot the function 


```python
PlotPolly(p,x,y, 'highway-mpg')
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-47-9f8e124d2c17> in <module>()
    ----> 1 PlotPolly(p,x,y, 'highway-mpg')
    

    <ipython-input-44-df6a50d78b75> in PlotPolly(model, independent_variable, dependent_variabble, Name)
          6     plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
          7     ax = plt.gca()
    ----> 8     ax.set_axis_bgcolor((0.898, 0.898, 0.898))
          9     fig = plt.gcf()
         10     plt.xlabel(Name)


    AttributeError: 'AxesSubplot' object has no attribute 'set_axis_bgcolor'



![output_106_1](https://user-images.githubusercontent.com/36623487/44046443-50b69b3a-9ee0-11e8-848a-8c58f576a845.png)



```python
np.polyfit(x, y, 3)
```




    array([-1.55663829e+00,  2.04754306e+02, -8.96543312e+03,  1.37923594e+05])



We can already see from plotting that this polynomial model performs better than the linear model. This is because the generated polynomial function  "hits" more of the data points.

<div class="alert alert-danger alertdanger" style="margin-top: 20px">
<h1> Question  #4: </h1>
<b> Create  11 order polynomial model with the variables x and y from above?  </b>
</div>


```python
# Here we use a polynomial of the 3rd order (cubic) 
f = np.polyfit(x, y, 11)
p = np.poly1d(f)
print(p)

PlotPolly(p,x,y, 'highway-mpg')

np.polyfit(x, y, 11)
```

                11             10             9           8         7
    -1.243e-08 x  + 4.722e-06 x  - 0.0008028 x + 0.08056 x - 5.297 x
              6        5             4             3             2
     + 239.5 x - 7588 x + 1.684e+05 x - 2.565e+06 x + 2.551e+07 x - 1.491e+08 x + 3.879e+08



    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-50-90fae9bdae95> in <module>()
          4 print(p)
          5 
    ----> 6 PlotPolly(p,x,y, 'highway-mpg')
          7 
          8 np.polyfit(x, y, 11)


    <ipython-input-44-df6a50d78b75> in PlotPolly(model, independent_variable, dependent_variabble, Name)
          6     plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
          7     ax = plt.gca()
    ----> 8     ax.set_axis_bgcolor((0.898, 0.898, 0.898))
          9     fig = plt.gcf()
         10     plt.xlabel(Name)


    AttributeError: 'AxesSubplot' object has no attribute 'set_axis_bgcolor'



![output_110_2](https://user-images.githubusercontent.com/36623487/44046445-51b7ff56-9ee0-11e8-9a6e-9fc83f584df4.png)


Double-click __here__ for the solution.

<!-- Your answer is below:

# calculate polynomial
# Here we use a polynomial of the 3rd order (cubic) 
f1 = np.polyfit(x, y, 11)
p1 = np.poly1d(f1)
print(p)
PlotPolly(p1,x,y, 'Length')

-->

The analytical expression for Multivariate Polynomial function gets complicated. For example, the expression for a second-order (degree=2)polynomial with two variables is given by:

\begin{equation*}
Yhat = a + b_1 X_1 +b_2 X_2 +b_3 X_1 X_2+b_4 X_1^2+b_5 X_2^2
\end{equation*}

 We can perform a polynomial transform on multiple features. First, we import the  module:


```python
from sklearn.preprocessing import PolynomialFeatures
```

We create a **PolynomialFeatures** object of degree 2: 


```python
pr=PolynomialFeatures(degree=2)
pr
```




    PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)




```python
Z_pr=pr.fit_transform(Z)
```

The original data is of 201 samples and 4 features 


```python
Z.shape
```




    (201, 2)



after the transformation, there 201 samples and 15 features


```python
Z_pr.shape
```




    (201, 6)



## Pipeline 

Data Pipelines simplify the steps of processing the data. We use the module  **Pipeline** to create a pipeline. We also use **StandardScaler** as a step in our pipeline.


```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
```

We create the pipeline, by creating a list of tuples including the name of the model or estimator and its corresponding constructor. 


```python
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
```

we input the list as an argument to the pipeline constructor 


```python
pipe=Pipeline(Input)
pipe
```




    Pipeline(memory=None,
         steps=[('scale', StandardScaler(copy=True, with_mean=True, with_std=True)), ('polynomial', PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)), ('model', LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False))])



We can normalize the data,  perform a transform and fit the model simultaneously. 


```python
pipe.fit(Z,y)
```




    Pipeline(memory=None,
         steps=[('scale', StandardScaler(copy=True, with_mean=True, with_std=True)), ('polynomial', PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)), ('model', LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False))])



 Similarly,  we can normalize the data, perform a transform and produce a prediction  simultaneously


```python
ypipe=pipe.predict(Z)
ypipe[0:4]
```




    array([15388.77780567, 15388.77780567, 16771.84474515, 11641.85647791])



<div class="alert alert-danger alertdanger" style="margin-top: 20px">
<h1> Question  #5: </h1>
<b>  Create a pipeline that Standardizes the data, then perform prediction using a linear regression model using the features Z and targets y  </b>
</div>

Double-click __here__ for the solution.

<!-- Your answer is below:

Input=[('scale',StandardScaler()),('model',LinearRegression())]

pipe=Pipeline(Input)

pipe.fit(Z,y)

ypipe=pipe.predict(Z)
ypipe[0:10]

-->

## Part 4: Measures for In-Sample Evaluation

When evaluating our models, not only do we want to visualise the results, but we also want a quantitative measure to determine how accurate the model is.

Two very important measures that are often used in Statistics to determine the accuracy of a model are:

- **R^2 / R-squared**
- **Mean Squared Error (MSE)**

**R-squared**

R squared, also known as the coefficient of determination, is a measure to indicate how close the data is to the fitted regression line.
The value of the R-squared is the percentage of variation of the response variable (y) that is explained by a linear model.



**Mean Squared Error (MSE)**

The Mean Squared Error measures the average of the squares of errors, that is, the difference between actual value (y) and the estimated value (ŷ).

### Model 1: Simple Linear Regression

Let's calculate the R^2


```python
#highway_mpg_fit
lm.fit(X, Y)
# Find the R^2
lm.score(X, Y)
```




    0.7609686443622008



We can say that ~ 49.659% of the variation of the price is explained by this simple linear model "horsepower_fit".

Let's calculate the MSE

We can predict the output i.e., "yhat" using the predict method, where X is the input variable:


```python
Yhat=lm.predict(X)
Yhat[0:4]
```




    array([13728.4631336 , 13728.4631336 , 17399.38347881, 10224.40280408])



 lets import the function **mean_squared_error** from the module **metrics**


```python
from sklearn.metrics import mean_squared_error
```

 we compare the predicted results with the actual results 


```python
#mean_squared_error(Y_true, Y_predict)
mean_squared_error(df['price'], Yhat)
```




    15021126.025174143



### Model 2: Multiple Linear Regression

Let's calculate the R^2


```python
# fit the model 
lm.fit(Z, df['price'])
# Find the R^2
lm.score(Z, df['price'])
```




    0.4966263556974878



We can say that ~ 80.896 % of the variation of price is explained by this multiple linear regression "multi_fit".

Let's calculate the MSE

 we produce a prediction 


```python
Y_predict_multifit = lm.predict(Z)

```

 we compare the predicted results with the actual results 


```python
mean_squared_error(df['price'], Y_predict_multifit)
```




    31632832.975589428



### Model 3: Polynomial Fit

Let's calculate the R^2

 let’s import the function **r2_score** from the module ** metrics** as we are using a different function  



```python
from sklearn.metrics import r2_score
```

We apply the function to get the value of r^2


```python
r_squared = r2_score(y, p(x))
r_squared
```




    0.7023769093756598



We can say that ~ 67.419 % of the variation of price is explained by this polynomial fit

### MSE

 We can also calculate the MSE:  


```python
mean_squared_error(df['price'], p(x))
```




    18703127.63085496



## Part 5: Prediction and Decision Making
### Prediction

In the previous section, we trained the model using the method **fit**. Now we will use the method **predict** to produce a prediction.Lets import **pyplot** for plotting; we will also be using some functions from numpy. 
 



```python
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline 
```

Create a  new input 


```python
new_input=np.arange(1,100,1).reshape(-1,1)
```

 Fit the model 


```python
lm.fit(X, Y)
lm
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



Produce a prediction 


```python
yhat=lm.predict(new_input)
yhat[0:5]
```




    array([-7796.47889059, -7629.6188749 , -7462.75885921, -7295.89884352,
           -7129.03882782])



we can plot the data 


```python
plt.plot(new_input,yhat)
plt.show()
```


![output_177_0](https://user-images.githubusercontent.com/36623487/44046446-51ef2850-9ee0-11e8-895e-b844d0faa85f.png)


### Decision Making: Determining a Good Model Fit

Now that we have visualized the different models, and generated the R-squared and MSE values for the fits, how do we determine a good model fit?

- *What is a good R-squared value?*

When comparing models, **the model with the higher R-squared value is a better fit** for the data.


- *What is a good MSE?*

When comparing models, **the model with the smallest MSE value is a better fit** for the data.#### Let's take a look at the values for the different models we get.

#### Let's take a look at the values for the different models.
Simple Linear Regression: Using Highway-mpg as a Predictor Variable of Price.
- R-squared: 0.49659118843391759
- MSE: 3.16 x10^7

Multiple Linear Regression: Using Horsepower, Curb-weight, Engine-size, and Highway-mpg as Predictor Variables of Price.
- R-squared: 0.80896354913783497
- MSE: 1.2 x10^7

Polynomial Fit: Using Highway-mpg as a Predictor Variable of Price.
- R-squared: 0.6741946663906514
- MSE: 2.05 x 10^7

### Simple Linear Regression model (SLR) vs Multiple Linear Regression model (MLR)

Usually, the more variables you have, the better your model is at predicting, but this is not always true. Sometimes you may not have enough data, you may run into numerical problems, or many of the variables may not be useful and or even act as noise. As a result, you should always check the MSE and R^2. 

So to be able to compare the results of the MLR vs SLR models, we look at a combination of both the R-squared and MSE to make the best conclusion about the fit of the model.
 

- **MSE ** 
The MSE of SLR is  3.16x10^7  while MLR has an MSE of 1.2 x10^7.  The MSE of MLR is much smaller. 


- **R-squared**: 
In this case, we can also see that there is a big difference between the R-squared of the SLR and the R-squared of the MLR. The R-squared for the SLR (~0.497) is very small compared to the R-squared for the MLR (~0.809). 

This R-squared in combination with the MSE show that MLR seems like the better model fit in this case, compared to SLR.

### Simple Linear Model (SLR) vs Polynomial Fit

- **MSE**: We can see that Polynomial Fit brought down the MSE, since this MSE is smaller than the one from the SLR. 

- **R-squared**: The R-squared for the Polyfit is larger than the R-squared for the SLR, so the Polynomial Fit also brought up the R-squared quite a bit.

Since the Polynomial Fit resulted in a lower MSE and a higher R-squared, we can conclude that this was a better fit model than the simple linear regression for predicting Price with Highway-mpg as a predictor variable.

### Multiple Linear Regression (MLR) vs Polynomial Fit

- **MSE**: The MSE for the MLR is smaller than the MSE for the Polynomial Fit.
- **R-squared**: The R-squared for the MLR is also much larger than for the Polynomial Fit.

## Conclusion: 

Comparing these three models, we conclude that **the MLR model is the best model** to be able to predict price from our dataset. This result makes sense, since we have 27 variables in total, and we know that more than one of those variables are potential predictors of the final car price. 

### About the Authors:  

This notebook written by [Mahdi Noorian PhD](https://www.linkedin.com/in/mahdi-noorian-58219234/) ,[Joseph Santarcangelo PhD]( https://www.linkedin.com/in/joseph-s-50398b136/), Bahare Talayian, Eric Xiao, Steven Dong, Parizad , Hima Vsudevan and [Fiorella Wenver](https://www.linkedin.com/in/fiorellawever/).
Copyright &copy; 2017 [cognitiveclass.ai](cognitiveclass.ai?utm_source=bducopyrightlink&utm_medium=dswb&utm_campaign=bdu). This notebook and its source code are released under the terms of the [MIT License](https://bigdatauniversity.com/mit-license/).


 <a href="http://cocl.us/DA0101EN_NotbookLink_bottom"><img src = "https://ibm.box.com/shared/static/cy2mwm7519t4z6dxefjpzgtbpi9p8l7h.png" width = 750, align = "center"></a>

