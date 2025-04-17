#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import sympy
from scipy import optimize
sympy.init_printing()


# In[ ]:


np.m


# In[16]:


def geometric_mean(x):
    return np.exp(np.mean(np.log(x)))
    


# In[21]:


example = np.array([2, 4, 8, ])


# In[22]:


geometric_mean(example)


# In[3]:


import sys


# In[4]:


sys.path


# In[5]:


import os


# In[10]:


os.add_dll_directory('C:\\Users\\user\\anaconda3\\DLLs')


# In[13]:


import cvxopt


# In[ ]:


## minimization of the area of a cylinder with a unit volume with sympy
r, h = sympy.symbols("r, h")
Area = 2 * sympy.pi * r**2 + 2 * sympy.pi * r * h
Volume = sympy.pi * r**2 * h
h_r = sympy.solve(Volume - 1)[0]
Area_r = Area.subs(h_r)
rsol = sympy.solve(Area_r.diff(r))[0]


# In[ ]:


_.evalf()


# In[ ]:


Area_r.diff(r, 2).subs(r, rsol)


# In[ ]:


Area_r.subs(r, rsol)


# In[ ]:


_.evalf()


# In[ ]:


## Using scipy optimization function for minimizing the area of a cylinder with unit volume


# In[ ]:


def f(r):
    return 2 * np.pi * r**2 + 2 / r
r_min = optimize.brent(f, brack=(0.1, 4))


# In[ ]:


r_min


# In[ ]:


f(r_min)


# In[ ]:


##using optimize minimize scalar function in scipy


# In[ ]:


optimize.minimize_scalar(f, bracket=(0.1, 4))


# In[ ]:


x = np.linspace(0, 2, 100)
len(x)


# In[ ]:


## plotting the objective function
r = np.linspace(1, 2, 10)
f = lambda r :2*np.pi*r**2 +2/r


fig, ax =plt.subplots(figsize =(12, 4))
ax.plot(r, f(r), lw = 1)
ax.axhline(0, ls =':', color = 'k')
ax.set_ylim(0, 30)
ax.set_xticks([0, 0.5, 1, 1.5, 2])
ax.set_title(r'$f(r) =2r^2 + 2/r$')


# In[ ]:


x1, x2 = sympy.symbols("x_1, x_2")
f_sym = (x1-1)**4 + 5 * (x2-1)**2 - 2*x1*x2
fprime_sym = [f_sym.diff(x_) for x_ in (x1, x2)]
sympy.Matrix(fprime_sym)


# In[ ]:


fhess_sym = [[f_sym.diff(x1_, x2_) for x1_ in (x1, x2)] for x2_ in (x1, x2)]


# In[ ]:


fhess_sym


# In[ ]:


## creating vectorised function using lambdify from sympy
f_lmbda = sympy.lambdify((x1, x2), f_sym, 'numpy')
fprime_lmbda = sympy.lambdify((x1, x2), fprime_sym, 'numpy')
fhess_lmbda = sympy.lambdify((x1, x2), fhess_sym, 'numpy')


# In[ ]:


#wrapping lambdify function to be compatible with scipy optimization function
def func_XY_to_X_Y(f):
    """
    Wrapper for f(X) -> f(X[0], X[1])
    """
    return lambda X: np.array(f(X[0], X[1]))


# In[ ]:


f = func_XY_to_X_Y(f_lmbda)
fprime = func_XY_to_X_Y(fprime_lmbda)
fhess = func_XY_to_X_Y(fhess_lmbda)


# In[ ]:


x_opt = optimize.fmin_ncg(f, (0, 0), fprime=fprime, fhess=fhess)


# In[ ]:


x_opt


# ## Diagonistic Information about the evaluation

# In[ ]:


fig, ax = plt.subplots(figsize=(6, 4))
x_ = y_ = np.linspace(-1, 4, 100)
X, Y = np.meshgrid(x_, y_)
c = ax.contour(X, Y, f_lmbda(X, Y), 50)
ax.plot(x_opt[0], x_opt[1], 'r*', markersize=15)
ax.set_xlabel(r"$x_1$", fontsize=18)
ax.set_ylabel(r"$x_2$", fontsize=18)
plt.colorbar(c, ax=ax)


# In[ ]:


### optimize the following functions
def f(X):
    x, y = X
    return (4 * np.sin(np.pi * x) + 6 * np.sin(np.pi * y)) + (x - 1)**2 + (y - 1)**2


# In[ ]:


x_start = optimize.brute(f, (slice(-3, 5, 0.5), slice(-3, 5, 0.5)), finish=None)


# In[ ]:


x_start


# In[ ]:


f(x_start)


# In[ ]:


x_opt = optimize.fmin_bfgs(f, x_start)


# In[ ]:


x_opt


# In[ ]:


f(x_opt)


# ## Visualisation

# In[ ]:


def func_X_Y_to_XY(f, X, Y):
       """
       Wrapper for f(X, Y) -> f([X, Y])
       """
       s = np.shape(X)
       return f(np.vstack([X.ravel(), Y.ravel()])).reshape(*s)


# In[ ]:


fig, ax = plt.subplots(figsize=(6, 4))
x_ = y_ = np.linspace(-3, 5, 100)
X, Y = np.meshgrid(x_, y_)
c = ax.contour(X, Y, func_X_Y_to_XY(f, X, Y), 25)
ax.plot(x_opt[0], x_opt[1], 'r*', markersize=15)
ax.set_xlabel(r"$x_1$", fontsize=18)
ax.set_ylabel(r"$x_2$", fontsize=18)
plt.colorbar(c, ax=ax)


# In[ ]:


#using optimize minimize
result = optimize.minimize(f, x_start, method= 'BFGS')


# In[ ]:


result.x


# ## Least Square Optimization
# 

# In[ ]:


beta = (0.25, 0.75, 0.5)
def f(x, b0, b1, b2):
    return b0 + b1 * np.exp(-b2 * x**2)


# In[ ]:


xdata = np.linspace(0, 5, 50)
y = f(xdata, *beta)
ydata = y + 0.05 * np.random.randn(len(xdata))


# In[ ]:


def g(beta):
    return ydata - f(xdata, *beta)


# In[ ]:


beta_start = (1, 1, 1)
beta_opt, beta_cov = optimize.leastsq(g, beta_start)


# In[ ]:


beta_opt


# In[ ]:


fig, ax = plt.subplots()
ax.scatter(xdata, ydata, label='samples')
ax.plot(xdata, y, 'r', lw=2, label='true model')
ax.plot(xdata, f(xdata, *beta_opt), 'b', lw=2, label='fitted model')
ax.set_xlim(0, 5)
ax.set_xlabel(r"$x$", fontsize=18)
ax.set_ylabel(r"$f(x, \beta)$", fontsize=18)
ax.legend()


# ## Linear optimization with constrained

# In[ ]:


def f(X):
   x, y = X
   return (x - 1)**2 + (y - 1)**2


# In[ ]:


x_opt = optimize.minimize(f, [1, 1], method='BFGS').x


# In[ ]:


bnd_x1, bnd_x2 = (2, 3), (0, 2)
x_cons_opt = optimize.minimize(f, [1, 1], method='L-BFGS-B',bounds=[bnd_x1, bnd_x2]).x


# In[ ]:


fig, ax = plt.subplots(figsize=(6, 4))
x_ = y_ = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_, y_)
c = ax.contour(X, Y, func_X_Y_to_XY(f, X, Y), 50)
ax.plot(x_opt[0], x_opt[1], 'b*', markersize=15)
ax.plot(x_cons_opt[0], x_cons_opt[1], 'r*', markersize=15)
bound_rect = plt.Rectangle((bnd_x1[0], bnd_x2[0]),bnd_x1[1] - bnd_x1[0], bnd_x2[1] - bnd_x2[0], facecolor="grey")
ax.add_patch(bound_rect)
ax.set_xlabel(r"$x_1$", fontsize=18)
ax.set_ylabel(r"$x_2$", fontsize=18)
plt.colorbar(c, ax=ax)


# ## Maximizing the volume of a rectangle with area of rectangle equal 1

# In[ ]:


x = x0, x1, x2, l = sympy.symbols("x_0, x_1, x_2, lambda")
f = x0 * x1 * x2
g = 2 * (x0 * x1 + x1 * x2 + x2 * x0) - 1
L = f + l * g


# In[ ]:


grad_L = [sympy.diff(L, x_) for x_ in x]


# In[ ]:


sols = sympy.solve(grad_L)
sols


# In[ ]:


f.subs(sols[0])


# In[ ]:


g.subs(sols[0])


# ## Maximizing the volume of a cube numerically

# In[ ]:


def f(X):
    return -X[0] * X[1] * X[2]
def g(X):
    return 2 * (X[0]*X[1] + X[1] * X[2] + X[2] * X[0]) - 1


# In[ ]:


constraint = dict(type='eq', fun=g)
result = optimize.minimize(f, [0.5, 1, 1.5], method='SLSQP', constraints=[constraint])


# In[ ]:


result


# In[ ]:


def f(X):
    return (X[0] - 1)**2 + (X[1] - 1)**2


# In[ ]:


def g(X):
    return X[1] - 1.75 - (X[0] - 0.75)**4


# In[ ]:


constraints = [dict(type='ineq', fun=g)]


# In[ ]:


x_opt = optimize.minimize(f, (0, 0), method='BFGS').x


# In[ ]:


x_cons_opt = optimize.minimize(f, (0, 0), method='SLSQP', constraints=constraints).x


# In[ ]:


x_opt


# In[ ]:


x_cons_opt


# In[ ]:


fig, ax = plt.subplots(figsize=(6, 4))
x_ = y_ = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_, y_)
c = ax.contour(X, Y, func_X_Y_to_XY(f, X, Y), 50)
ax.plot(x_opt[0], x_opt[1], 'b*', markersize=15)
ax.plot(x_, 1.75 + (x_-0.75)**4, 'k-', markersize=15)
ax.fill_between(x_, 1.75 + (x_-0.75)**4, 3, color='grey')
ax.plot(x_cons_opt[0], x_cons_opt[1], 'r*', markersize=15)
ax.set_ylim(-1, 3)
ax.set_xlabel(r"$x_0$", fontsize=18)
ax.set_ylabel(r"$x_1$", fontsize=18)
plt.colorbar(c, ax=ax)


# ## Solving Linear programming with python

# In[2]:


## f(x) =-x + 2y -3z
## subject to constraint
## x+y <1
## -x +3y <2
## -y+ z< 3


# In[4]:


c = np.array([-1.0, 2.0, -3.0])
A = np.array([[ 1.0, 1.0, 0.0],[-1.0, 3.0, 0.0],[ 0.0, -1.0, 1.0]])
b = np.array([1.0, 2.0, 3.0])


# In[ ]:


A_ = cvxopt.matrix(A)
b_ = cvxopt.matrix(b)
c_ = cvxopt.matrix(c)


# In[ ]:


sol =cvxopt.solvers.lp(c_, A_, b_)

