import numpy as np
import matplotlib.pyplot as plt


x = np.array([2.5, 3.7, 2.6, 5.4, 7.5, 6.9, 9.9])
t = np.linspace(1.0, 7.0, 7)
plt.plot(t, x)

np.random.seed(100)
w = np.random.uniform(-1.0, 1.0, 100).reshape(10, 10)
u = np.random.uniform(-1.0, 1.0, 10).reshape(10, 1)
v = np.random.uniform(-1.0, 1.0, 10).reshape(10, 1)
h = np.zeros(10).reshape(10, 1)

for i in x:
    a = u*i + np.dot(w, h)
    h = np.tanh(a)

y = np.dot(np.transpose(v), h)


def sigmoid(x):
    return 1 / (1 +np.exp(-x))

## tanh
xx = np.linspace(-10.0, 10.0, 500)
y1 = 1-np.tanh(xx)*np.tanh(xx)
y2 = np.tanh(xx)
y3 = sigmoid(xx)

##plt.plot(xx, y1)
plt.plot(xx, y2)
plt.plot(xx, y3)



def sigmoid(x):
    return 1 / (1 +np.exp(-x))
## lstm
x1 = 2.5

## forget gate
np.random.seed(100)
h0 = np.random.standard_normal((3, 1)).round(3)
x1 = np.array([[x1]])
h0x1 = np.concatenate((h0, x1), axis=0)
print("h0x1")
print(h0x1)


wf_hx = np.random.standard_normal((3, 4)).round(3)
print("wf_hx")
print(wf_hx)

ft = sigmoid(np.dot(wf_hx, h0x1))
print("ft")
print(ft)

c0 = np.random.standard_normal((3, 1)).round(3)
print("c0")
print(c0)
print("ft*c0")
print(ft*c0)


## input gate

h0 = np.random.standard_normal((3, 1)).round(3)
h0x1 = np.concatenate((h0, x1), axis=0)

wc_hx = np.random.standard_normal((3, 4)).round(3)
empc0 = np.tanh(np.dot(wc_hx, h0x1))
print("wc_hx")
print(wc_hx)
print("empc0")
print(empc0)

wi_hx = np.random.standard_normal((3, 4)).round(3)
print("wi_hx")
print(wi_hx)

i0 = sigmoid(np.dot(wi_hx, h0x1))
print("i0")
print(i0)

ct = empc0*i0 + ft*c0
print("ct")
print(ct)

w0_hx = np.random.standard_normal((3, 4)).round(3)
print("w0_hx")
print(w0_hx)

h1= np.tanh(ct)*sigmoid(np.dot(w0_hx, h0x1))

print("h1")
print(h1)