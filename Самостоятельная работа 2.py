# -*- coding: utf-8 -*-
"""
@author: artyomshutoff
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from numpy import dot

"""
Данные для построения регрессии
"""
y = np.array([370610, 431820, 289850, 402640, 535717, 589996, 441930, 506420, 318782, 305130], dtype = float)
x = np.array([121500, 123487, 107116, 67640, 187298, 149909, 116213, 91978, 76647, 75827], dtype = float)
y_text = 'Валовый рег. продукт на душу населения'
x_text = 'Инвестиции в основной капитал на душу населения'
"""
Данные для построения регрессии
"""

def f(b, k):
	"""
	Увел. y на условную единицу
	"""
	x0 = (0 - b) / k
	x1 = (1 - b) / k
	return x1 - x0

X = []
Y = []

for i in range(len(x)):
	X.append([1, x[i]])
for i in range(len(y)):
	Y.append([1, y[i]])
X = np.array(X, dtype=float)
Y = np.array(Y, dtype=float)

bk = dot(dot(inv(dot(X.transpose(), X)), X.transpose()), Y)
bk = np.array([bk[0][1], bk[1, 1]], dtype=float)
xs = np.linspace(min(x), max(x),100)
ys = bk[1] * xs + bk[0]

plt.figure(num='Самостоятельная работа №2', figsize=(12.80, 7.20), dpi=100)
plt.title('Самостоятельная работа №2')
plt.plot(xs, ys, '-r', label = 'Регрессия')

plt.xlabel(x_text)
plt.ylabel(y_text)

plt.scatter(x, y, label = 'Данные')
plt.tight_layout()
plt.legend()
plt.show()
out = round(f(bk[0], bk[1]), 3)

t0 = plt.text((plt.xlim()[1] - plt.xlim()[0]) / 2 + plt.xlim()[0], plt.ylim()[1] * 0.975, f'y = {round(bk[1], 3)}*x + {round(bk[0], 3)}', ha='center')
t0.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='gray'))

if out > 0:
	text = f'Вывод: x положительно влияет на y: чтобы увеличить y на 1 един. нужно увеличить x в среднем на {out}'
else:
	text = f'Вывод: x отрицательно влияет на y: чтобы уменьшить y на 1 един. нужно увеличить x в среднем на {out}'

t = plt.text((plt.xlim()[1] - plt.xlim()[0]) / 2 + plt.xlim()[0], plt.ylim()[0] * 1.05, text, ha='center')
t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='gray'))