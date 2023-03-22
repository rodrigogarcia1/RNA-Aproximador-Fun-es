import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MaxAbsScaler

print('Carregando Arquivo de teste')
arquivo = np.load('teste5.npy')
x = arquivo[0]
y = np.ravel(arquivo[1])

scale= MaxAbsScaler().fit(arquivo[1])
arquivo = np.ravel(scale.transform(arquivo[1]))
"""
regr = MLPRegressor(hidden_layer_sizes=(600,500),
                    max_iter=7000,
                    activation='relu', #{'identity', 'logistic', 'tanh', 'relu'},
                    solver='adam',
                    learning_rate = 'adaptive',
                    n_iter_no_change=50)"""


"""" Teste2
regr = MLPRegressor(hidden_layer_sizes=(2),
                    max_iter=10000,
                    activation='relu', #{'identity', 'logistic', 'tanh', 'relu'},
                    solver='adam',
                    learning_rate = 'adaptive',
                    n_iter_no_change=50)"""

"""Teste3
regr = MLPRegressor(hidden_layer_sizes=(10,10),
                    max_iter=10000,
                    activation='relu', #{'identity', 'logistic', 'tanh', 'relu'},
                    solver='adam',
                    learning_rate = 'adaptive',
                    n_iter_no_change=50)"""

regr = MLPRegressor(hidden_layer_sizes=(32,32),
                    max_iter=30000,
                    activation='logistic', #{'identity', 'logistic', 'tanh', 'relu'},
                    solver='adam',
                    learning_rate = 'adaptive',
                    n_iter_no_change=30000)


print('Treinando RNA')
regr = regr.fit(x,y)



print('Preditor')
y_est = regr.predict(x)
print(regr.best_loss_)


plt.figure(figsize=[14,7])

#plot curso original
plt.subplot(1,3,1)
plt.plot(x,y)

#plot aprendizagem
plt.subplot(1,3,2)
plt.plot(regr.loss_curve_)

#plot regressor
plt.subplot(1,3,3)
plt.plot(x,y,linewidth=1,color='yellow')
plt.plot(x,y_est,linewidth=2)




plt.show()