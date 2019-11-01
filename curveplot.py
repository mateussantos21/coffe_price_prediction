# -*- coding: utf-8 -*-
'''
Código produzido para experimentações de Trabalho de Conclusão do Curso de Engenharia de Controle e Automação
da Universidade Federal de Lavras (UFLA)

Título do Trabalho:
Redes Neurais aplicadas à Previsão do Preço de Commodities: Estudo sobre o Café Arabica Brasileiro

Autor:
Mateus Rodrigues Santos

Orientador:
Daniel Furtado Leite
'''

from requirements import *
warnings.filterwarnings("ignore")


# cafe = pd.read_csv('MercadoCafe.csv', sep=';')
# feature_names = ['Preço NYBOT','Variação NYBOT','Preço BM&F','Variação BM&F', 'Preço R$ Físico','Variação R$ Físico', 'Preço US$ Físico','Variação US$ Físico', 'Cotação Dólar', 'Posição Bancos Comprado', 'Posição Bancos Vendido', 'Diferença Posição Bancos', 'Posição Produtores Comprado', 'Posição Produtores Vendido', 'Posição Outros Comprado', 'Posição Outros Vendido', 'Diferença Posição Outros', 'Contrados Negociados Comprado', 'Contrados Negociados Vendido', 'Diferença Contrados Negociados', 'Contratos em Aberto']
# #fig, ax = plt.subplots()
# #im, cbar = heatmap(cafe.corr(), feature_names, feature_names, ax=ax, cmap="YlGn", cbarlabel="Coeficiente de Pearson")
# #fig.tight_layout()
#
# time_full = cafe['Data']
# preco_ny = cafe['Preco ny']
# preco_fisico = cafe['preco US$']
# preco_cotacao = cafe['preco Dolar']
# prices = cafe['Preco bmf']
#
# #plt.figure()
# #plt.plot(time_full,preco_ny, 'r-', label = 'Preço NYBOT (Us$)')
# #plt.plot(time_full,prices, 'g-', label = 'Preço BM&F (Us$)')
# #plt.plot(time_full,preco_fisico, 'b-', label = 'Preço CEPEA/ESALq (Us$)')
# #plt.plot(time_full,preco_cotacao, 'k-', label = 'Cotação do Dólar')
# #plt.legend()
# #plt.show()
#
# ## Auto Regressive Integrated Moving Average using statsmodel
# # http://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMA.html
# AR = 2
# I = 1
# MA = 0
#
# ## Create Multi Layer Perceptron Neural Network from sci kit learn library
# # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
# mlp = nn.MLPRegressor(hidden_layer_sizes=(100,), # number of neurons on each hidden layer
#                       activation='relu',            # activation function {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
#                       solver='adam',                # solver to minimize loss function {‘lbfgs’, ‘sgd’, ‘adam’}
#                       alpha=0.0001,                   # penalty parameter
#                       learning_rate='constant',     # function for learning rate {'constant', ‘invscaling’, ‘adaptive’}
#                       learning_rate_init=0.001,      # learning rate first value
#                       power_t=0.5,                  # the exponent for inverse scaling learning rate. It is used in updating effective learning rate
#                       max_iter=200,               # maximum number of iterations
#                       shuffle=True,                 # whether to shuffle samples in each iteration
#                       random_state=None,            # if int, random_state is the seed used by the random number generator
#                       tol=0.0001,                 # tolerance for optimization in n_iter_no_change iterations
#                       verbose=True,                 # print loss every iteration
#                       validation_fraction=0.1,      # proportion of set to use as validation
#                       n_iter_no_change=200)       # iterations to look at tol
#
#
# ## Create Gaussian Process Regressor using a Radial Basis Function kernel from sci kit learn library
# # https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor
# rbf = gp.GaussianProcessRegressor(kernel=gp.kernels.RBF(1.0),  # kernel function https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html#sklearn.gaussian_process.kernels.RBF
#                                   alpha=0.00001,                  # value added to the diagonal of the kernel matrix during fitting
#                                   optimizer='fmin_l_bfgs_b',   # function to optimize kernel’s parameters minimizing loss
#                                   n_restarts_optimizer = 200,   # numbers to reoptimize
#                                   random_state=13)              # if int, random_state is the seed used by the random number generator
#
#
# ## Adaptative Neuro Fuzzy Interference System using Tensor Flow library
# # https://github.com/tiagoCuervo/TensorANFIS
#
# anfis_rules = 100          # number of rules
# anfis_lr = 0.01            # learning rate
# anfis_num_epochs = 200   # epochs
# anfis_verbose = True       # print loss every iteration
#
# data_raw = np.zeros(( len(prices)))
# for i in range( len(prices) ):
#         data_raw[i] = prices[i]
#
# scaler = prepro.MinMaxScaler(feature_range=(0, 1))
# scaler.fit(data_raw.reshape(-1,1))
# data_scaled = scaler.transform(data_raw.reshape(-1,1))
#
# inputs = np.zeros((len(prices) - delays, delays))
# targets = np.zeros((len(prices) - delays,))
#
# for t in range(len(prices) - delays):
#     ins = []
#     for i in range(delays):
#         ins.append(data_scaled[t + i])
#     inputs[t, :] = ins
#     targets[t] = data_scaled[t + delays]
#
# X_train = inputs[:targets.size - 2*round(targets.size * percent_for_test), :]
# y_train = targets[:targets.size - 2*round(targets.size * percent_for_test)]
# X_test = inputs[targets.size - 2*round(targets.size * percent_for_test):, :]
# y_test = targets[targets.size - 2*round(targets.size * percent_for_test):]
# real_value = scaler.inverse_transform(y_test.reshape(-1,1))
# time_array = []
# for i in range(len(y_test)):
#     time_array.append(i)


mlp_curve = {}

# for epoch in range(1,1000,1):
#     mlp = nn.MLPRegressor(hidden_layer_sizes=(100,), learning_rate_init=0.001, max_iter=epoch,
#                           validation_fraction=0.1, n_iter_no_change=epoch, verbose=True)
#     mlp.fit(X_train, y_train)
#     ypredictedMLP = mlp.predict(X_test)
#     mlp_result = scaler.inverse_transform(ypredictedMLP.reshape(-1, 1))
#     mlp_score = math.sqrt(metrics.mean_squared_error(real_value, mlp_result))
#     mlp_curve[epoch] = mlp_score

# anfis_curve = {}
# iterator = 0
# for epoch in range(1,1000,1):
#     fis = ANFIS(n_inputs=delays, n_rules=20, learning_rate=0.01, mu_string="mu" + str(iterator),
#                 sigma_string="sigma" + str(iterator), y_string="y" + str(iterator))
#     anfis_pred = fis.predict(epoch, X_train, y_train, X_test, y_test, anfis_verbose)
#     anfis_result = scaler.inverse_transform(anfis_pred.reshape(-1, 1))
#     anfis_score = math.sqrt(metrics.mean_squared_error(real_value, anfis_result))
#     anfis_curve[epoch] = anfis_score
#     iterator += 1


from scipy.ndimage.filters import gaussian_filter1d

plt.figure(figsize=(20, 14))
plt.plot(list(mlp_curve.keys()), gaussian_filter1d(list(mlp_curve.values()), sigma=10), label = 'MLP score')
plt.xlabel('Época')
plt.ylabel('RMSE')
plt.legend()

# plt.figure(figsize=(20, 14))
# plt.plot(list(anfis_curve.keys()), gaussian_filter1d(list(anfis_curve.values()), sigma=10), label = 'ANFIS score')
# plt.xlabel('Época')
# plt.ylabel('RMSE')
# plt.legend()

plt.show()
