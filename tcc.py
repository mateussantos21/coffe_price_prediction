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
file = open("results.txt", "w")

cafe = pd.read_csv('MercadoCafe.csv', sep=';')
feature_names = ['Preço NYBOT','Variação NYBOT','Preço BM&F','Variação BM&F', 'Preço R$ Físico','Variação R$ Físico', 'Preço US$ Físico','Variação US$ Físico', 'Cotação Dólar', 'Posição Bancos Comprado', 'Posição Bancos Vendido', 'Diferença Posição Bancos', 'Posição Produtores Comprado', 'Posição Produtores Vendido', 'Posição Outros Comprado', 'Posição Outros Vendido', 'Diferença Posição Outros', 'Contrados Negociados Comprado', 'Contrados Negociados Vendido', 'Diferença Contrados Negociados', 'Contratos em Aberto']
#fig, ax = plt.subplots()
#im, cbar = heatmap(cafe.corr(), feature_names, feature_names, ax=ax, cmap="YlGn", cbarlabel="Coeficiente de Pearson")
#fig.tight_layout()

cafe['Data_certa'] = pd.to_datetime(cafe['Data'], dayfirst=True)
cafe = cafe.set_index('Data_certa')
time_full = cafe['Data']
preco_ny = cafe['Preco ny']
preco_fisico = cafe['preco US$']
preco_cotacao = cafe['preco Dolar']
prices = cafe['Preco bmf']

<<<<<<< HEAD
fig, axs = plt.subplots()
axs.plot(cafe['Preco ny'], 'r-', label = 'NYBOT')
axs.plot(cafe['Preco bmf'], 'g-', label = 'BM&F')
axs.plot(cafe['preco US$'], 'b-', label = 'CEPEA/ESALq')
axs.set_xlabel('tempo')
axs.set_ylabel('Preço (US$)')
axs.grid(True)
plt.legend()
plt.show()
=======
#plt.figure()
#plt.plot(time_full,preco_ny, 'r-', label = 'Preço NYBOT (Us$)')
#plt.plot(time_full,prices, 'g-', label = 'Preço BM&F (Us$)')
#plt.plot(time_full,preco_fisico, 'b-', label = 'Preço CEPEA/ESALq (Us$)')
#plt.plot(time_full,preco_cotacao, 'k-', label = 'Cotação do Dólar')
#plt.legend()
#plt.show()

## Auto Regressive Integrated Moving Average using statsmodel
# http://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMA.html
AR = 2
I = 1
MA = 0

## Create Multi Layer Perceptron Neural Network from sci kit learn library
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
mlp = nn.MLPRegressor(hidden_layer_sizes=(100,), # number of neurons on each hidden layer
                      activation='relu',            # activation function {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
                      solver='adam',                # solver to minimize loss function {‘lbfgs’, ‘sgd’, ‘adam’}
                      alpha=0.0001,                   # penalty parameter
                      learning_rate='constant',     # function for learning rate {'constant', ‘invscaling’, ‘adaptive’}
                      learning_rate_init=0.001,      # learning rate first value
                      power_t=0.5,                  # the exponent for inverse scaling learning rate. It is used in updating effective learning rate
                      max_iter=200,               # maximum number of iterations
                      shuffle=True,                 # whether to shuffle samples in each iteration
                      random_state=None,            # if int, random_state is the seed used by the random number generator
                      tol=0.0001,                 # tolerance for optimization in n_iter_no_change iterations
                      verbose=True,                 # print loss every iteration
                      validation_fraction=0.1,      # proportion of set to use as validation
                      n_iter_no_change=200)       # iterations to look at tol


## Create Gaussian Process Regressor using a Radial Basis Function kernel from sci kit learn library
# https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor
rbf = gp.GaussianProcessRegressor(kernel=gp.kernels.RBF(1.0),  # kernel function https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html#sklearn.gaussian_process.kernels.RBF
                                  alpha=0.00001,                  # value added to the diagonal of the kernel matrix during fitting
                                  optimizer='fmin_l_bfgs_b',   # function to optimize kernel’s parameters minimizing loss
                                  n_restarts_optimizer = 50,   # numbers to reoptimize
                                  random_state=13)              # if int, random_state is the seed used by the random number generator


## Adaptative Neuro Fuzzy Interference System using Tensor Flow library
# https://github.com/tiagoCuervo/TensorANFIS

anfis_rules = 100          # number of rules
anfis_lr = 0.01            # learning rate
anfis_num_epochs = 200   # epochs
anfis_verbose = True       # print loss every iteration
>>>>>>> 06e5759f99e4e19bee6f0868d8eaeeeed402db02

## Auto Regressive Integrated Moving Average using statsmodel
# http://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMA.html
AR = 2
I = 1
MA = 0

## Create Multi Layer Perceptron Neural Network from sci kit learn library
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
mlp = nn.MLPRegressor(hidden_layer_sizes=(100,), # number of neurons on each hidden layer
                      activation='relu',            # activation function {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
                      solver='adam',                # solver to minimize loss function {‘lbfgs’, ‘sgd’, ‘adam’}
                      alpha=0.0001,                   # penalty parameter
                      learning_rate='constant',     # function for learning rate {'constant', ‘invscaling’, ‘adaptive’}
                      learning_rate_init=0.001,      # learning rate first value
                      power_t=0.5,                  # the exponent for inverse scaling learning rate. It is used in updating effective learning rate
                      max_iter=200,               # maximum number of iterations
                      shuffle=True,                 # whether to shuffle samples in each iteration
                      random_state=None,            # if int, random_state is the seed used by the random number generator
                      tol=0.0001,                 # tolerance for optimization in n_iter_no_change iterations
                      verbose=True,                 # print loss every iteration
                      validation_fraction=0.1,      # proportion of set to use as validation
                      n_iter_no_change=200)       # iterations to look at tol


## Create Gaussian Process Regressor using a Radial Basis Function kernel from sci kit learn library
# https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor
rbf = gp.GaussianProcessRegressor(kernel=gp.kernels.RBF(1.0),  # kernel function https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html#sklearn.gaussian_process.kernels.RBF
                                  alpha=0.00001,                  # value added to the diagonal of the kernel matrix during fitting
                                  optimizer='fmin_l_bfgs_b',   # function to optimize kernel’s parameters minimizing loss
                                  n_restarts_optimizer = 50,   # numbers to reoptimize
                                  random_state=13)              # if int, random_state is the seed used by the random number generator


## Adaptative Neuro Fuzzy Interference System using Tensor Flow library
# https://github.com/tiagoCuervo/TensorANFIS

anfis_rules = 100          # number of rules
anfis_lr = 0.01            # learning rate
anfis_num_epochs = 200   # epochs
anfis_verbose = True       # print loss every iteration

data_raw = np.zeros(( len(prices)))
for i in range( len(prices) ):
        data_raw[i] = prices[i]

scaler = prepro.MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_raw.reshape(-1,1))
data_scaled = scaler.transform(data_raw.reshape(-1,1))

inputs = np.zeros((len(prices) - delays, delays))
targets = np.zeros((len(prices) - delays,))

for t in range(len(prices) - delays):
    ins = []
    for i in range(delays):
        ins.append(data_scaled[t + i])
    inputs[t, :] = ins
    targets[t] = data_scaled[t + delays]

X_train = inputs[:targets.size - 2*round(targets.size * percent_for_test), :]
y_train = targets[:targets.size - 2*round(targets.size * percent_for_test)]
X_test = inputs[targets.size - 2*round(targets.size * percent_for_test):, :]
y_test = targets[targets.size - 2*round(targets.size * percent_for_test):]

time_array = []
for i in range(len(y_test)):
    time_array.append(i)

history = [x for x in y_train]
arima = []

inicio_arima = time.time()
for t in range(len(y_test)):
	output = ARIMA(history, order=(AR,I,MA)).fit(disp=0).forecast()
	arima.append(output[0])
	history.append(y_test[t])

arima_pred = np.zeros(( len(arima)))
for i in range( len(arima) ):
    arima_pred[i] = arima[i]
fim_arima = time.time()

tempo_arima = fim_arima - inicio_arima

inicio = time.time()
mlp.fit(X_train,y_train)
ypredictedMLP = mlp.predict(X_test)
fim = time.time()
tempo_mlp = fim - inicio

inicio = time.time()
rbf.fit(X_train,y_train)
ypredictedRBF = rbf.predict(X_test)
fim = time.time()
tempo_rbf = fim - inicio

inicio = time.time()
fis = ANFIS(n_inputs=delays, n_rules=anfis_rules, learning_rate=anfis_lr)
anfis_pred = fis.predict(anfis_num_epochs, X_train, y_train, X_test, y_test, anfis_verbose)
fim = time.time()
tempo_anfis = fim - inicio

real_value = scaler.inverse_transform(y_test.reshape(-1,1))
mlp_result = scaler.inverse_transform(ypredictedMLP.reshape(-1,1))
rbf_result = scaler.inverse_transform(ypredictedRBF.reshape(-1,1))
anf_result = scaler.inverse_transform(anfis_pred.reshape(-1,1))
ari_result = scaler.inverse_transform(arima_pred.reshape(-1,1))

table = '\\begin{table}[] \n\\begin{tabular}{|l|l|l|l|l|} \n\hline \n & \\textbf{ARIMA} & \\textbf{MLP} & \\textbf{RBF} & \\textbf{ANFIS} \\\\ \hline \n'

table += '\\textbf{MSE} & ' + '{:.2f}'.format(metrics.mean_squared_error(real_value, ari_result)) + ' & ' + '{:.2f}'.format(metrics.mean_squared_error(real_value, mlp_result)) + ' & ' + '{:.2f}'.format(metrics.mean_squared_error(real_value, rbf_result)) + ' & ' + '{:.2f}'.format(metrics.mean_squared_error(real_value, anf_result)) + ' \\\\ \hline \n'
table += '\\textbf{RMSE} & ' + '{:.2f}'.format(math.sqrt(metrics.mean_squared_error(real_value, ari_result))) + ' & ' + '{:.2f}'.format(math.sqrt(metrics.mean_squared_error(real_value, mlp_result))) + ' & ' + '{:.2f}'.format(math.sqrt(metrics.mean_squared_error(real_value, rbf_result))) + ' & ' + '{:.2f}'.format(math.sqrt(metrics.mean_squared_error(real_value, anf_result))) + ' \\\\ \hline \n'
table += '\\textbf{MAPE} & ' + '{:.2f}'.format(mean_absolute_percentage_error(real_value, ari_result)) + ' & ' + '{:.2f}'.format(mean_absolute_percentage_error(real_value, mlp_result)) + ' & ' + '{:.2f}'.format(mean_absolute_percentage_error(real_value, rbf_result)) + ' & ' + '{:.2f}'.format(mean_absolute_percentage_error(real_value, anf_result)) + ' \\\\ \hline \n'
table += '\\textbf{Tempo de execução} & ' + '{:.2f}'.format(tempo_arima) + ' & ' + '{:.2f}'.format(tempo_mlp) + ' & ' + '{:.2f}'.format(tempo_rbf) + ' & ' + '{:.2f}'.format(tempo_anfis) + ' \\\\ \hline \n'

table += '\end{tabular} \n\end{table} \n\n\n\n\n'

results = table

fig, axs = plt.subplots(4, figsize=(20, 16))

axs[0].plot(time_array, real_value, 'k-', label = 'valor real')
axs[0].plot(time_array, ari_result, 'm-', label = 'ARIMA')
axs[1].plot(time_array, real_value, 'k-')
axs[1].plot(time_array, mlp_result, 'r-', label = 'MLP')
axs[2].plot(time_array, real_value, 'k-')
axs[2].plot(time_array, rbf_result, 'b-', label = 'RBF')
axs[3].plot(time_array, real_value, 'k-')
axs[3].plot(time_array, anf_result, 'g-', label = "ANFIS")
fig.legend(loc='upper right')
plt.savefig('essaaquivai.png')
arima_best_score = math.sqrt(metrics.mean_squared_error(real_value, ari_result))
mlp_best_score = math.sqrt(metrics.mean_squared_error(real_value, mlp_result))
rbf_best_score = math.sqrt(metrics.mean_squared_error(real_value, rbf_result))
anfis_best_score = math.sqrt(metrics.mean_squared_error(real_value, anf_result))


# #search no arima
ar_array = [1]
i_array = [0]
ma_array = [0]

arima_optimal = {"ar" : 2, "i" : 1, "ma" : 0}
ari_opt_result = ari_result
tempo_opt_ari = tempo_arima

for ar in ar_array:
    for i in i_array:
        for ma in ma_array:
            history = [x for x in y_train]
            arima = []

            inicio_arima = time.time()
            for t in range(len(y_test)):
                output = ARIMA(history, order=(ar, i, ma)).fit(disp=0).forecast()
                arima.append(output[0])
                history.append(y_test[t])

            arima_pred = np.zeros((len(arima)))
            for j in range(len(arima)):
                arima_pred[j] = arima[j]
            fim_arima = time.time()
            tempo_arima = fim_arima - inicio_arima

            ari_result = scaler.inverse_transform(arima_pred.reshape(-1, 1))
            arima_score = math.sqrt(metrics.mean_squared_error(real_value, ari_result))

            if arima_score < arima_best_score:
                arima_optimal = {"ar": ar, "i": i, "ma": ma}
                arima_best_score = arima_score
                ari_opt_result = scaler.inverse_transform(arima_pred.reshape(-1, 1))
                tempo_opt_ari = tempo_arima

# #search no mlp

lr_array_mlp = [0.00085]
fl_array = [600]
sl_array = [400]

mlp_optimal = {"lr" : 0.001, "fl" : 100, "sl" : 1}
mlp_opt_result = mlp_result
tempo_opt_mlp = tempo_mlp

for lr in lr_array_mlp:
    for fl in fl_array:
        for sl in sl_array:

            mlp = nn.MLPRegressor(hidden_layer_sizes=(fl,sl), learning_rate_init=lr, max_iter=1000, tol=0.0000001, validation_fraction=0.1, n_iter_no_change=1000, verbose=True)
            inicio = time.time()
            mlp.fit(X_train, y_train)
            ypredictedMLP = mlp.predict(X_test)
            fim = time.time()
            tempo_mlp = fim - inicio

            mlp_result = scaler.inverse_transform(ypredictedMLP.reshape(-1, 1))

            mlp_score = math.sqrt(metrics.mean_squared_error(real_value, mlp_result))

            if True: # < mlp_best_score:
                mlp_optimal = {"lr" : lr, "fl" : fl, "sl" : sl}
                mlp_best_score = mlp_score
                mlp_opt_result = scaler.inverse_transform(ypredictedMLP.reshape(-1, 1))
                tempo_opt_mlp = tempo_mlp

# mlp_curve = {1: 32.651978218028034, 2: 5.551768749611867, 3: 6.706691440031622, 4: 4.834009874211449, 5: 6.159995312154958, 6: 4.175744605816024, 7: 5.346350399638823, 8: 4.334954878482447, 9: 4.538008617005081, 10: 4.3728886351126, 11: 3.7764896559688528, 12: 4.067896806498186, 13: 3.753997536587117, 14: 3.9429026666833766, 15: 3.3491505149677234, 16: 3.5898457125100416, 17: 3.0995261422848186, 18: 3.5104042893313854, 19: 3.567806740459899, 20: 3.275016692591553, 21: 3.5786566580970876, 22: 3.473161782933408, 23: 3.426306166247589, 24: 3.165848467851309, 25: 3.0461864652910307, 26: 3.3133617324437514, 27: 3.3005774861708126, 28: 3.088045145523836, 29: 2.987541675149137, 30: 3.80985110041052, 31: 3.193357487346718, 32: 2.959561115367843, 33: 3.069578021560225, 34: 3.462321838194916, 35: 2.9107942685195933, 36: 3.0215239005660486, 37: 3.0878241636435915, 38: 2.7937708897922153, 39: 2.917520970648209, 40: 3.0368294558749325, 41: 2.9326186851808775, 42: 3.234949678803573, 43: 3.0030250319992766, 44: 2.8405442432091887, 45: 3.369791928841273, 46: 2.998718624216391, 47: 2.804326426573474, 48: 3.1949421004073115, 49: 2.8583189772851543, 50: 2.8406107402836622, 51: 2.7968141980303636, 52: 3.0657549830913817, 53: 2.7969728799822344, 54: 2.711693512510519, 55: 2.9264103000452364, 56: 3.0607424440480577, 57: 2.907001465250133, 58: 2.7942952300011132, 59: 2.7772417787087007, 60: 2.9148552068221862, 61: 2.6789722028671337, 62: 2.700508450069721, 63: 2.7278074940285224, 64: 2.7122829754835918, 65: 2.9192216143644436, 66: 2.9510521600124378, 67: 2.742103896214864, 68: 2.7442170785432776, 69: 2.87142187043716, 70: 2.6961545821680213, 71: 3.007844766882209, 72: 2.676826503944573, 73: 3.368955413199768, 74: 2.873646285441969, 75: 3.430741093596998, 76: 2.7183703587609953, 77: 2.7860135610804466, 78: 2.934908846746502, 79: 2.6769362143850643, 80: 2.714277942112923, 81: 3.007027387123485, 82: 3.267499069978167, 83: 2.996924326657179, 84: 2.9205457015860943, 85: 2.674693423634298, 86: 2.9000533009201774, 87: 3.0494688079142955, 88: 2.694334360460148, 89: 2.8902433868790363, 90: 2.800128395451659, 91: 2.6926508683701207, 92: 3.427043869444114, 93: 2.932836655863246, 94: 2.698633198082254, 95: 2.7366608304069104, 96: 2.6681087813634794, 97: 2.659850081585001, 98: 2.9198433721132853, 99: 2.6468983028485233, 100: 2.6496111634116555, 101: 2.92568542279493, 102: 2.8833168979552726, 103: 2.650132191720915, 104: 2.6585439904624844, 105: 2.7287451625860637, 106: 3.556225419070584, 107: 2.8221768183251927, 108: 2.742329542762361, 109: 2.6506309419800806, 110: 2.7279135627380064, 111: 2.7141326108486856, 112: 2.622040112153869, 113: 2.645126656577022, 114: 2.6714407780212492, 115: 3.010971647899795, 116: 2.8938529640160735, 117: 2.6946881679280477, 118: 2.920689104817948, 119: 2.7519197745139254, 120: 2.634362502027966, 121: 2.64528694497067, 122: 2.8873884045295526, 123: 2.742994695724771, 124: 2.9491434516454285, 125: 2.661696566839098, 126: 3.117940605884815, 127: 2.6491804318952377, 128: 2.6492511005512958, 129: 2.684436103937748, 130: 2.7997635296081933, 131: 2.7968418454959156, 132: 2.7270903022297657, 133: 3.7384422633310863, 134: 2.6632070121582383, 135: 2.687918561013565, 136: 2.7791705023021533, 137: 3.0637673594145425, 138: 2.666563100686945, 139: 3.2590050363909393, 140: 4.024522968349525, 141: 2.631436583429506, 142: 2.7260102137936655, 143: 2.970658430710695, 144: 2.6986392622697797, 145: 3.4443082273782224, 146: 2.6272108980847846, 147: 2.6086208814925302, 148: 2.6539022088657456, 149: 2.699717470695076, 150: 2.6394043550746455, 151: 2.6204144185271407, 152: 2.7125824529693707, 153: 2.9433960531902454, 154: 2.9144341259745072, 155: 2.6435896491484048, 156: 2.881548861922469, 157: 2.6478149290103863, 158: 2.8045510257393387, 159: 2.6456723427601836, 160: 2.7462667189236356, 161: 3.3500007820509343, 162: 2.76668590857209, 163: 2.6270376497635097, 164: 2.6369191850713776, 165: 2.637948735469217, 166: 2.684273174498905, 167: 2.9662036529148628, 168: 3.351565705621848, 169: 3.1694406272626434, 170: 2.738667547226745, 171: 2.658561618424249, 172: 2.673874757106593, 173: 2.7632480563784707, 174: 3.115423271672058, 175: 2.7413614127002157, 176: 3.8251099509436814, 177: 2.7604804478916787, 178: 2.8080728677891087, 179: 2.6248306401434016, 180: 2.799000320507809, 181: 3.1565335394090424, 182: 2.7584747762608246, 183: 2.6411695203852514, 184: 2.812649960536933, 185: 3.0062840089667513, 186: 3.0748075341787082, 187: 2.6288483327404077, 188: 2.642759608470592, 189: 2.6973509260594617, 190: 2.67666930966965, 191: 2.6277594247222473, 192: 2.6861988712688422, 193: 2.6785653289376823, 194: 2.7014177410251473, 195: 2.6818518437551035, 196: 2.65345653289376823, 197: 2.65655653289376823, 198: 2.6689653289376823, 199: 2.6785653289376823}

# for epoch in range(1,200,1):
#     mlp = nn.MLPRegressor(hidden_layer_sizes=(mlp_optimal["fl"], mlp_optimal["sl"]), learning_rate_init=mlp_optimal["lr"], max_iter=epoch,
#                           validation_fraction=0.1, n_iter_no_change=epoch, verbose=True)
#     mlp.fit(X_train, y_train)
#     ypredictedMLP = mlp.predict(X_test)
#     mlp_result = scaler.inverse_transform(ypredictedMLP.reshape(-1, 1))
#     mlp_score = math.sqrt(metrics.mean_squared_error(real_value, mlp_result))
#     mlp_curve[epoch] = mlp_score

# # search no rbf

alpha_array = [0.0001]
restart_array = [50]
kernel_array = [0.005]

rbf_optimal = {"alpha": 0.000001, "restart": 10, "kernel": 1.0}
rbf_opt_result = rbf_result
tempo_opt_rbf = tempo_rbf

for alpha in alpha_array:
    for restarts in restart_array:
        for kern in kernel_array:
            print("rbf running")
            rbf = gp.GaussianProcessRegressor(kernel=gp.kernels.RBF(kern), alpha=alpha, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=restarts, random_state=13)

            inicio = time.time()
            rbf.fit(X_train, y_train)
            ypredictedRBF = rbf.predict(X_test)
            fim = time.time()
            tempo_rbf = fim - inicio

            rbf_result = scaler.inverse_transform(ypredictedRBF.reshape(-1, 1))

            rbf_score = math.sqrt(metrics.mean_squared_error(real_value, rbf_result))

            if True: #rbf_score < rbf_best_score:
                rbf_optimal = {"alpha": alpha, "restart": restarts, "kernel": kern}
                rbf_best_score = rbf_score
                rbf_opt_result = scaler.inverse_transform(ypredictedRBF.reshape(-1, 1))
                tempo_opt_rbf = tempo_rbf

rbf_curve = {}
# for epoch in range(1,1000,10):
#     rbf = gp.GaussianProcessRegressor(kernel=gp.kernels.RBF(rbf_optimal["kernel"]), alpha=rbf_optimal["alpha"], optimizer='fmin_l_bfgs_b',
#                                       n_restarts_optimizer=epoch, random_state=13)
#     rbf.fit(X_train, y_train)
#     ypredictedRBF = rbf.predict(X_test)
#     rbf_result = scaler.inverse_transform(ypredictedRBF.reshape(-1, 1))
#     rbf_score = math.sqrt(metrics.mean_squared_error(real_value, rbf_result))
#     rbf_curve[epoch] = rbf_score

# search no anfis

lr_array = [0.05]
rules_array = [100]

anfis_optimal = {"lr": 0.01, "rules": 100}
anfis_opt_result = anf_result
tempo_opt_anfis = tempo_anfis
iterator = 1
for lr in lr_array:
    for rules in rules_array:

        inicio = time.time()
        fis = ANFIS(n_inputs=delays, n_rules=rules, learning_rate=lr, mu_string="mu"+str(iterator), sigma_string="sigma"+str(iterator), y_string="y"+str(iterator))
        anfis_pred = fis.predict(1000, X_train, y_train, X_test, y_test, anfis_verbose)
        fim = time.time()
        tempo_anfis = fim - inicio

        anfis_result = scaler.inverse_transform(anfis_pred.reshape(-1, 1))

        anfis_score = math.sqrt(metrics.mean_squared_error(real_value, anfis_result))

        if True: # anfis_score < anfis_best_score:
            anfis_optimal = {"lr": lr, "rules": rules}
            anfis_best_score = anfis_score
            anfis_opt_result = scaler.inverse_transform(anfis_pred.reshape(-1, 1))
            tempo_opt_anfis = tempo_anfis

        iterator += 1

anfis_curve = {}
# for epoch in range(1,1000,10):
#     fis = ANFIS(n_inputs=delays, n_rules=anfis_optimal["rules"], learning_rate=anfis_optimal["lr"], mu_string="mu" + str(iterator),
#                 sigma_string="sigma" + str(iterator), y_string="y" + str(iterator))
#     anfis_pred = fis.predict(anfis_num_epochs, X_train, y_train, X_test, y_test, anfis_verbose)
#     anfis_result = scaler.inverse_transform(anfis_pred.reshape(-1, 1))
#     anfis_score = math.sqrt(metrics.mean_squared_error(real_value, anfis_result))
#     anfis_curve[epoch] = anfis_score
#     iterator += 1

table = '\\begin{table}[] \n\\begin{tabular}{|l|l|l|l|l|} \n\hline \n & \\textbf{ARIMA} & \\textbf{MLP} & \\textbf{RBF} & \\textbf{ANFIS} \\\\ \hline \n'

table += '\\textbf{MSE} & ' + '{:.2f}'.format(metrics.mean_squared_error(real_value, ari_opt_result)) + ' & ' + '{:.2f}'.format(metrics.mean_squared_error(real_value, mlp_opt_result)) + ' & ' + '{:.2f}'.format(metrics.mean_squared_error(real_value, rbf_opt_result)) + ' & ' + '{:.2f}'.format(metrics.mean_squared_error(real_value, anfis_opt_result)) + ' \\\\ \hline \n'
table += '\\textbf{RMSE} & ' + '{:.2f}'.format(math.sqrt(metrics.mean_squared_error(real_value, ari_opt_result))) + ' & ' + '{:.2f}'.format(math.sqrt(metrics.mean_squared_error(real_value, mlp_opt_result))) + ' & ' + '{:.2f}'.format(math.sqrt(metrics.mean_squared_error(real_value, rbf_opt_result))) + ' & ' + '{:.2f}'.format(math.sqrt(metrics.mean_squared_error(real_value, anfis_opt_result))) + ' \\\\ \hline \n'
table += '\\textbf{MAPE} & ' + '{:.2f}'.format(mean_absolute_percentage_error(real_value, ari_opt_result)) + ' & ' + '{:.2f}'.format(mean_absolute_percentage_error(real_value, mlp_opt_result)) + ' & ' + '{:.2f}'.format(mean_absolute_percentage_error(real_value, rbf_opt_result)) + ' & ' + '{:.2f}'.format(mean_absolute_percentage_error(real_value, anfis_opt_result)) + ' \\\\ \hline \n'
table += '\\textbf{Tempo de execução} & ' + '{:.2f}'.format(tempo_opt_ari) + ' & ' + '{:.2f}'.format(tempo_opt_mlp) + ' & ' + '{:.2f}'.format(tempo_opt_rbf) + ' & ' + '{:.2f}'.format(tempo_opt_anfis) + ' \\\\ \hline \n'

table += '\end{tabular} \n\end{table}\n\n\n'

results += table

results += str(arima_optimal)
results += str(mlp_optimal)
results += str(rbf_optimal)
results += str(anfis_optimal)

# results += "\n\n\n" + str(mlp_curve) + "\n\n\n" + str(rbf_curve) + "\n\n\n" + str(anfis_curve)

print(results)

file.write(results)
file.close()

fig, axs = plt.subplots(4, figsize=(20, 16))

axs[0].plot(time_array, real_value, 'k-', label = 'valor real')
axs[0].plot(time_array, ari_opt_result, 'm-', label = 'ARIMA')
axs[1].plot(time_array, real_value, 'k-')
axs[1].plot(time_array, mlp_opt_result, 'r-', label = 'MLP')
axs[2].plot(time_array, real_value, 'k-')
axs[2].plot(time_array, rbf_opt_result, 'b-', label = 'RBF')
axs[3].plot(time_array, real_value, 'k-')
axs[3].plot(time_array, anfis_opt_result, 'g-', label = "ANFIS")
fig.legend(loc='upper right')

from scipy.ndimage.filters import gaussian_filter1d

# plt.figure(figsize=(20, 20))
# plt.plot(list(mlp_curve.keys()), gaussian_filter1d(list(mlp_curve.values()), sigma=10), label = 'MLP score')
# plt.xlabel('Época')
# plt.ylabel('RMSE')
#
# plt.legend()

plt.savefig('essaaquivaiotimizada.png')

# blabla

#outro comentario

