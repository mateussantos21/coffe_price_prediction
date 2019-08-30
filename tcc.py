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


cafe = pd.read_csv('MercadoCafe.csv', sep=';')
feature_names = ['Preço NYBOT','Variação NYBOT','Preço BM&F','Variação BM&F', 'Preço R$ Físico','Variação R$ Físico', 'Preço US$ Físico','Variação US$ Físico', 'Cotação Dólar', 'Posição Bancos Comprado', 'Posição Bancos Vendido', 'Diferença Posição Bancos', 'Posição Produtores Comprado', 'Posição Produtores Vendido', 'Posição Outros Comprado', 'Posição Outros Vendido', 'Diferença Posição Outros', 'Contrados Negociados Comprado', 'Contrados Negociados Vendido', 'Diferença Contrados Negociados', 'Contratos em Aberto']
fig, ax = plt.subplots()
im, cbar = heatmap(cafe.corr(), feature_names, feature_names, ax=ax, cmap="YlGn", cbarlabel="Coeficiente de Pearson")
fig.tight_layout()

time_full = cafe['Data']
preco_ny = cafe['Preco ny']
preco_fisico = cafe['preco US$']
preco_cotacao = cafe['preco Dolar']
prices = cafe['Preco bmf']

plt.figure()
plt.plot(time_full,preco_ny, 'r-', label = 'Preço NYBOT (Us$)')
plt.plot(time_full,prices, 'g-', label = 'Preço BM&F (Us$)')
plt.plot(time_full,preco_fisico, 'b-', label = 'Preço CEPEA/ESALq (Us$)')
#plt.plot(time_full,preco_cotacao, 'k-', label = 'Cotação do Dólar')
plt.legend()
plt.show()

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

X_train = inputs[:targets.size - round(targets.size * percent_for_test), :]
y_train = targets[:targets.size - round(targets.size * percent_for_test)]
X_test = inputs[targets.size - round(targets.size * percent_for_test):, :]
y_test = targets[targets.size - round(targets.size * percent_for_test):]

time = []
for i in range(len(y_test)):
    time.append(i)


history = [x for x in y_train]
arima = []

for t in range(len(y_test)):
	output = ARIMA(history, order=(AR,I,MA)).fit(disp=0).forecast()
	arima.append(output[0])
	history.append(y_test[t])

arima_pred = np.zeros(( len(arima)))
for i in range( len(arima) ):
        arima_pred[i] = arima[i]


mlp.fit(X_train,y_train)
ypredictedMLP = mlp.predict(X_test)

rbf.fit(X_train,y_train)
ypredictedRBF = rbf.predict(X_test)

fis = ANFIS(n_inputs=delays, n_rules=anfis_rules, learning_rate=anfis_lr)
anfis_pred = fis.predict(anfis_num_epochs, X_train, y_train, X_test, y_test, anfis_verbose)

real_value = scaler.inverse_transform(y_test.reshape(-1,1))
mlp_result = scaler.inverse_transform(ypredictedMLP.reshape(-1,1))
rbf_result = scaler.inverse_transform(ypredictedRBF.reshape(-1,1))
anf_result = scaler.inverse_transform(anfis_pred.reshape(-1,1))
ari_result = scaler.inverse_transform(arima_pred.reshape(-1,1))

table = '\\begin{table}[] \n\\begin{tabular}{|l|l|l|l|l|} \n\hline \n & \\textbf{ARIMA} & \\textbf{MLP} & \\textbf{RBF} & \\textbf{ANFIS} \\\\ \hline \n'

table += '\\textbf{MSE} & ' + '{:.2f}'.format(metrics.mean_squared_error(real_value, ari_result)) + ' & ' + '{:.2f}'.format(metrics.mean_squared_error(real_value, mlp_result)) + ' & ' + '{:.2f}'.format(metrics.mean_squared_error(real_value, rbf_result)) + ' & ' + '{:.2f}'.format(metrics.mean_squared_error(real_value, anf_result)) + ' \\\\ \hline \n'
table += '\\textbf{RMSE} & ' + '{:.2f}'.format(math.sqrt(metrics.mean_squared_error(real_value, ari_result))) + ' & ' + '{:.2f}'.format(math.sqrt(metrics.mean_squared_error(real_value, mlp_result))) + ' & ' + '{:.2f}'.format(math.sqrt(metrics.mean_squared_error(real_value, rbf_result))) + ' & ' + '{:.2f}'.format(math.sqrt(metrics.mean_squared_error(real_value, anf_result))) + ' \\\\ \hline \n'
table += '\\textbf{MAPE} & ' + '{:.2f}'.format(mean_absolute_percentage_error(real_value, ari_result)) + ' & ' + '{:.2f}'.format(mean_absolute_percentage_error(real_value, mlp_result)) + ' & ' + '{:.2f}'.format(mean_absolute_percentage_error(real_value, rbf_result)) + ' & ' + '{:.2f}'.format(mean_absolute_percentage_error(real_value, anf_result)) + ' \\\\ \hline \n'

table += '\end{tabular} \n\end{table}'

print(table)

plt.figure()
plt.plot(time,real_value, 'k-', label = 'valor real')
plt.plot(time,ari_result, 'p-', label = 'ARIMA')
plt.plot(time,mlp_result, 'r-', label = 'MLP')
plt.plot(time,rbf_result, 'b-', label = 'RBF')
plt.plot(time,anf_result, 'g-', label = "ANFIS")
plt.legend(loc='upper right')
plt.show()