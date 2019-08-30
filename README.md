# Redes Neurais aplicadas à Previsão do Preço de Commodities: Estudo sobre o Café Arabica Brasileiro

Código produzido para experimentações de Trabalho de Conclusão do Curso de Engenharia de Controle e Automação
da Universidade Federal de Lavras (UFLA)

Autor:
Mateus Rodrigues Santos

Orientador:
Daniel Furtado Leite

## Requisitos

Python 3, e suas bibliotecas listadas em *requirements.txt*.

## Rodando o experimento

Clonar o repositório e rodar o seguinte comando:

```
pip install -r requirements.txt
```

Rodar o programa:

```
python tcc.py
```

Importante ressaltar que o código ainda está em desenvolvimento, sendo assim a cada rodada do programa será treinada uma nova rede, e não utilizada a melhor rede obtida. Este é o objetivo da experimentação.

## To Do:

Lista de afazeres nos próximos passos:

* Passar toda parametrização para o arquivo config.json;
* Encontrar parametrização final e gerar objeto das redes treinadas;
* Implementar grid search para parametrização automatizada;
* Implementar utilização das demais features presentes no banco de dados nos modelos;