# Projeto

Está é uma implementação de uma Rede Neural Artificial Perceptron (arquitetura feedforwards de camadas simples com um único neurônio de saída, sem realimentação). Nesta implementação, utiliza-se a função de ativação Degrau Bipolar. Os pesos sinápitcos e o limiar são gerados pseudo-aleatoriamente.

## Como rodar

1. Crie o ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Instale as dependências:
    pip install -r requirements.txt

3. Execute:
    python -m src.main ou

    no Linux: 
    1. chmod +x scripts/run_perceptron.sh
    2. ./scripts/run_perceptron.sh
