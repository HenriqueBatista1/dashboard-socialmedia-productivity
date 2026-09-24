# 📊 Dashboard: Mídias Sociais, Dopamina e Produtividade

Este projeto consiste em um dashboard desenvolvido em Python para a **Atividade 1** da disciplina de Visualização de Dados. A solução analisa a relação entre hábitos digitais, busca por recompensas (*dopamina*) e os impactos perceptivos na produtividade e foco dos usuários.

---

## 🎯 Definição do Projeto

* **Público-Alvo:** Estudantes universitários e educadores.
* **Demanda de Informação:** Compreender como hábitos digitais e o tempo de tela influenciam a atenção, o sono e a eficiência nas tarefas diárias.
* **Pergunta Central (Destaque Visual):** *Existe associação entre o tempo de uso diário de redes sociais e a queda na taxa de produtividade autorrelatada?*

---

## 🌐 Deploy

Acesse o dashboard publicado no Streamlit:

https://digital-habit-tracker.streamlit.app/

### Fonte dos dados

O dataset utilizado está disponível no Kaggle:

https://www.kaggle.com/datasets/manaswinsripatnala/social-media-dopamine-and-productivity-dataset/data

---

## 🚀 Passo a Passo para Instalação e Execução

### 1. Clonar o Repositório
```bash
git clone https://github.com/HenriqueBatista1/dashboard-socialmedia-productivity.git
cd dashboard-socialmedia-productivity
```

### 2. Criar um Ambiente Virtual
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar as Dependências
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Executar o Dashboard
```bash
streamlit run app.py
```

### 5. Executar os testes

```bash
python -m pytest tests/ -q
```

Os testes unitários garantem a integridade do tratamento de dados, a ordenação correta das categorias e a consistência da estrutura dos gráficos gerados pelo dashboard.