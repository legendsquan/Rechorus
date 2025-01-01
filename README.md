# Rechorus
Step 1.
1. Install [Anaconda](https://docs.conda.io/en/latest/miniconda.html) with Python >= 3.10
2. Clone the repository

```bash
git clone https://github.com/THUwangcy/ReChorus.git
```

3. Install requirements and step into the `src` folder

```bash
cd ReChorus
pip install -r requirements.txt
cd src
```
Step 2.
1.Open a new folder named my_models

2.Put all python codes into the model
```bash
cp -r ...
```
3.copy Boost_main.py under main_py

4. Run model with the build-in dataset

```bash
python main.py --model_name Boost--emb_size 64 --lr 1e-3 --l2 1e-6 --dataset 'Grocery_and_Gourmet_Food'
```
