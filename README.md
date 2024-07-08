## NullClass Task 3

This task is about creating a feature to translate the language with a combination of two languages at the same time . We should be able to convert the 2 different languages at the same time . translate `English` to `French` and `Hindi` at the same time . This model should work only for `10` letter English words . If we enter below 10 letters or above `10` letters it should not work. The model files are saved in folder `english_to_french_lstm_model` and `english_to_hindi_model`. The tokenizers are saved in `english_tokenizer.json`, `english_tokenizer_hindi.json`, `frencg_tokenizer.json` and `hindi_tokenizer.json`.

In order to run the notebook, follow the steps:

1. Create a conda environment

```bash
conda create --name nullclass python=3.9
```
2. Activate the environment

```bash
conda activate nullclass
```
3. Install cudnn plugin
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

4. Install tensorflow
```bash
pip install --upgrade pip
# Anything above 2.10 is not supported on the GPU on Windows Native
pip install "tensorflow<2.11" 
```

The same environment `nullclass` can be used for running notebooks for other tasks as well. Now run the notebook named `task3.ipynb`.
