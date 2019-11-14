# kg-cvae-generator

kg-CVAEの実装。Python3 / torch 1.1.0



## Introduction

▼ paper

> [Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders（ACL17）](https://arxiv.org/abs/1703.10960)

BASE : [pingpong-ai/kg-cvae-generator](https://github.com/pingpong-ai/kg-cvae-generator)：**Python3/PyTorch 1.1.0**



## Getting Started


韓国語の学習

    python main_kor.py 

英語の学習

    python main_eng.py 

学習済みモデルで推論

    python inference.py 



学習の際、学習済み埋め込みモデルを活用することができます。英語の場合Glove.txt, ハングルの場合Fasttext.binファイル形式に対応しますｙ。

埋め込みモデルの使用はcorpud configファイルのword2vec_path変数に埋め込みファイルのパスを指定してください。

- 英語の場合は[Stanfordの埋め込みモデル](https://nlp.stanford.edu/projects/glove/)でTwitterの200次元の埋め込みを使用しています。
- 韓国語の場合には、木のwiki（韓国のwiki）を学習させた300次元の埋め込みを使用しています。



## Dataset


- 英語の場合、ワン著者の論文のSwitchboardコーパス（JSONLフォーマット）を参照しています。
- 韓国語の場合には、[恋愛の科学](https://scienceoflove.co.kr/)で抽出された会話データを活用しました。example_kor.jsonファイルを参照してください。



## Requirements

```
torch==1.1.0
tqdm
numpy
nltk (for english)
```



## 参考

**Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders（ACL17）**

[snakeztc/NeuralDialog-CVAE](https://github.com/snakeztc/NeuralDialog-CVAE)：Python2.7/Tensorflow1.3.0 /cuDNN 6 

- [hfef7ui2/final_year_project_kgCVAE](hfef7ui2/final_year_project_kgCVAE)：**Python3.5**/TensorFlow 1.3.0/cuDNN 6
- [ruotianluo/NeuralDialog-CVAE-pytorch](ruotianluo/NeuralDialog-CVAE-pytorch)：Python2.7/PyTorch 0.4
  - [pingpong-ai/kg-cvae-generator](https://github.com/pingpong-ai/kg-cvae-generator)：**Python3/PyTorch 1.1.0**