=========
kg-cvae-generator
=========
핑퐁에서 구현한 KG-CVAE 기반 문장 생성 모듈입니다.

Introduction
------------
이 코드는 ACL 2017에서 발표된 **Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders** 를 PyTorch를 이용하여 재구현한 것입니다. 원 논문은 `ACL Anthology <https://www.aclweb.org/anthology/P17-1061/>`_ 를 참조해주십시오. 

이 코드는 원 저자의 구현(`snakeztc/NeuralDialog-CVAE <https://github.com/snakeztc/NeuralDialog-CVAE>`_)과 이를 부분적으로 PyTorch로 구현한 Ruotian Luo님의 구현(`ruotianluo/NeuralDialog-CVAE-pytorch <https://github.com/ruotianluo/NeuralDialog-CVAE-pytorch>`_)를 참조하였습니다.

Reference
---------
혹시 이 코드를 직접 사용하시거나 데이터 셋을 활용하시는 경우에는 원 저자의 다음 코드를 참조해주시기 바랍니다.

.. code-block:: text
 
    [Zhao et al, 2017]:
     @inproceedings{zhao2017learning,
       title={Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders},
       author={Zhao, Tiancheng and Zhao, Ran and Eskenazi, Maxine},
       booktitle={Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
       volume={1},
       pages={654--664},
       year={2017}
     }

Getting Started
---------------

한국어 훈련을 위해서는 다음 파일을 실행시켜 주십시오.

.. code-block:: shell

    python main_kor.py 
    
영어 훈련을 위해서는 다음 파일을 실행시켜 주십시오.

.. code-block:: shell

    python main_eng.py 
    
훈련된 모델로 실험하기 위해서는 다음 파일을 실행시켜 주십시오.

.. code-block:: shell

    python inference.py 

훈련 시킬 때, 기훈련 임베딩을 활용할 수 있습니다. 영어의 경우 Glove txt파일, 한글의 경우 Fasttext bin파일 형식을 지원합니다. 

기훈련 임베딩을 사용하시기 위해서는 corpus config 파일의 word2vec_path 변수에 기훈련 임베딩 파일 경로를 명시해주십시오. 

* 영어의 경우 `Stanford 임베딩 <https://nlp.stanford.edu/projects/glove/>`_ 에서 Twitter를 활용한 200차원 임베딩을 기본으로 사용하고 있습니다.
* 한국어의 경우, 나무위키로 학습시킨 300차원 임베딩을 기본으로 사용하였습니다.

Dataset
---------------
* 영어의 경우, 원 저자 논문의 Switchboard 말뭉치 (JSONL 포맷)를 참조하였습니다.
* 한국어의 경우, `연애의 과학 <https://scienceoflove.co.kr/>`_ 에서 추출된 대화 데이터를 활용하였습니다. example_kor.json 파일을 참조해주십시오.

Requirements
------------

.. code-block:: text

    torch==1.1.0
    tqdm
    numpy
    nltk (for english)
    

Authors
-------
**Pingpong AI Research, Machine Learning Engineers**

- Written by `구상준 Sangjun Koo`_ , `장성보 Seongbo Jang`_.
- Previously researched by 서수인 Suin Seo.

.. _구상준 Sangjun Koo: koosangjun@scatterlab.co.kr
.. _장성보 Seongbo Jang: seongbo@scatterlab.co.kr

License
-------
본 코드에서 원 저자들이 작성한 코드 기반으로 부가적으로 작성된 모듈 (decoder_fn_lib.py, corpus_eng.py 등)을 제외한 나머지 소스에 대해서는 다음 아파치 라이선스 2.0을 따르고 있습니다.

Copyright 2019 Pingpong AI Research, ScatterLab `Apache License 2.0 <https://github.com/pingpong-ai/chatspace/blob/master/LICENSE>`_
