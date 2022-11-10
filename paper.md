# To-Do-List

- [2.1 Dataset](#21-dataset)에서 한국어의 토큰화 단계를 포함한 데이터의 전처리 과정 설명

---

# Identification of fall incident records in clinical notes using fastText and BERT-based models

## 1. Introduction

## 2. Methods

### 2.1 Dataset

우리는 대한민국(Republic of Korea) 내 3개 권역(region)의 주요 병원에서 임상 노트 데이터를 얻었다. (N=25916)

하지만, 병원에서의 낙상 사고 발생률이 낮기 때문에, 이러한 임상 노트 데이터 중 낙상 사고를 *진술*하는 데이터는 극히 일부이다.

낙상 클래스 데이터의 부족으로 데이터세트는 불균형해지고, 불균형한 데이터세트는 소수 클래스에 대한 오류 증가 문제를 동반한다. [Ref-Imbalanced-Dataset](#ref-imbalanced-dataset)

이러한 문제를 보완하기 위해, KOPS(KOrea Patient Safety reporting & learning system)의 낙상 사건 진술문을 추가로 이용하였다. (N=8564)

모든 데이터는 일부 영문 의학용어를 제외하고 한국어로 쓰여졌으며, 낙상 식별을 위한 인공 신경망 모델 훈련에 사용되기 전 익명화 처리를 거쳤다.

영문 데이터를 테스트 하기 위해, Google Cloud의 Cloud Translation API를 사용하여 한국어 데이터셋을 영문으로 번역하였다.

TABLE

### 2.2 Word Embeddings of fastText

인공 신경망 모델이 텍스트 기반 자연어를 학습하기 위해서는, 비정형화된 텍스트들은 숫자로 변환되어 나타내져야(represent) 한다.

단어를 표현하는 방법에 따라 자연어 처리의 성능이 달라지기 때문에 단어를 수치화하기 위한 많은 연구가 있었다.
Since the performance of NLP (Natural Language Processing) varies depending on the method of word representation, there have been many studies to digitize words.

그 중, word embeddings는 인공 신경망 학습으로 각 단어들을 벡터화하여 단어를 분산 표현하는 기법이며, 현재 가장 많이 사용되고 있는 기법이다.

word embeddings 기법들 중 하나인 fastText는 Meta (Previously known as Facebook) AI Research Team으로부터 발표되었다.[Ref-fastText](#ref-fasttext) 하나의 단어로부터 분해되는 단어들을 서브워드(subword)라고 부르는데, fastText는 이러한 서브워드(subword)를 고려하여 최적의 단어 표현법을 학습한다.

각 단어들은 서브워드(subword)를 표현하기 위해, 글자(character) 단위 n-gram의 구성으로 취급되며, n 값을 통해 서브워드(subword)들을 몇 개로 분리할지 결정할 수 있다.

fastText는 서브워드(subword)가 가지는 특징으로, corpus에서 발생 빈도가 적은 희귀 단어(Rare Word)에 잘 대응할 수 있고, 심지어 corpus에 나타나지 않은 Out Of Vocabulary에도 쉽게 대응할 수 있다는 장점이 있다.

만약 어떤 희귀 단어(Rare Word)의 n-gram이 다른 어떤 단어의 n-gram과 중복된다면, 적지 않은 임베딩 벡터값을 얻게 된다. 또한, Out Of Vocabulary의 서브워드(subword)가 corpus에 존재하여 학습되었다면, Out Of Vocabulary도 영향을 받아 임베딩 벡터값을 얻게 된다.

clinical narratives는 정형화 되어 있지 않으므로, 오타(Typo)나 맞춤법이 틀린 단어가 종종 나타나는데, 이와 같이 노이즈가 많은 비정형 데이터는, fastText가 가지는 강점 때문에, Word Embeddings 기법들 중 fastText 기법이 가장 적합하다.

우리는 fastText 방법이 구현된 [fastText library](https://github.com/facebookresearch/fastText)로 낙상 진술을 판단하는 text classifier를 훈련시켰다.[ref-fasttext-text-classification](#ref-fasttext-text-classification)

### 2.3 Fine-tuning BERT models

트랜스포머 모델은 어텐션 메커니즘만을 사용하여 인코더-디코더 구조를 구현한 네트워크 아키텍쳐이다.

NLP에 일반적으로 사용되었던 인공 신경망인 RNN(Recurrent Neural Network)과 RNN 기반의 LSTM에 비해, 트랜스포머 모델은 정보 손실과 기울기 소실(vanishing gradient) 측면에서 강점을 가진다.

Google이 설계한 BERT(Bidirectional Encoder Representations from Transformers)[Ref-BERT](#ref-bert)는 트랜스포머 모델을 바탕으로 사전 학습된 언어 모델이며, Pre-training 단계와 Fine-tuning 단계를 거쳐 사용된다.

BERT 모델은 Pre-training 단계에서, BooksCorpus (800M words) [Ref-BooksCorpus](#ref-bookscorpus)와 English Wikipedia (2,500M words)처럼 레이블이 없는 텍스트 데이터로 이미 사전 훈련되었기 때문에 적은 학습 데이터 만으로도 NLP의 다양한 분야에서 state-of-the-art results을 가진다.

Pre-training 단계에서 자연어에 대한 일반적인 지식을 얻은 pre-trained BERT 모델은 Fine-tuning 단계를 거쳐 원하는 자연어 문제를 해결할 수 있도록 훈련된다.

우리는 공개적으로 출시된 BERT 모델들 중, BERT-Base-Cased 모델과 BERT-Base-Multilingual-Cased 모델을 사용하였다. Cased 모델은 알파벳의 대문자와 소문자를 구분하는 모델로, 임상노트에서 자주 사용되는 의학용어와 의학약어가 대소문자로 구분되어 표현되기 때문에, 이점을 갖기 위하여 사용하였다. 더불어 Multilingual 모델은 영어와 한국어를 포함한 104개의 언어로 사전 학습된 모델이다. 이 모델은 여러 언어로 표현된 데이터에 강점을 가지므로, 우리는 이 모델을 사용하였다.

이외에도, 우리는 BERT 기반 한국어 언어 모델인 KLUE-BERT-Base 모델[Ref-KLUE-BERT](#ref-klue-bert)과 _임상_ 도메인에 특화된 Bio+Clinical BERT 모델[Ref-Bio+ClinicalBERT]을 사용하여 _낙상 사건을_ 식별하고자 하였다.

KLUE-BERT-Base 모델은 BERT-Base 모델을 기반으로 한국어 473M sentences와 6,519,504,240 Words를 사용한 한국어 특화 언어 모델이다.

Bio+Clincial BERT 모델은 biomedical text를 학습한 BioBERT-finetuned 모델[Ref-BioBERT]을 기반으로 MIMIC-III 데이터(2M clinical notes)[Ref-MIMIC-III](#ref-mimic-iii)를 추가로 사전학습한 모델이며, clinical NLP tasks에서 BERT 모델과 BioBERT 모델보다 좋은 성능을 가지는 것이 검증되었다.[Ref-Bio+ClinicalBERT]

우리는 이러한 사전 학습된 BERT 계열의 모델들을 우리의 데이터셋으로 미세 조정함으로서, *낙상 사건 식별*에 적합한 신경망을 한 층 추가하였다.

### 2.4 Evaluation

We chose the F-Measure as the main metric to evaluate the quality of the models. F-Measure corresponds to the harmonic mean between precision and recall.

## 3. Results

### 3.1 FastText Model

### 3.2 Fine-tuned BERT-based Models

#### A. Fine-tuned KLUE BERT-Base model

#### B. Fine-tuned BERT-Base-multilingual-cased model

#### C. Fine-tuned BERT-Base-cased model

#### D. Fine-tuned Bio+Clinical BERT model

## 4. Conclusion

## References

### [1]

### []

### [Ref-Imbalanced-Dataset]

An Improved Algorithm for Neural Network Classification of Imbalanced Training Sets

### [Ref-BERT]

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

### [Ref-fastText]

Enriching Word Vectors with Subword Information

### [Ref-fastText-text-classification]

FastText.zip: Compressing text classification models
Bag of Tricks for Efficient Text Classification

### [Ref-BooksCorpus]

Aligning Books and Movies- Towards Story-like Visual Explanations by Watching Movies and Reading Books

### [Ref-KLUE-BERT]

KLUE: Korean Language Understanding Evaluation

### [Ref-BioBERT]

BioBERT- a pre-trained biomedical language representation model for biomedical text mining

### [Ref-Bio+ClinicalBERT]

Publicly Available Clinical BERT Embeddings

### [Ref-MIMIC-III]

MIMIC-III, a freely accessible critical care database

---

## 미사용 레퍼런스

### [Ref-Multilingual-BERT]

How multilingual is Multilingual BERT?

> generalize crosslingually (언어 간 일반화)

> M-BERT가 다국어 표현을 생성하지만, 이러한 표현은 특정 언어 쌍에 영향을 미치는 체계적인 결함을 보인다는 결론을 내릴 수 있습니다.

### [Ref-F-measure]

- MUC-4 EVALUATION METRICS
- The truth of the F-measure

F1 Measure는 Precision과 Recall의 중요성을 동일하게 보고 있다.

만약 Recall이 Precision보다 더 중요하다면 F2 Measure를, Precision이 Recall보다 더 중요하다면 F0.5 Measure를 사용할 수 있다.

여기서 2나 0.5는 베타의 값이라는 것을 기억해 두기 바란다.

---

추출된 Keyword에 대해서 성능을 분석하는 방법으로
Recall과 Precision을 활용하는 F1-Score를 사용하였다.
Recall은 실제 True인 것 중에서 모델이 True라고 예측한 것의 비율이다. 통계학에서는 Sensitivity로, 그
외 분야에서는 Hit rate라는 용어로도 사용된다.[5]

(2)
Precision은 모델이 True라고 분류한 것들 중 실제
True인 것의 비율이다. Precision은 Positive 정답률,
PPV(Positive Predictive Value)라고도 불린다.[5]

F1-Score는 Recall과 Precision의 조화평균이다. F1
Score는 데이터 Label이 불균형 구조일 때, 모델의 성능을 정확하게 평가할 수 있으며, 성능을 하나의 숫자로 표현할 수 있다.[5]
