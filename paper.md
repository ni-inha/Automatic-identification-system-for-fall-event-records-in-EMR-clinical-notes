# Identification of fall incident records in clinical notes using fastText and BERT-based models

## 1. Introduction

## 2. Methods

### 2.1 Word Embeddings of fastText

인공 신경망 모델이 텍스트 기반 자연어를 학습하기 위해서는, 비정형화된 텍스트들은 숫자로 변환되어 나타내져야(represent) 한다.

단어를 표현하는 방법에 따라 자연어 처리의 성능이 달라지기 때문에 단어를 수치화하기 위한 많은 연구가 있었다.
Since the performance of NLP (Natural Language Processing) varies depending on the method of word representation, there have been many studies to digitize words.

그 중, word embeddings는 인공 신경망 학습으로 각 단어들을 벡터화하여 단어를 분산 표현하는 기법이며, 현재 가장 많이 사용되고 있는 기법이다.

word embeddings 기법들[Ref-Word-Embeddings](#ref-word-embeddings) 중 하나인 fastText는 Meta (Previously known as Facebook) AI Research Team으로부터 발표되었다.[Ref-fastText](#ref-fasttext) 하나의 단어로부터 분해되는 단어들을 서브워드(subword)라고 부르는데, fastText는 이러한 서브워드(subword)를 고려하여 최적의 단어 표현법을 학습한다.

각 단어들은 서브워드(subword)를 표현하기 위해, 글자(character) 단위 n-gram의 구성으로 취급되며, n 값을 통해 서브워드(subword)들을 몇 개로 분리할지 결정할 수 있다.

fastText는 서브워드(subword)가 가지는 특징으로, corpus에서 발생 빈도가 적은 희귀 단어(Rare Word)에 잘 대응할 수 있고, 심지어 corpus에 나타나지 않은 Out Of Vocabulary에도 쉽게 대응할 수 있다는 장점이 있다.

만약 어떤 희귀 단어(Rare Word)의 n-gram이 다른 어떤 단어의 n-gram과 중복된다면, 적지 않은 임베딩 벡터값을 얻게 된다. 또한, Out Of Vocabulary의 서브워드(subword)가 corpus에 존재하여 학습되었다면, Out Of Vocabulary도 영향을 받아 임베딩 벡터값을 얻게 된다.

clinical narratives는 정형화 되어 있지 않으므로, 오타(Typo)나 맞춤법이 틀린 단어가 종종 나타나는데, 이와 같이 노이즈가 많은 비정형 데이터는, fastText가 가지는 강점 때문에, Word Embeddings 기법들 중 fastText 기법이 가장 적합하다.

우리는 fastText 방법이 구현된 [fastText library](https://github.com/facebookresearch/fastText)로 낙상 진술을 판단하는 text classifier를 훈련시켰다.[ref-fasttext-text-classification](#ref-fasttext-text-classification)

### 2.2 Fine-tuning BERT models

fastText를 포함한 기존의 static word embedding 방식은 문맥의 의미를 내포하는 단어 임베딩 벡터를 생성할 수 없다.

contextualized word embedding 방식의 한 종류인 BERT(Bidirectional Encoder Representations from Transformers)[Ref-BERT](#ref-bert)는 출력 벡터에 대한 평균을 사용하여 지도 학습을 진행하고 효율적인 문장 임베딩을 수행하면서 이러한 문제를 해결한다.

트랜스포머[Ref-Transformer](#ref-transformer)은 어텐션 메커니즘만을 사용하여 인코더-디코더 구조를 구현한 네트워크 아키텍쳐이며, contextualized word embedding을 추출하는 기본 모델이다.

트랜스포머 모델은, NLP에 일반적으로 사용되었던 인공 신경망인 RNN(Recurrent Neural Network)과 RNN 기반의 LSTM에 비해, 정보 손실과 기울기 소실(vanishing gradient) 측면에서 강점을 가진다.

BERT는 이런 트랜스포머 모델을 바탕으로 사전 학습된 언어 모델이며, Pre-training 단계와 Fine-tuning 단계를 거쳐 사용된다.

BERT 모델은 Pre-training 단계에서, BooksCorpus (800M words) [Ref-BooksCorpus](#ref-bookscorpus)와 English Wikipedia (2,500M words)처럼 레이블이 없는 텍스트 데이터로 이미 사전 훈련되었기 때문에 적은 학습 데이터 만으로도 NLP의 다양한 분야에서 state-of-the-art results을 가진다.

Pre-training 단계에서 자연어에 대한 일반적인 지식을 얻은 pre-trained BERT 모델은 Fine-tuning 단계를 거쳐 원하는 자연어 문제를 해결할 수 있도록 훈련된다.

우리는 공개적으로 출시된 BERT 모델들 중, BERT-Base-Cased 모델과 BERT-Base-Multilingual-Cased 모델을 사용하였다. Cased 모델은 알파벳의 대문자와 소문자를 구분하는 모델로, 임상노트에서 자주 사용되는 의학용어와 의학약어가 대소문자로 구분되어 표현되기 때문에, 이점을 갖기 위하여 사용하였다. 더불어 Multilingual 모델은 영어와 한국어를 포함한 104개의 언어로 사전 학습된 모델이다. 이 모델은 여러 언어로 표현된 데이터에 강점을 가지므로, 우리는 이 모델을 사용하였다.

이외에도, 우리는 BERT 기반 한국어 언어 모델인 KLUE-BERT-Base 모델[Ref-KLUE-BERT](#ref-klue-bert)과 _임상_ 도메인에 특화된 Bio+Clinical BERT 모델[Ref-Bio+ClinicalBERT]을 사용하여 _낙상 사건을_ 식별하고자 하였다.

KLUE-BERT-Base 모델은 BERT-Base 모델을 기반으로 한국어 473M sentences와 6,519,504,240 Words를 사용한 한국어 특화 언어 모델이다.

Bio+Clincial BERT 모델은 biomedical text를 학습한 BioBERT-finetuned 모델[Ref-BioBERT]을 기반으로 MIMIC-III 데이터(2M clinical notes)[Ref-MIMIC-III](#ref-mimic-iii)를 추가로 사전학습한 모델이며, clinical NLP tasks에서 BERT 모델과 BioBERT 모델보다 좋은 성능을 가지는 것이 검증되었다.[Ref-Bio+ClinicalBERT]

우리는 이러한 사전 학습된 BERT 계열의 모델들을 우리의 데이터셋으로 미세 조정함으로서, *낙상 사건 식별*에 적합한 신경망을 한 층 추가하였다.

### 2.3 Dataset

우리는 대한민국(Republic of Korea) 내 3개 권역(region)의 주요 병원에서 임상 노트 데이터를 얻었다. (N=25916)

하지만, 병원에서의 낙상 사고 발생률이 낮기 때문에, 이러한 임상 노트 데이터 중 낙상 사고를 *진술*하는 데이터는 극히 일부이다.

낙상 클래스 데이터의 부족으로 데이터세트는 불균형해지고, 불균형한 데이터세트는 소수 클래스에 대한 오류 증가 문제를 동반한다. [Ref-Imbalanced-Dataset](#ref-imbalanced-dataset)

이러한 문제를 보완하기 위해, KOPS(KOrea Patient Safety reporting & learning system)의 낙상 사건 진술문을 추가로 이용하였다. (N=8564)

모든 데이터는 일부 영문 의학용어를 제외하고 한국어로 쓰여졌다.

TABLE

### 2.4 Data Pre-processing

모든 데이터는 인공 신경망 모델 훈련에 사용되기 전 익명화 처리를 거쳤다.

English Language Model을 사용하기 위해서는 영문 데이터를 사용해야 하므로, Google Cloud의 Cloud Translation API를 사용하여 한국어 데이터셋을 영문으로 번역하였다.

Korean Language Model을 사용하는 경우, 익명화된 데이터와, 그 익명화한 데이터에 추가적인 전처리를 수행한 데이터로 종류를 구분하였다.

우리는 전처리 과정에서는 정규화 규칙 및 도메인 용어 사전을 기반으로 영문 의학용어를 한국어로 바꾸는데에 집중하였고, 동의어 사전을 통해 여러개의 표현으로 쓰인 동의어들을 하나의 표현으로 통일시켰다.

또한, 한국어의 특성에 맞게 품사를 기준으로 토큰화하였다. 이 토큰화 과정에는 품사 사전과 불용어 사전 등이 사용되었다.

TABLE

### 2.5 Evaluation

우리는 모델의 성능을 평가하기 위한 주요 metric으로서 Precision과 Recall을 활용하는 F1-score와 F2-score를 선택했다. [Ref-F-measure](#ref-f-measure)

<!--
- Equations
  - $\mathrm{Precision}=\frac{\textsl{True\,Positives}}{\textsl{True\,Positives}+\textsl{False\,Postives}}$
  - $\mathrm{Recall}=\frac{\textsl{True\,Positives}}{\textsl{True\,Positives}+\textsl{False\,Negatives}}$
    -->

- Precision and Recall formulas
  - $\mathrm{Precision}=\frac{True\,Positives}{True\,Positives+False\,Potives}$
  - $\mathrm{Recall}=\frac{True\,Positives}{True\,Positives+False\,Negatives}$

Precision은 모델이 True라고 예측한 것들 중 실제 True인 것의 비율이고, Recall은 실제 True인 것 중에서 모델이 True라고 예측한 것의 비율이다.

- $F_\beta$-score formula
  - $F_\beta =(1+\beta ^2)\cdot \frac{precision\cdot recall}{(\beta ^2\cdot precision)+recall}$

F1-score($\beta=1$)는 Precision과 Recall의 조화 평균이므로 Precision과 Recall의 중요성을 동일하게 여긴다.

그러나 수많은 데이터 중 *낙상 데이터*를 식별하는 작업은 Precision보다 Recall이 더 중요하기 때문에, Recall에 더 비중을 두는 F2-score($\beta=2$)도 함께 고려하였다.

## 3. Results

우리는 모델의 학습과 평가에 한국어와 영어 2종류의 언어를 사용하였다. 이 섹션에서는 모델의 학습 결과를 언어별로 나누어 다양한 지표로 평가하였으며, 모든 평가 결과는 테이블에 기록되어있다.

TABLE

전반적으로 모델의 종류와 관계없이, 0.93 이상의 F1-score와 0.90 이상의 F2-score를 가졌으며, Korean Language Model의 성능이 English Language Model보다 좋은 결과를 보였다.

또한, contextualized word embedding 기반의 BERT 모델은, 문맥의 의미를 내포하는 단어 임베딩 벡터를 생성할 수 없는 fastText에 비해 우수한 성능을 보였다.

마지막으로, BERT기반 모델들은 더 사전학습된 모델일수록, 낙상 진술문을 더 잘 식별하는 경향을 보였다.

### 3.1 Korean Task

우리는 Korean Task의 fastText Model을 훈련시키기 위해 De-identified Korean data, Pre-processed Korean Data 2종류를 사용하였다. fastText Model에서 정규화, 동의어처리, 토큰화 등을 거친 Pre-processed Korean Data는 익명화만을 거친 데이터에 비해 모든 항목에서 우수한 결과를 보였다. 데이터가 토큰화를 포함한 전처리 과정을 거치면서 몇몇 단어가 대체되거나 생략되었으므로, Pre-processed Korean Data의 문맥은 손상된다. 그러나 오히려 불필요한 단어들을 제거하는 것이 non-contextualized word embedding 방식의 fastText 모델에는 이점을 주었다.

fastText 모델과는 다르게, contextualized word embedding 방식을 사용하는 BERT 기반 모델에서는 문맥이 유지되는 De-identified Korean data만을 사용하여 평가하였다. KLUE BERT-base 모델과 BERT-base-multilingual-cased 모델을 비교할 때, KLUE BERT-base 모델은 Precision에서 강점을 가지며, BERT-base-multilingual-cased 모델은 Recall에서 강점을 가지기 때문에 F2-score가 더 높은 것은 BERT-base-multilingual-cased이었다. 하지만 두 모델들간에 유의미한 차이를 보이지는 않았다.

### 3.2 English Task

English Task를 평가하기 위해서는 De-identified English data 한 종류만을 사용하였다.

먼저, fastText 모델을 확인해보면 0.9483의 F1-score와 0.9508의 F2-score를 보였다. 이는 Korean Task의 fastText Model과 비교하였을 때, 0.02 정도의 차이였다. 우리는 English task가 Korean task에 비해 낮은 성능을 보이는 이유로, 번역의 오류와 언어적 특성 때문으로 추정한다.

BERT-base-multilingual-cased 모델에서는 0.9304의 F1-score와 0.9012의 F2-score를 보였다. Korean Task에서 보였던 성능과 확연한 차이를 보였는데, 우리는 이것의 원인으로 이는 번역된 데이터의 특성과 Multilingual-BERT의 특징을 지목한다.

Multilingual-BERT[Ref-Multilingual-BERT](#ref-multilingual-bert)는 여러 언어가 한 문장에 등장하는 Code-switching의 경우에는 언어 간 일반화(generalize crosslingually)를 지원하지만, 글자를 소리나는대로 표기하는 Transliterate의 경우에는 cross-lingual transferability를 지원하지 않는다. 우리가 한국어 데이터를 영문 데이터로 번역하기 위해 사용한 Google Translation API는 번역에 있어 애매모호한 단어가 있으면, 그것을 Transliterate하여 영문으로 변환한다. 바로 이 과정에서 데이터의 손실이 발생했고, 이것이 성능으로까지 이어진 것으로 추측한다.

BERT-base-cased 모델에서는 BERT-base-multilingual-cased 모델과 달리, 높은 성능을 나타냈다. 이것은 Korean Task에서 사용된 다른 모델들과 비교해보았을 때도 큰 차이가 없었다.

BERT 모델에 biomedical text와 임상 노트를 사전 학습한 Bio+Clinical BERT의 성능은 English Task에서 사용된 모델들 중 모든 지표에서 가장 우수했다. 이 모델은 도메인 기반의 사전 학습 모델이 성능에 얼마나 영향을 끼칠 수 있는지 보여준다.

## 4. Conclusion

## References

### [1]

### []

### [Ref-Imbalanced-Dataset]

An Improved Algorithm for Neural Network Classification of Imbalanced Training Sets

### [Ref-Word-Embeddings]

1. Efficient Estimation of Word Representations in Vector Space
2. GloVe: Global Vectors for Word Representation

### [Ref-Transformer]

Attention is all you need

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

### [Ref-F-measure]

- MUC-4 EVALUATION METRICS
- The truth of the F-measure

### [Ref-Multilingual-BERT]

How multilingual is Multilingual BERT?
