# 1030_SKAX_모듈1_생성형 AI 이해 및 Prompt Engineering


## 실습환경

- 호스트OS: Windows 10 Enterprise 22H2 (19045.6093)
- Music AI SUNO 무료계정으로 음악 생성
- OpenAI Tokenizer로 토큰 확인
- Prompt 실습: Claude Free (Sonnet 4.5), Gemini Pro (2.5 Flash), OpenRouter; Nemotron-H3 (Nemotron-H Nano)


---

## Lab 1. 생성형 AI에 대한 이해

> 생성형 AI의 기본 개념을 정의하고, AI와 AGI의 차이점을 비교하며 LLM의 작동 원리, 한계 및 최신화 방식을 탐구.


### 1. AI와 생성형 AI
- 인공 지능(AI)의 광범위한 개념과, 그 하위 분야로서 '새로운 컨텐츠를 생성'하는 생성형 AI를 구분합니다.
- **인공지능(AI)**: '기계를 인간과 더 유사하게 만든다'라는 광범위한 개념.
- **생성형 AI**: '새로운 컨텐츠를 의미 있고 지능적으로 생성'하는 AI의 하위 분야


### 2. AGI(Artificial General Intelligence)

- 인간과 유사한 지능 및 자가 학습(self-teach) 능력을 갖춘 소프트웨어를 구현하려는 이론적 AI 연구 분야.

- AGI is a field of theoretical AI research that _attempts to create software with human-like Intelligence and the ability to self-teach.
- The aim is for the software to be able to perform tasks that **it is not necessarily trained or developed for**.

> [AWS - What is Artifical-General-Intelicence](https://aws.amazon.com/what-is/artificial-general-intelligence/?nc1=h_ls)


### 3. Foundation Model 및 LLM

- 생성형 AI의 근간이 되는 모델로, 광범위한 비정형 데이터를 기반으로 훈련된 머신러닝 모델.

- **Prompt**: 사용자가 AI와 상호작용하는 입력창(e.g., GTP 대화창)
- **Foudation Model (FM)**:
	- 일반화되고 레이블이 없는 광범위한 데이터를 기반으로 훈련된 MachineLearning 모델.
	- 질문 답변, 에세이 작성, 이미지 캡션 등 다양한 일반적인 작업을 수행할 수 있다.
- **Large Language Model (LLM)**:
	- FM의 일종으로, 요약, 텍스트 생성, 분류, 개방형 대화, 정보 추출 등의 **언어 기반 작업**에 특화된 모델.
	- 파라미터 수에 따라 분류하기도 한다.
	  (e.g., 30bil+ LLM, 7bil+- sLLM)

> [AWS - What is Generative-AI](https://aws.amazon.com/ko/what-is/generative-ai/)


### 4. LLM의 작동 원리 및 한계

- 자연어를 수치적 데이터로 변환(Embedding)하여 연산을 수행하며, 학습 시점 이후의 정보 부재로 인한 성능 저하(Model Decay) 한계를 가진다.

#### 4-1. 임베딩 (Embedding)

- LLM의 핵심 선행 작업
- 자연어를 컴퓨터가 이해할 수 있는 숫자(bit, Vector)로 변환하는 과정.
- **수많은 단어를 컴퓨터가 수치적으로 이해하고 있어야, 그 다음 논리적 연산 처리가 가능함**.


#### 4-2. 모델의 한계
- **성능 저하(Model Decay)**:
	- 모델은 **이전에 학습한 데이터를 기반**으로 컨텐츠를 생성함.
	- 따라서 모델이 만들어진 이후부터는, 시간이 흐름에 따라 최신 정보가 부족하여 성능이 하락하게 됨.
- **보안 및 환각 (Security & Halucination)**:
	- 학습 데이터 외부의 정보(e.g., 회사 내부 기밀)에는 접근이 불가.
	- 학습하지 않은 내용에 대해 그럴듯한 거짓 정보를 생성하는 **환각 현상**이 발생할 수 있음.


#### 4-3. 모델 최신화 전략
- **RAG(Retrieval-Augmented Generation)**: 모델 외부의 지식(e.g., 실시간 검색)을 허용하여, LLM이 이를 참고해 답변을 생성하도록 하는 방식.
- **Fine-Tuning**: 모델에 직접 새로운 데이터를 추가 학습시켜 모델 자체를 변경(업데이트)하는 방식.


### 5. 생성형 AI의 활용 및 동향
- Biz. Trend
- 리뷰 요약
- Multi-Modal AI 영상분석
- [Music AI - facebook/melodyflow](https://huggingface.co/facebook/melodyflow-t24-30secs) 
- [Music AI - SUNO](https://suno.com/home)
- Gartner Hype Cycle: 최신 기술 동향 및 시장 기대치 파악에 활용


---

## Lab 2. LLM 응용 기술: Fine-Tuning과 RAG, NLP

> 대규모 언어 모델(LLM)의 성능을 향상시키는 주요 접근법 두 가지인 Fine-Tuning과 RAG를 비교하고, 자연어 처리(NLP)의 기본 개념을 정의.


### 1. NLP(Natural Language Processing)
- 자연어 처리(NLP)는 ML을 사용하여 **컴퓨터가 인간의 언어를 이해하고 소통하도록 돕는 인공 지능(AI)의 하위 분야**.

> [IBM - NLP란 무엇인가요?](https://www.ibm.com/kr-ko/think/topics/natural-language-processing)
> [Oracle - 자연어 처리(NLP)란?](https://www.oracle.com/kr/artificial-intelligence/natural-language-processing/)
> [DeepLearning.AI - Natural Language Processing](https://www.deeplearning.ai/resources/natural-language-processing/)


### 2. LLM 성능 향상 방법론

- 사전 학습된(Pre-trained) LLM의 지식을 확장하거나 특정 작업에 최적화하는 두 가지 주요 기술

#### 2-1. Fine-Tuning
- 모델에 **신규 데이터를 추가 학습**시켜서, 모델을 변경하는 것 (e.g. 버전 업)


#### 2-2. RAG(Retrieval-Augmented Generation)

- 모델이 학습되지 않은 데이터에 접근할 수 있도록 구현하는 기술을 RAG라고 한다.
- LLM이 참고할 수 있는, 모델 외부의 새로운 지식 기반을 만드는 기술.
- 그렇게 만든 외부 지식 베이스를 참고해서 답변을 하게 되면
	- LLM의 성능이 지속적으로 향상된다고 말할 수 있을까?
	- 그렇다고 한다.
	- 기존에는 LLM을 계속 학습시켜서 그 모델의 크기를 부풀려왔다.
		- 여기에 필연적으로 비용이 크게 발생하고, LLM의 성능 저하 이슈도 발생.

> [Appen - 파인튜닝이란? - LLM 구축 방법](https://kr.appen.com/blog/fine-tuning/)
> [Appen - Guide to Chain-of-Thought Prompting](https://www.appen.com/whitepapers/chain-of-thought)
> [제주한라대학교 인공지능학과 - 파인튜닝](https://mlops2024.jeju.ai/projects/202021012/finetuning/index.html)
> [Red Hat - RAG vs. fine-tuning](https://www.redhat.com/en/topics/ai/rag-vs-fine-tuning)
> [Bottleneck feature? Fine-tuning?](https://haandol.github.io/2016/12/25/define-bottleneck-feature-and-fine-tuning.html)





---

## Lab 3. 자언어 처리를 위한 텍스트 표현 및 트랜스포머 아키텍처 분석

> 자연어 처리(NLP) 태스크 수행을 위한 텍스트 데이터의 수치적 변환 방법론,
> 현대 언어 모델의 근간이 되는 트랜스포머 아키텍처의 핵심 구성 요소.


### 1. Tokenizer

- LLM이 텍스트를 처리하는 기본 단위(Token)로 분리하는 작업.
- 초기 LLM은 한자, 조사 등 구조가 복잡한 한국어 어휘를 의미적으로 파악하고 토큰화하는 데에 한계가 있었다.

> [OpenAI - Tokenizer](https://platform.openai.com/tokenizer)


### 2. 텍스트 표현 방법론 (Text Representation)

_컴퓨터가 자연어를 처리하기 위해서는 텍스트를 수치적 데이터(벡터)로 변환하는 과정이 필수적입니다._


#### 2-1. 빈도 기반 표현: Bag of Words (Bow)
- BoW는 문서를 구성하는 단어의 **빈도(frequency)** 에 초점을 맞춘 통계적 텍스트 표현 방식
- 모든 단어(Token)을 벡터로 변환하되, 각각의 토큰은 전체 어휘(Vocabulary) 內
  _특정 단어의 등장 횟수 + 중요도(e.g., TF-IDF)를 나타낸다_.
- 단어의 순서나 문맥적 의미를 보존하지 못한다.
  (e.g., "A가 B를 때렸다"와 "B가 A를 때렸다" 두 문장을 동일하게 인식할 수 있다.)

> [codong - 3. 카운트 기반의 단어 표현](https://codong.tistory.com/34)
> [glowdp - 카운트 기반 언어모델](https://glowdp.tistory.com/142)


#### 2-2. 분포 기반 표현: 단어 임베딩 (Word Embedding)

- 빈도 기반 표현의 한계를 극복하기 위해, 단어의 '의미'를 벡터 공간에 투영하는 방식이 제안되었다.
- _의미적 추론_, _분포 가설의 구현_

- 단어를 밀집된(dense) 실수(Real Number) 벡터로 표현하며, 단어 간 의미적 관계(문맥)을 포착한다.

> [glowdp - 의미기반 언어 지식 표현 체계](https://glowdp.tistory.com/145)


##### 2-2-1. 이론적 배경: 분포 가설(Distributional Hypothesis)

- **단어 임베딩의 핵심**
- 특정 단어의 의미는 해당 단어의 주변에 등장하는 단어들에 의해 형성된다는 가설.
- 유사한 문맥(context)에서 등장하는 단어들은 유사한 의미를 가질 것으로 가정한다.

> [Github.QA](https://github.com/boost-devs/peer-session/issues/62)
> [hezma - 8. Distributional Representation of Words](https://hezma.tistory.com/102)


##### 2-2-2. 구현: 단어의 분산 표현(Distributional Representation)

- **분포 가설**을 구현하여 자연어를 벡터로 표현하는 과정을 의미한다.
- 자연어 단어를 Embedding 과정을 거쳐 _(숫자로 고쳐, bit로 바꾸어서)_
  **밀집된 _(dense)_ 실수 _(Real Number)_ 벡터로 표현하는 것**.
- **Embedding Vector**는 단어 간의 _**의미적, 문법적 관계**를 벡터 공간 내의 **거리, 방향**으로 표현할 수 있다_.
- e.g., **Word2Vec**


### 3. 언어 모델(Language Model)

- 문장을 보고, '단어 시퀀스를 찾아내서', 이게 수학적(확률 상)으로 "얼마나 그럴 듯 하냐" 따져주는 모델
- _텍스트 데이터의 통계적 패턴을 학습하여_,
	- 특정 단어 시퀀스(문장)의 등장 확률을 계산하거나,
	- 이전 단어들이 주어졌을 때 다음 단어를 예측하는 모델.


### 4. Transformer Architecture

- 2017년 Google이 발표한 "Attention Is All You Need" 논문에서 소개된 딥러닝 모델 아키텍처.
- 기존의 RNN(Recurrent Neural Network) 계열 모델의 한계를 극복, 자연어 처리(NLP) 분야에 혁신을 가져왔다.


#### 4-1. Introduct.

> 순환(Recurrence) 구조 대신 Self-Attention 메커니즘을 도입하여, 시퀀스 내 단어 간의 장거리 의존성(long-range dependency) 문제를 해결.
> _입력된 텍스트(Token)는 **단어의 의미(Word Embedding)** 와 **순서(Positional Embedding)** 정보를 모두 포함하는 벡터로 변환되어 모델에 입력됩니다._

- Transformer는 입력된 텍스트를 처리하고, 그 의미를 파악하며, 새로운 텍스트를 생성하는 데 핵심적인 역할을 한다.
- 이러한 과정은 단어를 '숫자 벡터'로 표현하는 방식에 의존한다.
	- Transformer는 입력 정보를 **Word Embedding**을 사용하여 단어의 의미적, 문맥적 정보를 벡터 형태로 받아들이고,
	- 이를 **Self-Attention** 메커니즘을 통해 효과적으로 처리한다.


#### 4-2. 주요 구성 요소

- Transformer는 세 가지 구성 요소:
	  입력한 임베딩(Embedding)
	  인코더(Encoder)
	  디코더(Decoder)
  로 구성됩니다.

##### 4-2-1. Embedding Layer

- 입력 텍스트의 각 단어(Token)를 **의미 정보를 함축한** 벡터로 변환한다.
- Word Embedding: 단어(Token)의 의미를 담음
- Positional Embedding: 단어(Token)의 위치 정보를 담음.
  Self-Attention 메커니즘은 본질적으로 순서를 고려하지 않으므로, 단어의 절대적/상대적 위치(순서) 정보를 벡터에 명시적으로 추가한다.

##### 4-2-2. Encoder

- 입력 시퀀스 전체의 문맥 정보를 이해하고 압축하는 역할을 수행한다.
- Self-Attention: Encoder의 주요 메커니즘.
  입력 시퀀스의 모든 단어 쌍 간의 연관도를 계산한다.
  이 과정을 통해 각 단어는 문장 전체의 문맥을 반영한 벡터 표현 (Context Vector)을 생성한다.
- BERT와 같은 모델은 인코더 구조를 활용하기에, 문맥 이해에 특화.

##### 4-2-3. Decoder

- 인코더가 생성한 문맥 벡터를 바탕으로 **새로운 출력 시퀀스(e.g., 번역된 문장)**를 한 단어씩 확률 예측으로써 생성한다.
- Masked Self-Attention: 이미 생성된 단어들 사이의 관계를 파악한다.
  _출력 시퀀스를 생성할 때, 현재 예측하려는 시점 이후의 단어들(미래 정보)을 참조하지 못하도록 마스킹(Masking)하여, 이미 생성된 단어들 사이의 관계만 파악합니다_.
- Cross-Attention:
  _인코더가 학습한 입력 문맥(Context Vector)과 디코더가 현재까지 생성한 출력 간의 관계를 계산합니다.
  이를 통해 입력 문맥에 가장 적합한 다음 단어를 예측하도록 Attention을 집중합니다_.
- GPT와 같은 모델은 디코더 구조를 활용하여 텍스트 생성에 특화된다.
- _BART처럼 Encoder-Decoder 구조를 모두 사용하는 모델도 존재_.

> [glowdp - Encoder Model(BERT)](https://glowdp.tistory.com/134)
> [glowdp - Decoder Model(GPT)](https://glowdp.tistory.com/135)
> [glowdp - Encoder-Decoder Model(BART)](https://glowdp.tistory.com/136)
> [@hubmanskj - LLM이란](https://medium.com/@hugmanskj/%EA%B1%B0%EB%8C%80-%EC%96%B8%EC%96%B4-%EB%AA%A8%EB%8D%B8-large-language-model-%EC%97%90-%EB%8C%80%ED%95%9C-%EC%9D%B4%ED%95%B4-%EC%96%B8%EC%96%B4%EB%AA%A8%EB%8D%B8%EC%9D%B4%EB%9E%80-d52a120efe52)
> [@hubmanskj - LLM 기준과 특징](https://medium.com/@hugmanskj/%EA%B1%B0%EB%8C%80-%EC%96%B8%EC%96%B4-%EB%AA%A8%EB%8D%B8-large-language-model-%EC%97%90-%EB%8C%80%ED%95%9C-%EC%9D%B4%ED%95%B4-llm%EC%9D%98-%EA%B8%B0%EC%A4%80%EA%B3%BC-%ED%8A%B9%EC%A7%95%EC%9D%80-%EB%AC%B4%EC%97%87%EC%9D%BC%EA%B9%8C-0551b7b9d3bd)
> [seanpark11 - LLM의 학습 방법론](https://seanpark11.tistory.com/165)


---

## Lab 4. Prompt Engineering


> LLM이 정확한 결과를 도출하도록 명확한 자연어 문장을 뽑아내는 작업.
> 문장을 오해의 소지 없도록 작성하고, 요구사항을 A to Z까지 녹여내는 것


### 1. LLM 출력 구성 요소
- LLM 출력은, 생성된 단어 다음에 올 토큰을 확률을 통해 결정하고 그 과정을 반복하는 것.


#### 1-1. 출력 길이(Output Length / Max Token)

- 최대 한도가(LLM 제한) 있어서 출력 길이가 제한을 넘어서면 토큰 예측을 중단함.


#### 1-2. 생성 무작위성 제어(Temperature)

- 확률이 높은 토큰만 선택하도록 하면, 천편일률적인 답변만 받게 될 것이다.
- **Temperature**: 토큰 선택의 무작위성을 제어하는 척도.
	- _온도가 낮으면 (e.g., 0.1)_: 확률이 높은 토큰만 선택하여 **경직되고 일관된** 답변 생성.
	- _온도가 높으면 (e.g., 1.0)_: 토큰 간 확률이 비슷해져 **창의적이고 다양한** 답변 생성.
- **Top-K**: 예측 토큰의 확률로 순위를 매겨, 상위 **\<Top-K>개** 안에서만 토큰을 선택함.
- **Top-p**: 예측 토큰의 확률로 백분위 분포를 따지고, 누적 확률이 상위 **\<Top-p>%** 안에 속하는 토큰 집합 내에서만 선택함.
	- 대입 조건의 Top-p가 10%다. == 1등급, 2등급 안에서만 입학생을 고르겠다.


#### 1-3. 프롬프트 유형(User/System Prompt)

- **User Prompt**: - 대화창에서 사용자가 FM(Foundational Model)에게 전달하는 **실시간 입력**.
- **System Prompt**: 관리자(개발자)가 초기에 FM에 설정한 **전반적인 지침 또는 역할**.


### 2. Prompt 구성 5원칙

- **목적**: 질의의 근본적인 이유와 해결하고자 하는 문제 정의.
	- 나는 뭘 하다가 막혀서 얘를 찾아왔나
- **명확성**: 요구사항을 구체적이고 명료하게 명세화.
	- 나의 요구사항 명세서: 
- **맥락 (Context)**: 요구사항을 이해하는 데 필요한 배경 정보 제공.
	- 나의 요구사항 명세서: 
- **형식 (Format)**: 원하는 출력의 구조나 틀을 지정.
- **피드백**: 이전 대화나 결과를 바탕으로 요구사항을 수정 및 보완.


### 3. Prompt Design Framework (Techniques)

#### 3-1. 역할/페르소나 (Role/Persona)
- "당신은 SI 개발부서에서 10년을 일한 시니어 웹 개발자입니다."
	- 임시질문: 신입이 기술교육을 받다가 번아웃이 와버렸어. 진도를 못 따라오고 실습 과제도 제때 제출하지 못하고 있네. 사수로써 어떻게 해야 할까?
	- [Supporting a burned-out new employee through training](https://claude.ai/share/e5f6c9c5-f5be-4c6b-9728-0de14bbea4cd)
- "당신은 시골 초등학교 분교의 교사입니다."
	- 임시질문: 담당 학급의 학생이 오늘 출석을 안 했어. 어떡하지?
	- [Student absence follow-up procedure](https://claude.ai/share/80cae52b-e4cd-45d3-bb00-3ac1b502c19b)
- "당신은 10년 이상의 제조 industry에서 센서 데이터로 공정 최적화를 분석한 데이터 분석가입니다."
	- 임시질문: 협력사의 반도체 공장에서 이번 분기 수급량을 턱 없이 낮은 수준으로밖에 메우지 못할 것 같다고 연락이 도착했어. 이제 두 달 남았는데, 팀장급인 나는 이 회사에서 어떻게 해야 할까?
	- [Semiconductor supply shortage crisis management](https://claude.ai/share/29e874cf-d8bb-453c-9b22-91011ed275f8)
- "당신은 기업교육 과정 설계 전문가입니다."
	- 임시질문: 당장 다음주 월요일에 SK AX 본사에 가서 미팅을 진행해야 해. 무지 긴장되어서 떨려.
	- [Preparing for SK AX headquarters meeting](https://claude.ai/share/8c2a99d1-4cbe-486c-bd5f-0b0500344a05)

#### 3-2. 청중 (Audience)

- "(초등학생에게) 횡단보도 교통 안전 수칙에 대해서 설명해 주세요."
	- [Crosswalk Safety Guideline](https://claude.ai/share/66ae265d-60a6-4466-bfb8-c617489c88c7)
- "최근 달리기를 시작한 초보 러너에게 달리기의 바른 자세에 대해서 설명해 주세요."
	- [Proper running form and technique](https://claude.ai/share/d5c09823-4167-42da-82bb-bb6c1c0a59c6)

#### 3-3. 지식/정보 (Knoweldge/Information)
- "(매슬로우의 욕구단계이론을 기반으로) 소비자의 행동을 분류해 주세요."
	- [Consumer behavior classification](https://claude.ai/share/0c3a7f92-bd46-48f9-b464-b3a201172833)
- "(켈러의 ARCS 동기 이론을 반영하여) 생성형 AI 특강 1시간 교육의 세부 활동을 설계해 주세요."
	- [Generative AI one-hour workshop design](https://claude.ai/share/f65016f3-5fe1-4c0d-957f-58db7d7079b7)

#### 3-4. 과업/목표 (Task/Goal)

- "나 다이어트 중인데 배고파."
- "나 다이어트 중인데 배고파. 날 말려줘."
- "나 다이어트 중인데 배고파. 날 말려줘. 아 근데 장도 봐야 해. 냉장고에 먹을 게 들어있지 않거든. 근데 장보러 가면 시식코너에서 잔뜩 집어먹겠지? 아 살 빼야 하는데... 배달을 시킬까. 나 근데 스마트폰을 안써. 냉장고 정리가 필요한데 또.. 해야 하는데.. 아배고파..."
- [Managing hunger while dieting](https://claude.ai/share/f44c68b1-49b3-430a-b561-140e479bb855)

- "생성형 AI 사이트 (찾아줘)"
	- [Generative AI platforms overview](https://claude.ai/share/26eddada-7102-4db8-bac8-1823e0ec88ea)

#### 3-5. 정책/규칙, 스타일, 제약 (Policy/Rule, Style, Constraints)

- 절연한 친구의 모바일청첩장 메시지에 대한 답장을 (배알이 꼴리지만 티는 내고싶지 않은 말투로) 만들어 줘. (문자메시지 유료 전환 안 되는 선에서.)
	- [절연한 친구 청첩장 답장 예시](https://gemini.google.com/share/b236b9cff5b1)

#### 3-6. 형식/구조 (Format/Structure)

- 지금까지의 대화를 바탕으로 나는 어떤 사람인 것 같은지 분석해서 (500자 이내의) (HTML5 형식으로) 알려줘. 
	- [사용자 대화 기반 프로필 분석](https://gemini.google.com/share/68c2e9d9d18e)

#### 3-7. 예시 (Examples/Shot-Learning)

- _입력과 출력의 예시를 제공하여 모델이 원하는 패턴을 학습하도록 유도_.
- **Zero|One|Few shot**
- **Zero-shot**: 예시 없이(Zero) 작업 지시.
- **One-shot**: 하나의(One) 예시를 제공하고 작업 지시.
- **Few-shot**: 두 개 이상의(Few) 예시를 제공하여 작업 지시.
- [문장 긍정/부정 분류](https://gemini.google.com/share/8c6bf4dcece4)
- [인용문에서 저자 추출](https://gemini.google.com/share/3fc626e366b7)


#### 3-8. 단계적 질의 (Concept/Step-by-Step)

- 복잡한 질문을 한 번에 요청하기보다, 맥락에 따라 여러 단계(_e.g., 1. 조사, 2. 분석, 3. 추천_)로 쪼개어 분석의 흐름을 유도.
- "오픈 소스 데이터베이스 중에 뭘 쓰는 게 가장 좋을까?"
	- [오픈 소스 데이터베이스 추천 가이드](https://gemini.google.com/share/2879c55ecb76)
- "오픈 소스 데이터베이스를 조사해서 각각의 특징을 분석해줘."
- "개인 프로젝트를 진행한다면 어떤 DB를 가장 추천하겠어?"
	- [오픈 소스 데이터베이스 특징 비교 분석](https://gemini.google.com/share/c9f5144d826c)


#### 3-9. 후카츠 프롬프트(Fukatsu Prompt)

```text
#명령문
당신은 20년 경력의 IT회사 SI개발부 수석이자 베테랑 개발자입니다.
기존의 정부 공공사업체의 레거시 환경을 쿠버네티스를 활용한 클러스터 환경으로 전환하는 사업에 참여합니다.
K8S 환경을 구축하기 위한 전반적인 로드맵을 기획하고자 합니다.
아래의 제약 조건과 입력문을 기반으로, 최상의 인터뷰 질의서를 만들어 주세요.

#제약조건
- 실제 실무에서 비슷한 프로젝트를 진행했을 때 어떻게 환경을 구성했는지를 위주로 분석합니다.
- 사용되는 모든 프로그램의 버전 간 호환성을 최우선으로 염두에 두어야 합니다.
- 물리적인 환경 역시 스펙을 따져 고려해야 합니다.
- 프로젝트에 참여하는 모든 팀원은 K8S를 깊이 있게 이해하고 있다는 전제를 깔아둡니다.
- 사업 기간을 1년으로 감안하여, 중장기적인 로드맵을 구체적으로 작성해야 합니다.

#출력형식
GitHub Markdown 문법을 따른다.

#입력문
- Tomcat과 Spring Boot 2.7.18 버전의 공공기관 레거시 프로젝트를 쿠버네티스 클러스터 환경으로 전환하려고 합니다.
- 공통 문서를 작성하는 데에 어려움이 많습니다.
- 작업을 도와주세요.
```
	- 뭔가 이상한 결과 도출됨. 동일 템플릿으로 '인터뷰 질의 예시'를 엄청 물어본 것 같다.
	- [K8S 전환 프로젝트 질의서](https://gemini.google.com/share/99095bbe68c8)
