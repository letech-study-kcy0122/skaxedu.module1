## Need
- 교육 인증사진 필요


## 실습환경
- 호스트OS: Windows 10 Enterprise 22H2 (19045.6093)
- Music AI SUNO 무료계정으로 음악 생성
- OpenAI Tokenizer로 토큰 확인
- Prompt 실습: Claude Free (Sonnet 4.5), Gemini Pro (2.5 Flash), OpenRouter; Nemotron-H3 (Nemotron-H Nano)

---

## Lab 1. 생성형 AI에 대한 이해
- 진단평가, 출결 진행
- 교안, 프롬프트 텍스트 다운로드


### Introduct.
- LLM이 하는 일은 무엇인가?
- 자연어(사람이 사용하는 언어)를 컴퓨터가 이해할 수 있는 숫자(bit)로 변환하는 작업이 필요하다. (== **Embedding** 과정.)
- 수많은 단어를 컴퓨터가 이해하고 있어야, 그 다음 논리적 연산 처리가 가능하지 않겠나.
- 모델이 만들어진 이후부터는 지속적으로 성능이 하락하게 된다.
- 왜? 모델은 이전에 학습한 데이터를 기반으로 컨텐츠를 생성할 수 있기 때문이다.
- 시간이 흐르며 생기는 수많은 데이터들도 모델이 참고할 수 있다면, _이 모델이 최신화되었다_고 말할 수 있을까?
- 회사 내부 정보(기밀)에는 접근 불가, Halucination 경계
- 사용되는 파라미터의 수가 많으면(최소 30bil+) LLM, 상대적으로 수가 적으면(+-7bil) sLLM.
- 외부지식을 허용(직접 검색하게)하면 RAG, 모델에 직접 추가 학습시키는 방식을 쓰면 (Fine-Tuning)
- 언어가 공손한지/무례한지에 따라 응답 품질에 영향을 준다. (교육에서는 예의바른 언어가 순응답을 한다고 말하나, 저널에서는 생성형AI의 자존심을 긁을 때 품질이 올라간다는 말도 하더라.)


### AI와 생성형 AI
- 인공 지능(AI)는 '기계를 인간과 더 유사하게 만든다'라는 광범위한 개념.
- 생성형 AI는 '새로운 컨텐츠를 의미 있고 지능적으로 생성'하는 하위 분야

#### 구조
- **Prompt**: GTP 대화창
- **Foudation Model**: 생성형 AI의 FM으로는 LLM(Large Language Model)을 사용한다.
	- FM은 일반화되고 레이블이 없는 광범위한 데이터를 기반으로 훈련된 MachineLearning 모델이다.
	- 해당 모델은 질문에 답하고, 에세이를 작성하며, 이미지에 캡션을 다는 등 다양한 일반적인 작업을 수행할 수 있다.
	- 대규모 언어 모델(LLM)은 요약, 텍스트 생성, 분류, 개방형 대화, 정보 추출 등의 **언어 기반 작업**에 특화되어 있다.
- 새로운 컨텐츠를 생성해내는 것.

> [AWS - What is Generative-AI](https://aws.amazon.com/ko/what-is/generative-ai/)


### AGI(Artificial General Intelligence)
- AGI is a field of theoretical AI research that _attempts to create software with human-like Intelligence and the ability to self-teach.
- The aim is for the software to be able to perform tasks that **it is not necessarily trained or developed for**.

> [AWS - What is Artifical-General-Intelicence](https://aws.amazon.com/what-is/artificial-general-intelligence/?nc1=h_ls)


### Gartner's Hype Cycle

> [Gartner Hype Cycle 2025](https://www.gartner.com/en/articles/hype-cycle-for-emerging-technologies)



### 생성형 AI의 활용
- Biz. Trend
- 리뷰 요약
- Multi-Modal AI 영상분석
- [Music AI - facebook/melodyflow](https://huggingface.co/facebook/melodyflow-t24-30secs) 
- [Music AI - SUNO](https://suno.com/home)


---

## Lab 3.


#### Fine-Tuning
- 모델에 신규 데이터를 추가 학습시켜서, 모델을 변경하는 것 (e.g. 버전 업)


#### RAG
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


### NLP(Natural Language Processing)
- 자연어 처리(NLP)는 ML을 사용하여 **컴퓨터가 인간의 언어를 이해하고 소통하도록 돕는 인공 지능(AI)의 하위 분야**.

> [IBM - NLP란 무엇인가요?](https://www.ibm.com/kr-ko/think/topics/natural-language-processing)
> [Oracle - 자연어 처리(NLP)란?](https://www.oracle.com/kr/artificial-intelligence/natural-language-processing/)
> [DeepLearning.AI - Natural Language Processing](https://www.deeplearning.ai/resources/natural-language-processing/)


---

### Bag of Words(Bow) 빈도 기반 언어 표현(e.g., BoW Vector)
- 단어의 **빈도**에 초점을 맞추어 _문서나 문장을 벡터로 변환_한다.
- 단어의 순서나 문맥적 의미를 파악하지 못하는 한계가 있다.

> [codong - 3. 카운트 기반의 단어 표현](https://codong.tistory.com/34)
> [glowdp - 카운트 기반 언어모델](https://glowdp.tistory.com/142)


### 의미적 추론, 분포 가설의 구현, Word Embedding
- 단어를 밀집된(dense) 실수(Real Number) 벡터로 표현하며, 단어 간 의미적 관계(문맥)을 포착한다.
> [glowdp - 의미기반 언어 지식 표현 체계](https://glowdp.tistory.com/145)


#### 단어의 분산 표현(Distributional Representation)
- 자연어를 Embedding 과정을 거쳐(숫자로 고쳐, bit로 바꾸어서) Vector로 표현하는 것.


#### 분포 가설(Distributional Hypothesis)
- 단어의 의미는 주변 단어에 의해 형성된다는 가설을 말한다.
- **인접한 단어들은 그 의미도 유사할 것이라는 전제**

> [Github.QA](https://github.com/boost-devs/peer-session/issues/62)
> [hezma - 8. Distributional Representation of Words](https://hezma.tistory.com/102)


#### Word Embedding(e.g., Word2Vec)


### What is Language Model?
- 문장을 보고, '단어 시퀀스를 찾아내서', 이게 수학적(확률 상)으로 "얼마나 그럴 듯 하냐" 따져주는 모델


### Transformer Architecture(Embedding, Encoder, Decoder)
- 2017년 Google이 발표한 "Attention Is All You Need" 논문에서 소개된 딥러닝 모델 아키텍처.
- 기존의 RNN(Recurrent Neural Network) 계열 모델의 한계를 극복, 자연어 처리(NLP) 분야에 혁신을 가져왔다.

- Transformer는 입력된 텍스트를 처리하고, 그 의미를 파악하며, 새로운 텍스트를 생성하는 데 핵심적인 역할을 한다.
- 이러한 과정은 단어를 '숫자 벡터'로 표현하는 방식에 의존한다.
	- Transformer는 입력 정보를 **Word Embedding**을 사용하여 단어의 의미적, 문맥적 정보를 벡터 형태로 받아들이고,
	- 이를 **Self-Attention** 메커니즘을 통해 효과적으로 처리한다.

- Transformer는 세 가지 구성 요소로 이루어진다.
	1) Embedding
		- 입력 텍스트의 각 단어(Token)를 **의미 정보를 함축한** 벡터로 변환한다.
		- Word Embedding: 단어의 의미를 담음
		- Positional Embedding: 단어의 위치 정보를 담음.
		  이것으로 문장 내 단어의 순서 정보를 활용하며, 순서를 무시하는 Self-Attention 메커니즘의 한계를 보완한다.
	2) Encoder
		- 생성된 문맥 정보를 포착하고 요약한다.
		- Self-Attention: 입력 시퀀스의 각 단어가 해당 시퀀스 내의 다른 모든 단어와 얼마나 관련이 있는지 계산한다.
		  그로써 단어의 문맥적 의미를 강화한 벡터 표현(Context Vector)를 생성한다.
	3) Decoder
		- 생성된 문맥 벡터를 바탕으로 **새로운 출력 시퀀스**를 한 단어씩 생성한다.
		- Self-Attention: 이미 생성된 단어들 사이의 관계를 파악한다.
		- Cross-Attention: Encoder가 학습한 입력 문맥에 가장 적합한 다음 단어를 예측하도록 지속해서 Attention을 유지한다.
		
> [glowdp - Encoder Model(BERT)](https://glowdp.tistory.com/134)
> [glowdp - Decoder Model(GPT)](https://glowdp.tistory.com/135)
> [glowdp - Encoder-Decoder Model(BART)](https://glowdp.tistory.com/136)
> [@hubmanskj - LLM이란](https://medium.com/@hugmanskj/%EA%B1%B0%EB%8C%80-%EC%96%B8%EC%96%B4-%EB%AA%A8%EB%8D%B8-large-language-model-%EC%97%90-%EB%8C%80%ED%95%9C-%EC%9D%B4%ED%95%B4-%EC%96%B8%EC%96%B4%EB%AA%A8%EB%8D%B8%EC%9D%B4%EB%9E%80-d52a120efe52)
> [@hubmanskj - LLM 기준과 특징](https://medium.com/@hugmanskj/%EA%B1%B0%EB%8C%80-%EC%96%B8%EC%96%B4-%EB%AA%A8%EB%8D%B8-large-language-model-%EC%97%90-%EB%8C%80%ED%95%9C-%EC%9D%B4%ED%95%B4-llm%EC%9D%98-%EA%B8%B0%EC%A4%80%EA%B3%BC-%ED%8A%B9%EC%A7%95%EC%9D%80-%EB%AC%B4%EC%97%87%EC%9D%BC%EA%B9%8C-0551b7b9d3bd)
> [seanpark11 - LLM의 학습 방법론](https://seanpark11.tistory.com/165)


---
+ 11:30 점심식사 시작 -> 13:00

## Lab 4.


### Tokenizer
- 예전 버전의 LLM기반 AI는 한국어 토큰을 잘 구분하지 못했다.
- 한글은 구조적으로 한자도 많고, 어휘를 의미적으로 파악하기가 어렵다.

> [OpenAI - Tokenizer](https://platform.openai.com/tokenizer)


### Prompt Engineering
- LLM이 정확한 결과를 도출하도록 명확한 자연어 문장을 뽑아내는 작업.
- 문장을 오해의 소지 없도록 작성하고, 요구사항을 A to Z까지 녹여내는 것


#### LLM 출력 구성
- LLM 출력은, 생성된 단어 다음에 올 토큰을 확률을 통해 결정하고 그 과정을 반복하는 것.

##### Output Length (Max Token)
- 최대 한도가(LLM 제한) 있어서 출력 길이가 제한을 넘어서면 토큰 예측을 중단함.

##### Temperature
- 토큰 선택의 무작위성을 제어
- 확률이 높은 토큰만 선택하도록 하면, 천편일률적인 답변만 받게 될 것이다.
- 또다른 척도로 '무작위성'을 부여한다.
	- 온도가 높으면 활동성이 높아진다 -> 창의적
	- 온도가 낮으면 춥다 -> 경직됨 -> 확률이 높은 답변만 선택함
- 온도가 높아질 수록 토큰 간의 확률이 비슷해진다. 뭘 선택할지 궁금해진다.

- **Top-K**: 예측 토큰의 확률로 순위 매겨서 상위 <Top-K>개 안에서만 선택한다.
- **Top-p**: 예측 토큰의 확률로 백분위 분포를 따지고, 상위 <Top-p>% 안에 속하는 토큰만 선택한다.
	- 대입 조건의 Top-p가 10%다. == 1등급, 2등급 안에서만 입학생을 고르겠다.

##### User/System Prompt
- **User Prompt**: 대화창에서 사용자가 FM에게 전달하는 입력
- **System Prompt**: 관리자(개발자)가 초기에 FM에 설정한 지침

###### Prompt 구성 5원칙
- **목적**
	- 나는 뭘 하다가 막혀서 얘를 찾아왔나
- **명확성**
	- 나의 요구사항 명세서: 
- **맥락**
	- 나의 요구사항 명세서: 
- **형식**
	- 원하는 틀에 끼워맞춘 대답
- **피드백**
	- 이전 대화를 베이스로 요구사항 추가

##### Prompt Design Framework (Techniques)
1) Role/Persona
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
2) Audience
	- "(초등학생에게) 횡단보도 교통 안전 수칙에 대해서 설명해 주세요."
		- [Crosswalk Safety Guideline](https://claude.ai/share/66ae265d-60a6-4466-bfb8-c617489c88c7)
	- "최근 달리기를 시작한 초보 러너에게 달리기의 바른 자세에 대해서 설명해 주세요."
		- [Proper running form and technique](https://claude.ai/share/d5c09823-4167-42da-82bb-bb6c1c0a59c6)
3) Knoweldge/Information
	- "(매슬로우의 욕구단계이론을 기반으로) 소비자의 행동을 분류해 주세요."
		- [Consumer behavior classification](https://claude.ai/share/0c3a7f92-bd46-48f9-b464-b3a201172833)
	- "(켈러의 ARCS 동기 이론을 반영하여) 생성형 AI 특강 1시간 교육의 세부 활동을 설계해 주세요."
		- [Generative AI one-hour workshop design](https://claude.ai/share/f65016f3-5fe1-4c0d-957f-58db7d7079b7)
4) Task/Goal
	- "나 다이어트 중인데 배고파."
	- "나 다이어트 중인데 배고파. 날 말려줘."
	- "나 다이어트 중인데 배고파. 날 말려줘. 아 근데 장도 봐야 해. 냉장고에 먹을 게 들어있지 않거든. 근데 장보러 가면 시식코너에서 잔뜩 집어먹겠지? 아 살 빼야 하는데... 배달을 시킬까. 나 근데 스마트폰을 안써. 냉장고 정리가 필요한데 또.. 해야 하는데.. 아배고파..."
	- [Managing hunger while dieting](https://claude.ai/share/f44c68b1-49b3-430a-b561-140e479bb855)
	
	- "생성형 AI 사이트 (찾아줘)"
		- [Generative AI platforms overview](https://claude.ai/share/26eddada-7102-4db8-bac8-1823e0ec88ea)
5) Policy/Rule, Style, Constraints
	- 절연한 친구의 모바일청첩장 메시지에 대한 답장을 (배알이 꼴리지만 티는 내고싶지 않은 말투로) 만들어 줘. (문자메시지 유료 전환 안 되는 선에서.)
		- [절연한 친구 청첩장 답장 예시](https://gemini.google.com/share/b236b9cff5b1)
6) Format/Structure
	- 지금까지의 대화를 바탕으로 나는 어떤 사람인 것 같은지 분석해서 (500자 이내의) (HTML5 형식으로) 알려줘. 
		- [사용자 대화 기반 프로필 분석](https://gemini.google.com/share/68c2e9d9d18e)
7) Examples
	- [인용문에서 저자 추출](https://gemini.google.com/share/3fc626e366b7)
0) 후카츠 프롬프트(Fukatsu Prompt)
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

- **Zero|One|Few shot**
	- 예시를 전혀 안 들어줬다. == Zero shot
	- 예시를 하나 들었다. == One shot
	- 예시를 두 개 이상 들었다. == Few shot
	- [문장 긍정/부정 분류](https://gemini.google.com/share/8c6bf4dcece4)

- **System & Role**
	- **System Prompting**: 일반적으로는 FM 초기 세팅에서. GPT의 프로젝트나 개인맞춤 지침 정도.
	- **Role Prompting**: 특정 상황에서의 역할 수행 지시.

- **Concept**
	- 한 질문을 복잡하게 만들기보단, 맥락에 따라 쪼개서 분석의 흐름(단계)를 지정해 생성형AI를 이끌어가자.
	- "오픈 소스 데이터베이스 중에 뭘 쓰는 게 가장 좋을까?"
		- [오픈 소스 데이터베이스 추천 가이드](https://gemini.google.com/share/2879c55ecb76)
	- "오픈 소스 데이터베이스를 조사해서 각각의 특징을 분석해줘."
	- "개인 프로젝트를 진행한다면 어떤 DB를 가장 추천하겠어?"
		- [오픈 소스 데이터베이스 특징 비교 분석](https://gemini.google.com/share/c9f5144d826c)


























