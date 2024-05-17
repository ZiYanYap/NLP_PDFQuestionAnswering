## Streamlit ##
[https://nlp-pdf-question-answering-yapziyan.streamlit.app/](https://nlp-pdf-question-answering-yapziyan.streamlit.app/)

![image](https://github.com/ZiYanYap/NLP_PDFQuestionAnswering/assets/74231464/4426ac60-31ec-40a4-bc2b-67643a2b8f15)

## Evaluation of Different Embedding Models and Question-Answering (QA) Models
The evaluation tests cover 2 embedding models and 5 QA models, resulting in ten unique model combinations. Each combination is tested against 5 predefined questions and sample answers.
For each question, the response similarity and response time of each combination are recorded.

The response similarity will be based on the following metrics:
- Levenshtein Distance: Measures the number of single-character edits required to change the generated answer into the expected answer, providing a quantitative measure of response accuracy.
- Acceptance (Human-Like Performance, HLP): A binary metric assessing whether the generated response meets the quality standard of a reasonable human answer.

The response time will be based on the average time taken from the moment a user sends a query to the moment the application displays a response.


**Embedding models used:**
- sentence-transformers/all-mpnet-base-v2 (https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- Alibaba-NLP/gte-base-en-v1.5 (https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5)

**QA models used:**
- deepset/roberta-base-squad2 (https://huggingface.co/deepset/roberta-base-squad2)
- timpal0l/mdeberta-v3-base-squad2 (https://huggingface.co/timpal0l/mdeberta-v3-base-squad2)
- distilbert/distilbert-base-cased-distilled-squad (https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad)
- deepset/bert-large-uncased-whole-word-masking-squad2 (https://huggingface.co/deepset/bert-large-uncased-whole-word-masking-squad2)
- mrm8488/longformer-base-4096-finetuned-squadv2 (https://huggingface.co/mrm8488/longformer-base-4096-finetuned-squadv2)

## Summary of Results ##
**Response Quality**

![image](https://github.com/ZiYanYap/NLP_PDFQuestionAnswering/assets/74231464/696999b2-056c-46f2-ba26-68565d9d31f3)
![image](https://github.com/ZiYanYap/NLP_PDFQuestionAnswering/assets/74231464/32ff5199-019d-4682-8a74-08af73c00f58)

**Response Time**

![image](https://github.com/ZiYanYap/NLP_PDFQuestionAnswering/assets/74231464/5c47e0e3-612c-43a1-b78b-3611053d8371)
![image](https://github.com/ZiYanYap/NLP_PDFQuestionAnswering/assets/74231464/347bdf72-2883-45ca-8868-d83f782acd82)
