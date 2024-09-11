
# Данные  
Ниже приведено подробное описание разнообразных наборов данных,   
которые мы предлагаем использовать для инструктивного дообучения мультимодальной модели под задачи, схожие с соревнованием.  
  
  
Основная задача мультимодального supervised fine-tuning - заложить в модели способность не только извлекать полезную информацию  
из визуальной и аудио модальностей, но и умение применять эту информацию для решения более комплексных задач, как то:   
ведение мультимодального диалога, развернутый ответ на вопросы по видео и детальное нарративное описание видеозаписи.  
  
  
Поэтому предлагаемые наборы данных мы разделили на две категории:  
  
* Video Instruction Tuning Datasets  
* Video Conversation Datasets  
  
## Video Instruction Tuning Datasets  
  
### VideoChat  
  
#### Описание  
Это инструктивный видео-центричный мультимодальный набор данных, основанный на видео из датасета WebVid-10M.   
Он содержит детализированные последовательные описания видеозаписей и диалоги на их основе,   
полученные с помощью ChatGPT из покадровых описаний.   
  
#### Примеры  
<div align="center">  
  <img src="https://github.com/OpenGVLab/InternVideo/blob/main/Data/instruction_data/assert/conversation.png?raw=true" alt="examples" width="600">  
</div>  
<div align="center">  
  <img src="https://github.com/OpenGVLab/InternVideo/blob/main/Data/instruction_data/assert/detailed_description.png?raw=true" alt="examples" width="600">  
</div>  
  
  
📄 <a href="https://arxiv.org/abs/2305.06355" style="color: black; text-decoration: bold;"> Статья </a>      
🗃️ <a href="https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data" style="color: black; text-decoration: bold"> Датасет </a>   
  
### VideoChat2  
  
#### Описание  
Набор данных, состоящий из 2 млн. пар изображение-текст и видео-текст из разнообразных источников.   
VideoChat2 объединил различные датасеты в единый формат, отфильтровал изначальные некачественные пары,   
сбалансировал в рамках 6-ти задач: мультимодальный диалог (Conversation), классификация (Classification),   
детальное (Detailed Caption) и краткое (Simple Caption) описание изображения/видео, мультимодальное рассуждение   
(Reasoning) и визуальный QA (VQA).  
  
#### Примеры  
<div align="center">  
  <img src="https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/assert/data.png?raw=true" alt="examples" width="600">  
</div>  
  
📄 <a href="https://arxiv.org/abs/2311.17005" style="color: black; text-decoration: bold;"> Статья </a>      
🗃️ <a href="https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/DATA.md" style="color: black; text-decoration: bold"> Датасет </a>   
  
### Video-ChatGPT VideoInstruct100K  
  
#### Описание  
Набор данных, состоящий из 100 000 пар видео-инструкция, которые были получены путем комбинации полуавтоматической и человеческой разметки. Каждая пара состоит из видеозаписи и соответствующей инструкции в виде вопрос-ответ (QA). Покрывает такие задачи как:   
 - видео саммаризация;  
 - вопросы/ответы, основанные на детальном описании видеозаписи;  
 - генеративные вопросы/ответы;  
В качестве источников видео и кратких описаний использовался сабсет ActivityNet-200 датасета.  
  
  
#### Примеры  
  
<div align="center">  
  <img src="https://raw.githubusercontent.com/zhangtemplar/zhangtemplar.github.io/master/uPic/maazVideoChatGPTDetailedVideo2023-5-x102-y384.png" alt="examples" width="600">  
</div>  
  
📄 <a href="https://arxiv.org/abs/2306.05424" style="color: black; text-decoration: bold;"> Статья </a>      
🗃️ <a href="https://github.com/mbzuai-oryx/Video-ChatGPT?tab=readme-ov-file" style="color: black; text-decoration: bold"> Датасет </a>   
  
### VCG+  
  
#### Описание  
Набор данных, который основывается на Video-ChatGPT VideoInstruct.   
Этот датасет улучшает качество разметки путем фильтрации шума и нерелевантных деталей.  
  
#### Примеры  
  
<div align="center">  
  <img src="assets/video_annotation_pipeline.png" alt="examples" width="800">  
</div>  
  
📄 <a href="https://arxiv.org/abs/2406.09418" style="color: black; text-decoration: bold;"> Статья </a>      
🗃️ <a href="https://huggingface.co/datasets/MBZUAI/video_annotation_pipeline" style="color: black; text-decoration: bold"> Датасет </a>  
  
## Video Conversation Datasets  
  
### ActivityNet-QA   
  
#### Описание  
  
Набор данных, который содержит 58 000 вопросно-ответных пар, аннотированных вручную людьми. Вопросы основываются на 5800 видеозаписях, полученных из популярного набора данных ActivityNet и описаны в генеративной open-ended (OE) постановке. В разрезе сплитов элементы набора данных распределяются следующим образом:   
* Обучающий сплит: 32,000 QA пар на 3,200 видео  
* Валидационный сплит: 18,000 QA пар на 1,800 видео  
* Тестовый сплит: 8,000 QA пар на 800 видео  
  
  
#### Примеры  
<div align="center">  
  <img src="https://production-media.paperswithcode.com/datasets/Screenshot_2021-01-29_at_10.24.12.png" alt="examples" width="400">  
</div>  
  
📄 <a href="https://arxiv.org/abs/1906.02467" style="color: black; text-decoration: bold;"> Статья </a>      
🗃️ <a href="https://github.com/MILVLG/activitynet-qa" style="color: black; text-decoration: bold"> Датасет </a>   
  
### NExT-QA и NExT-GQA  
  
#### Описание  
NExT-QA содержит 5440 видео и порядка 52 тыс. вручную размеченных вопросно-ответных пар, сгруппированных по трем типам: обычные, временные и описательные вопросы.  
  
<div align="center">  
  <img src="https://github.com/doc-doc/NExT-QA/blob/main/images/res-mc-oe.png?raw=true" alt="examples" width="600">  
</div>  
  
NExT-GQA (short for Grounded) датасет расширяет разметку NExT-QA набора данных через добавление 10.5 тыс. временных меток начала и конца ответа на видеозаписи для соответствующей пары вопрос/ответ. Эти метки размечены вручную.   
  
<div align="center">  
  <img src="https://github.com/doc-doc/NExT-GQA/blob/main/misc/res.png?raw=true" alt="examples" width="600">  
</div>  
  
  
📄 <a href="https://arxiv.org/abs/2105.08276" style="color: black; text-decoration: bold;"> Статья NExT-QA </a>    
🗃️ <a href="https://github.com/doc-doc/NExT-QA" style="color: black; text-decoration: bold"> NExT-QA </a>   
  
📄 <a href="https://arxiv.org/abs/2309.01327" style="color: black; text-decoration: bold;"> Статья NExT-GQA </a>
🗃️ <a href="https://github.com/doc-doc/NExT-GQA" style="color: black; text-decoration: bold"> NExT-GQA </a>   
  
### STAR  
  
#### Описание  
Набор данных, который состоит из 60 тыс. пар вопрос-ответ, 240 тыс. дополнительных вариантов ответов,   
144 тыс. графов ситуаций, структурирующих информацию об объектах и их взаимодействиях на видео,   
а также 22 тыс. обрезанных видеозаписей. Видео в датасете заимствованы из видеозаписей человеческой активности,   
на которых изображены процессы взаимодействия человека и окружающей среды в разнообразных повседневных сценах.   
Созданные вопросы к этим видеозаписям разделяются на 4 категории, которых охватывают разные навыки модели   
понимания видео модальности.  
  
#### Примеры  
<div align="center">  
  <img src="https://github.com/csbobby/STAR_Benchmark/blob/main/img/NeurIPS2021_star_teaser.png?raw=true" alt="examples" width="600">  
</div>  
  
📄 <a href="https://arxiv.org/abs/2405.09711" style="color: black; text-decoration: bold;"> Статья </a>  
🗃️ <a href="https://bobbywu.com/STAR" style="color: black; text-decoration: bold"> Датасет </a>
