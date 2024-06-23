# I will be using this file for creating a table about the milestone of different LLM architectures.
# As this tables are not available readily on the web, I have to go through various reasearch papers and extract the information and make table out of it.

# So lets create the table for the milestone of different LLM architectures.

import pandas as pd

# Creating the table
data = [
    {
        'Model': 'Transformer',
        'Year': '2017',
        'Institution': 'Google',
        'Paper Name': 'Attention is All You Need',
        'Authors': 'Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin',
        'Abstract': "Attention is all you need is a groundbreaking paper that introduced the Transformer architecture, a neural network model for NLP tasks that relies solely on attention mechanisms to process input sequences. The paper's contributions have had a significant impact on the field of deep learning and have inspired further research and advancements in the field.The Transformer model has become one of the most widely used models in NLP and has been applied to a wide range of tasks, including machine translation, language modeling, question answering, and text summarization. The model's ability to capture long-term dependencies and contextual relationships between words makes it well-suited for many NLP tasks and has enabled significant improvements in performance on these tasks.The paper also introduced several techniques for reducing the computational complexity of the model, which have made it more feasible to use the model for longer input sequences and larger datasets.Overall, the Attention Is All You Need paper represents a significant milestone in the development of neural network models for NLP tasks and has paved the way for further advancements in the field."
    },
    {
        'Model' : 'GPT 1.0',
        'Year' : '2018',
        'Institution' : 'OpenAI',
        'Paper Name' : 'Improving Language Understanding by Generative Pre-training',

        'Authors' : 'Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever',
        'Abstract' : "The GPT-1 model was the first in a series of models developed by OpenAI that use a transformer architecture for natural language processing tasks. The model is pre-trained on a large corpus of text data and fine-tuned on specific tasks to achieve state-of-the-art performance on a wide range of NLP tasks. The GPT-1 model introduced several key innovations, including the use of unsupervised pre-training to learn general language representations and the use of a transformer architecture to capture long-range dependencies in text data. The model has been widely used in the NLP community and has inspired further research and advancements in the field. The GPT-1 model represents a significant milestone in the development of neural network models for NLP tasks and has paved the way for further advancements in the field."
    },
    {
        'Model' : 'BERT',
        'Year' : '2018',
        'Institution' : 'Google',
        'Paper Name' : 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
        'Authors' : 'Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova',
        'Abstract' : "BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained transformer model that has achieved state-of-the-art performance on a wide range of NLP tasks. The model is pre-trained on a large corpus of text data using a masked language modeling objective and fine-tuned on specific tasks to achieve high performance. BERT introduced several key innovations, including the use of bidirectional context to capture dependencies between words in text data and the use of transformer architecture to model long-range dependencies. The model has been widely used in the NLP community and has inspired further research and advancements in the field. BERT represents a significant milestone in the development of neural network models for NLP tasks and has paved the way for further advancements in the field."
    },
    {
        'Model' : 'GPT 2.0',
        'Year' : '2019',
        'Institution' : 'OpenAI',
        'Paper Name' : 'Language Models are Unsupervised Multitask Learners',
        'Authors' : 'Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever',
        'Abstract' : "The GPT-2 model is a large-scale transformer model developed by OpenAI that has achieved state-of-the-art performance on a wide range of NLP tasks. The model is pre-trained on a large corpus of text data using a transformer architecture and fine-tuned on specific tasks to achieve high performance. GPT-2 introduced several key innovations, including the use of unsupervised pre-training to learn general language representations and the use of a transformer architecture to capture long-range dependencies in text data. The model has been widely used in the NLP community and has inspired further research and advancements in the field. GPT-2 represents a significant milestone in the development of neural network models for NLP tasks and has paved the way for further advancements in the field."
    },
    {
        'Model' : 'T5',
        'Year' : '2019',
        'Institution' : 'Google',
        'Paper Name' : 'Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer',
        'Authors' : 'Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu',
        'Abstract' : "T5 (Text-to-Text Transfer Transformer) is a large-scale transformer model developed by Google that has achieved state-of-the-art performance on a wide range of NLP tasks. The model is pre-trained on a large corpus of text data using a text-to-text framework and fine-tuned on specific tasks to achieve high performance. T5 introduced several key innovations, including the use of a text-to-text framework to unify different NLP tasks and the use of a transformer architecture to capture long-range dependencies in text data. The model has been widely used in the NLP community and has inspired further research and advancements in the field. T5 represents a significant milestone in the development of neural network models for NLP tasks and has paved the way for further advancements in the field."
    },
    {
        'Model' : 'GPT 3.0',
        'Year' : '2020',
        'Institution' : 'OpenAI',
        'Paper Name' : 'Language Models are Few-Shot Learners',
        'Authors' : 'Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei',
        'Abstract' : "The GPT-3 model is a large-scale transformer model"
    },
    {
        'Model' : 'Megatron-LM',
        'Year' : '2019',
        'Institution' : 'NVIDIA',
        'Paper Name' : 'Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism',
        'Authors' : 'Shaden Smith, Mostofa Patwary , Brandon Norick , Patrick LeGresley , Samyam Rajbhandari, Jared Casper, Zhun Liu, Shrimai Prabhumoye, George Zerveas, Vijay Korthikanti, Elton Zhang, Rewon Child, Reza Yazdani Aminabadi, Julie Bernauer, XiaSong, Mohammad Shoeybi, Yuxiong He, Michael Houston, Saurabh Tiwary, and Bryan Catanzaro',
        'Abstract' : "Megatron-LM is a large-scale transformer model developed by NVIDIA that has achieved state-of-the-art performance on a wide range of NLP tasks. The model is trained using model parallelism, which allows for the efficient training of models with billions of parameters. Megatron-LM introduced several key innovations, including the use of model parallelism to scale up the size of transformer models and the use of a transformer architecture to capture long-range dependencies in text data. The model has been widely used in the NLP community and has inspired further research and advancements in the field. Megatron-LM represents a significant milestone in the development of neural network models for NLP tasks and has paved the way for further advancements in the field."
    },
    {
        'Model' : 'ZeRo',
        'Year' : '2019',
        'Institution' : 'Microsoft',
        'Paper Name' : 'ZeRo:  Memory Optimizations Toward Training Trillion Parameter Models',
        'Authors' : 'Samyam Rajbhandari,Jeff Rasley,Olatunji Ruwase, Yuxiong He',
        'Abstract' : "Large deep learning models offer significant accuracy gains, but training billions to trillions of parameters is challenging. Existing solutions such as data and model parallelisms exhibit fundamental limitations to fit these models into limited device memory, while obtaining computation, communication and development efficiency. We develop a novel solution, Zero Redundancy Optimizer (ZeRO), to optimize memory, vastly improving training speed while increasing the model size that can be efficiently trained. ZeRO eliminates memory redundancies in data- and model-parallel training while retaining low communication volume and high computational granularity, allowing us to scale the model size proportional to the number of devices with sustained high efficiency. Our analysis on memory requirements and communication volume demonstrates: ZeRO has the potential to scale beyond 1 Trillion parameters using today's hardware. We implement and evaluate ZeRO: it trains large models of over 100B parameter with super-linear speedup on 400 GPUs, achieving throughput of 15 Petaflops. This represents an 8x increase in model size and 10x increase in achievable performance over state-of-the-art. In terms of usability, ZeRO can train large models of up to 13B parameters (e.g., larger than Megatron GPT 8.3B and T5 11B) without requiring model parallelism which is harder for scientists to apply. Last but not the least, researchers have used the system breakthroughs of ZeRO to create the world's largest language model (17B parameters) with record breaking accuracy"
    },
    {
        'Model' : 'PaLM',
        'Year' : '2022',
        'Institution' : 'Google',
        'Paper Name' : 'PaLM: Scaling Language Modeling with Pathways',
        'Authors' : 'Aakanksha Chowdhery Sharan Narang Jacob Devlin Maarten Bosma Gaurav Mishra Adam Roberts Paul Barham Hyung Won Chung Charles Sutton Sebastian Gehrmann Parker Schuh Kensen Shi Sasha Tsvyashchenko Joshua Maynez Abhishek Rao† Parker Barnes Yi Tay Noam Shazeer‡ Vinodkumar Prabhakaran Emily Reif Nan Du Ben Hutchinson Reiner Pope James Bradbury Jacob Austin Michael Isard Guy Gur-Ari Pengcheng Yin Toju Duke Anselm Levskaya Sanjay Ghemawat Sunipa Dev Henryk Michalewski Xavier Garcia Vedant Misra Kevin Robinson Liam Fedus Denny Zhou Daphne Ippolito David Luan‡ Hyeontaek Lim Barret Zoph Alexander Spiridonov Ryan Sepassi David Dohan Shivani Agrawal Mark Omernick Andrew M. Dai Thanumalayan Sankaranarayana Pillai Marie Pellat Aitor Lewkowycz Erica Moreira Rewon Child Oleksandr Polozov† Katherine Lee Zongwei Zhou Xuezhi Wang Brennan Saeta Mark Diaz Orhan Firat Michele Catasta† Jason Wei Kathy Meier-Hellstern Douglas Eck Jeff Dean Slav Petrov Noah Fiedel',
        'Abstract' : "PaLM is a large-scale transformer model developed by Google that has achieved state-of-the-art performance on a wide range of NLP tasks. The model is trained using a novel architecture called Pathways, which allows for the efficient training of models with trillions of parameters. PaLM introduced several key innovations, including the use of Pathways to scale up the size of transformer models and the use of a transformer architecture to capture long-range dependencies in text data. The model has been widely used in the NLP community and has inspired further research and advancements in the field. PaLM represents a significant milestone in the development of neural network models for NLP tasks and has paved the way for further advancements in the field."
    },
    {
        'Model' : 'Megatron-Turing NLG',
        'Year' : '2021',
        'Institution' : 'Microsoft',
        'Paper Name' : 'Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model',
        'Authors' : 'Yuxiong He, Edouard Grave, Sainbayar Sukhbaatar, Zhiheng Huang, Mostofa Patwary, Weizhu Chen, Arul Menezes, Hao Wu, Ke Gong, Jamie Kiros, Licheng Yu, Kaushik Subramanian, James Hale, Sheng Zhao, Rami Al-Rfou, Alexander Huth, Jasha Droppo, Adam Fourney, Jason Weston',
        'Abstract' : "Pretrained general-purpose language models can achieve state-of-the-art accuracies in various natural language processing domains by adapting to downstream tasks via zero-shot, few-shot and finetuning techniques. Because of their success, the size of these models has increased rapidly, requiring high-performance hardware, software, and algorithmic techniques to enable training such large models. As the result of a joint effort between Microsoft and NVIDIA, we present details on the training of the largest monolithic transformer based language model, Megatron-Turing NLG 530B (MT-NLG), with 530 billion parameters. In this paper, we first focus on the infrastructure as well as the 3D parallelism methodology used to train this model using Deep Speed and Megatron. Next, we detail the training process, the design of our training corpus, and our data curation techniques, which we believe is a key ingredient to the success of the model. Finally, we discuss various evaluation results, as well as other interesting observations and new properties exhibited by MT-NLG. We demonstrate that MT-NLG achieves superior zero-, one-, and few-shot learning accuracies on several NLP benchmarks and establishes new state-of-the-art results. We believe that our contributions will help further the development of large-scale training infrastructures, large-scale language models, and natural language generations."
    }
]

data_frame = pd.DataFrame(data)
data_frame.to_csv('Data/llm_milestone.csv', index=False)
print(data_frame.head())