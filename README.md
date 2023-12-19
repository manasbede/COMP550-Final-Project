# Medical Abstracts text classification using CNN-LSTM with Attention mechanism
This project investigates the efficacy of pretrained BERT and a novel LSTM-CNN with Attention model in tackling text classification using the Ohsumed dataset. The primary objective is to evaluate the performance of these models across various metrics including accuracy, CPU and memory utilization, and their ability to handle long-range dependencies. BERT excels in capturing long-range dependencies but falls short in local attention, while the proposed LSTM-CNN with Attention model leverages LSTM and Attention for long-range dependencies and CNN for local attention. Thus, the hypothesis posits that the LSTM-CNN with Attention model can achieve comparable accuracy to BERT while operating with reduced computational resources. This shift enables exploration of alternative, resource-efficient approaches to identify the most optimal solution. The outcomes of this comparative analysis promise valuable insights for natural language processing (NLP) and medical text classification, enriching these domains with practical findings.

### Project files
1. `prepare_dataset.py` : Code to convert the ohsumed dataset into required Train and Test data format.
2. `preprocessdata.py` : Code to preprocess the data, we applied techniques like stop word removal, removal of white spaces and punctuation and lemmatization.
3. `dataset_visualization.py` :  Code for dataset visualizations to study words distribution in every sample and distribution of classes in the dataset.
4. `LSTM.py` : This file contains implementation for LSTM and LSTM with GLOVE word embedding model.
5. `LSTM_Attention.py` : Code for LSTM with Attention and LSTM with Attention and word embedding model.
6. `LSTM_CNN.py` : Code for LSTM CNN model with word embedding.
7. `BiLSTM_Attention.py`  : Code for BiLSTM with Attention and BiLSTM with Attention and word embedding models.
8. `BiLSTM_CNN_Attention_Glove.py` :  Code for BiLSTM-CNN with Attention and Glove word embedding model.
9. `BiLSTM_CNN_Attention_Word2Vec.py` : Code for BiLSTM-CNN with Attention and Word2Vec word embedding model.
10. `Utilization.py` :  Code to calculate CPU and RAM utilization for every model.
11. `evaluation_on_long_text.py`  : Code to evaluate performance of above-mentioned models on long texts.
12. `cnn-lstm.py`  : Code to implement different architectures to combine results of LSTM and CNN models.
13. `bert.py` : Code for pretrained bert model with and without chunking.

The exploration into text classification using the Ohsumed dataset showcased compelling outcomes. Integrating BiLSTM with Attention mechanisms effectively tackled the vanishing gradient issue, notably boosting model performance. The addition of CNN acted as a local attention layer, capturing intricate patterns in the data. Furthermore, Self-Attention aided in comprehending global text semantics, resulting in a holistic understanding.
Employing pretrained Word Embeddings optimized data representation for Neural Network models, markedly enhancing performance. Remarkably, the CNN-LSTM model, incorporating attention and word embeddings, rivaled pretrained BERT models in text classification tasks, offering a resource-efficient alternative.This study prioritized evaluating computational resource utilization. The CNN-LSTM model with attention and glove word embeddings notably consumed fewer CPU and RAM resources than the intensive BERT model. The practical advantage of the LSTM variant over fine-tuning BERT models becomes evident in resource-constrained environments.In analyzing longer texts, the LSTM variant outperformed the BERT model, achieving 45.13% accuracy. Tailoring these models to specific tasks remains crucial due to variations in performance across datasets. The study emphasizes the importance of combining LSTM, CNN, and attention mechanisms to handle long-range dependencies while capturing word semantics contextually.
This study substantiates the efficacy of the LSTM-CNN Attention variant model in achieving comparable accuracy to state-of-the-art models while consuming fewer computational resources, specifically tailored for the Ohsumed dataset. The amalgamation of LSTM and CNN with attention mechanisms presents a promising avenue for text classification tasks, emphasizing the importance of model architecture selection and fine-tuning for optimal performance across diverse datasets and tasks.


