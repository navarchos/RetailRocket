Скачать данные:

pip install kaggle

kaggle datasets download -d retailrocket/ecommerce-dataset -p ./data/raw --unzip

_________________________________________________________________

session based gru model (item embeddings only) full softmax):

Epoch 15 | Loss: 1.5169 | val Recall@20: 0.2851 | val MRR@20: 0.1761

session based gru model (item + events embeddings) full softmax:

Epoch 15 | Loss: 1.1413 | Recall@20: 0.2866 | MRR@20: 0.1745
