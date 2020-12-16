import torch
from transformers import BertForSequenceClassification, BertTokenizer
from torchvision.models import resnet18


class VQAModel(torch.nn.Module):
    """
      A class describing a model used to solve the Visual Question Answer task.
      This model is specialized on (yes/no) questions with number of output neurons of 2.
      There are 3 major components :
      A text embedding model   :  based on BertModel implementation from the transformers library.
                                  The model gives a representation for each word in the question.
                                  It is followed by a linear projection module to project the word representations
                                  to the same dimension as the image embedding.
                                  A question embedding is then obtained via an LSTM (the last hidden output)
      An image embedding model :  based on resnet18 implementation from the torchvision library.
                                  The representation of the image is given by a fully connected layer that projects
                                  the image to the same embedding_size dimension as the text embedding
      A classification head    :  takes a vector resulting from the concatenation of the question and image embeddings
                                  It consists of a MLP with linear layers followed by ReLU activation function and dropout layers.

    """

    def __init__(self, embedding_size, hidden_size_ch, d_out, dropout, device):
        super(VQAModel, self).__init__()
        self.text_model = BertForSequenceClassification.from_pretrained('bert-base-cased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.image_model = resnet18(pretrained=True)

        self.freeze_model()

        self.hidden_size_text = self.text_model.classifier.in_features
        self.hidden_size_image = self.image_model.fc.in_features

        self.embedding_size = embedding_size

        self.text_model.classifier = torch.nn.Linear(self.hidden_size_text, self.embedding_size, bias=True)
        self.image_model.fc = torch.nn.Linear(self.hidden_size_image, self.embedding_size, bias=True)

        #self.rnn = torch.nn.LSTM(input_size=self.embedding_size, hidden_size=self.embedding_size, batch_first=True)

        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(2 * self.embedding_size, hidden_size_ch),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size_ch, hidden_size_ch),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size_ch, d_out)
        )

        self.device = device

    def forward(self, image, question, initial_hidden):
        """
                   image : [batch_size, 3, 224, 224]
                question : [batch_size, text_len]
          initial_hidden : [embedding_size]
        """

        enc_question = self.tokenizer(list(question), add_special_tokens=True, padding=True, truncation=True,
                                      return_tensors="pt")  # dictionnary {'input_ids': ..., 'attention_mask': ...}
        input_ids = enc_question['input_ids'].to(self.device)
        attention_mask = enc_question['attention_mask'].to(self.device)

        output = self.text_model(input_ids, attention_mask)  # a tuple (last_hidden_state, pooler_output)

        # [batch_size, embedding_size]
        embedded_question = output.logits #self.text_projection(last_hidden_state)  # [batch_size, seq_len, embedding_size]

        # [batch_size, embedding_size]
        embedded_image = self.image_model(image)  # [batch_size, embedding_size]

        embedding = torch.cat((embedded_question, embedded_image), dim=1)  # [batch_size, 2 * embedding_size]

        output = self.classification_head(embedding)  # [batch_size, d_out]

        # [batch_size, d_out]
        return output
    
    def freeze_model(self):
        #for param in self.image_model.parameters():
        #    param.requires_grad = False
        
        for param in self.text_model.parameters():
            param.requires_grad = False