# to import pre-trained model 
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm 

from sklearn.metrics.pairwise import cosine_similarity

class ModelLoader(nn.Module):
    def __init__(self, modelName):
        super(ModelLoader, self).__init__()  # AttributeError: cannot assign module before Module.__init__() call
        self.modelName = modelName
        if self.modelName == "SentenceBert":
            self.tokenizer= AutoTokenizer.from_pretrained('model')     # Note: if you wanna Load model from HuggingFace Hub, set the value of argument as following:
            self.model = AutoModel.from_pretrained('model')            # 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        elif self.modelName == "SimCSE":                                                                                           
            self.tokenizer = AutoTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese')      # Load model from HuggingFace Hub
            self.model =AutoModelForMaskedLM.from_pretrained('IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese')

        else :
            raise Exception("only two type model are supported , they are SentenceBert and SimCSE, try SentenceBert or SimCSE modelName")
    
    def cosineScore(self, rowTensorA, rowtensorB):  # row vector only one dimention
        Avec = rowTensorA.detach().numpy().squeeze()
        Bvec = rowtensorB.detach().numpy().squeeze()
        similarity_socre = Avec.dot( Bvec ) / ( np.linalg.norm(Avec) * np.linalg.norm(Bvec))
        return similarity_socre
    
    def getSentenceEmbeddingsTuple(self, sentenceTuples):  # (sentenceId, sentence) as a tuple
        SentenceEmbeddings = []
        for sentenceTuple in sentenceTuples:  # sentenceTuple include its id and sentence
            SentenceEmbeddings.append( (sentenceTuple[0], self.getSentenceEmbedding(sentenceTuple[1])) )
        
        return  SentenceEmbeddings
         
    def getSentenceEmbeddings(self, sentences):  # (sentenceId, sentence) as a tuple
        SentenceEmbeddings = []
        for sentence in sentences:  # sentenceTuple include its id and sentence
            SentenceEmbeddings.append( self.getSentenceEmbedding(sentence) )
        
        return  SentenceEmbeddings
         
    def getSentenceEmbeddingsMeanpooling(self, sentences):  # (sentenceId, sentence) as a tuple
        SentenceEmbeddings = []
        for sentence in sentences:  # sentenceTuple include its id and sentence
            SentenceEmbeddings.append( self.getSentenceEmbedding(sentence) )
        
        row_wise_sum = sum(SentenceEmbeddings)  # row-wise sum 

        # divided by row-vector nums
        Meanpooling = torch.div(row_wise_sum, len(SentenceEmbeddings))
        return  Meanpooling
     
    def getSentenceEmbedding(self, sentence):
        if self.modelName == "SentenceBert":
            return  self.get_sentence_embedding_SentenceBert(sentence)
        else:
            return  self.get_sentence_embedding_SimCSE(sentence)
    
    
    def get_sentence_embedding_SentenceBert(self, one_sentence):
        #Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] #First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # Tokenize sentences
        encoded_input = self.tokenizer(one_sentence, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling. In this case, max pooling.
        sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embedding 

    def get_sentence_embedding_SentenceBert_Has_Grad(self, one_sentence):
        #Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] #First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # Tokenize sentences
        encoded_input = self.tokenizer(one_sentence, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        #with torch.no_grad():
        model_output = self.model(**encoded_input)

        # Perform pooling. In this case, max pooling.
        sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embedding 
    
    
    def get_sentence_embedding_SimCSE(self, one_sentence):  # one_sentence like "我们去散步，天气不错" and so on
        inputs=self.tokenizer(one_sentence, return_tensors="pt")
        outputs = self.model(**inputs, output_hidden_states=True)
        sentence_tokens_embedding = outputs.hidden_states[-1][:,1:-1,:].squeeze() # BertBase共有12个encoder layer，每层encoder layer的输出都保存在隐状态中了
        sentence_embedding=torch.mean(sentence_tokens_embedding, dim=0) # 平均池化的句子向量
        return sentence_embedding


class QAEncoderModel(nn.Module):
    """ Query and Answer Dual Encoder Model with pre-trained Model Sentence-Bert """
    def __init__(self):
        super(QAEncoderModel, self).__init__() 
        self.Encoder = ModelLoader("SentenceBert")

    def forward(self, querys, answers):
        query_embedding = self.Encoder.get_sentence_embedding_SentenceBert_Has_Grad(querys) 
        answer_embedding = self.Encoder.get_sentence_embedding_SentenceBert_Has_Grad(answers)
        return query_embedding, answer_embedding 


def get_sentences_embedding_approach_one(sentences = ['completeness cheak', '爱是一种选择——浅析《傲慢与偏见》中的婚姻观']) : 
    tokenizer= AutoTokenizer.from_pretrained('model')  # Load model from local directory, here is a relative path of model directory
    model = AutoModel.from_pretrained('model')   
    #model.train()
    model.eval()
    
    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentences_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    print(sentences_embedding) # it is a tensor here
    
def get_sentences_embedding_approach_two(sentences = ['completeness cheak', '爱是一种选择——浅析《傲慢与偏见》中的婚姻观']) : 
    model = QAEncoderModel()
    model.eval()
    sentences_embedding = model.Encoder.get_sentence_embedding_SentenceBert(sentences)
    print(sentences_embedding) # it is a tensor here
    
if __name__ == "__main__" : 
    get_sentences_embedding_approach_one()
    print("\nthey should be same each other!\n")
    get_sentences_embedding_approach_two()