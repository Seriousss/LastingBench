import re
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize, word_tokenize
from rank_bm25 import BM25Okapi
from langchain_text_splitters import NLTKTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from collections import Counter

def replace_case_insensitive(text: str, old: str, new: str) -> str:
    pattern = re.compile(re.escape(old), re.IGNORECASE)

    return pattern.sub(new, text)
def get_word_list(s1):
    # Separate sentences by word, Chinese by word, English by word, numbers by space
    regEx = re.compile('[\W]')   
    res = re.compile(r"([\u4e00-\u9fa5])")    #  [\u4e00-\u9fa5] for Chinese

    p1 = regEx.split(s1.lower())
    str1_list = []
    for str in p1:
        if res.split(str) == None:
            str1_list.append(str)
        else:
            ret = res.split(str)
            for ch in ret:
                str1_list.append(ch)

    list_word1 = [w for w in str1_list if len(w.strip()) > 0]  

    return  list_word1
def get_word_len(s1):
    return len(get_word_list(s1))

regex = r'([。？！；\n.!?;]\s*)'
def retriveDoc(text,query,top_k=3):
    import os
    sentences = sent_tokenize(text)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"))
    # Create vector database through FAISS (built from sentence list)
    vector_store = FAISS.from_texts(sentences, embeddings)
    
    retrieved_docs = vector_store.similarity_search(query, k=top_k)
    print("Retrieved sentences:", retrieved_docs)
    
    # Return results, can adjust the return structure as needed, here returns a dictionary containing context
    return retrieved_docs


def most_similar_sentence_bm25(paragraph, target_sentence):
    """
    Use BM25 algorithm to find the most similar sentence to target_sentence in the given paragraph,
    return (most similar sentence, score).
    """
    # 1. First split the paragraph into a list of sentences
    sentences = sent_tokenize(paragraph)

    # 2. Tokenize each sentence
    tokenized_sentences = [word_tokenize(sent) for sent in sentences]

    # 3. Create a retrieval instance using BM25Okapi
    bm25 = BM25Okapi(tokenized_sentences)

    # 4. Tokenize the target sentence
    target_tokens = word_tokenize(target_sentence)

    # 5. Use BM25 to calculate similarity scores for each sentence
    scores = bm25.get_scores(target_tokens)
    # scores.shape == (len(sentences),)

    # 6. Find the index of the sentence with the highest score
    max_idx = scores.argmax()

    # Return the most similar sentence and its score
    return sentences[max_idx]


def f1_score_text(pred, gold):
    pred_tokens = word_tokenize(pred)
    gold_tokens = word_tokenize(gold)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def compute_best_sentence_f1(pred_text, gold_text):
    pred_sentences = sent_tokenize(pred_text)
    gold_sentences = sent_tokenize(gold_text)
    f1_scores = []
    for pred in pred_sentences:
        best_f1 = 0.0
        for gold in gold_sentences:
            f1 = f1_score_text(pred, gold)
            if f1 > best_f1:
                best_f1 = f1
        f1_scores.append(best_f1)
    avg_f1 = sum(f1_scores) / len(pred_sentences) if pred_sentences else 0.0
    return avg_f1