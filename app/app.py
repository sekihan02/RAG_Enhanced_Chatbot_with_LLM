from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import os
import pandas as pd
import ast
from scipy import spatial
from tqdm import tqdm
from PyPDF2 import PdfReader
from langdetect import detect

import openai
from openai import OpenAI
from dotenv import load_dotenv

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['UPLOAD_FOLDER'] = 'uploads'  # アップロードされたファイルを保存するフォルダ
socketio = SocketIO(app)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# モデルとエンコーディングの設定
MODEL_NAME = "gpt-3.5-turbo-0125"
MODEL4o_NAME = "gpt-4o-2024-05-13"

EMBEDDING_MODEL = "text-embedding-3-small"
embedding_encoding = "cl100k_base"

max_tokens = 1000
TEMPERATURE = 0.7

# OpenAIクライアントの初期化
client = OpenAI()

# 関連度を計算するための関数
def relatedness_fn(x, y):
    return 1 - spatial.distance.cosine(x, y)

# search function
def strings_ranked_by_relatedness(query: str, csv_files, top_n: int = 3):
    related_texts = []
    query_embedding_response = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
    query_embedding = query_embedding_response.data[0].embedding
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, converters={'embedding': ast.literal_eval})

        strings_and_relatednesses = [
            (row["text"], relatedness_fn(query_embedding, row["embedding"]), csv_file)
            for _, row in df.iterrows()
        ]
        related_texts.extend(strings_and_relatednesses)

    related_texts.sort(key=lambda x: x[1], reverse=True)
    # print(related_texts[:2])
    strings, relatednesses, files = zip(*related_texts[:top_n])
    return strings, relatednesses, files

def generate_response(query: str, csv_files, top_n=3):
    strings, relatednesses, files = strings_ranked_by_relatedness(query, csv_files, top_n=top_n)
    context = "\n\n".join([f"File: {file}\nText: {text}" for text, file in zip(strings, files)])
    prompt = [
        {'role': 'system', 'content': 'You are an expert assistant who answers questions based on provided documents.'},
        {'role': 'user', 'content': f"Answer the following question based on the provided context.\n\nQuestion: {query}\n\nContext:\n{context}"}
    ]
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=prompt,
        temperature=0.7,
    )
    # 重複ファイル名削除チェック
    used_files = set()
    for file, text in zip(files, strings):
        if file not in used_files:
            # print(f"File: {file}")
            used_files.add(file)
    return response.choices[0].message.content, used_files, strings

def translate_to_japanese(model_name, detailed_outline):
    """
    Translates the detailed outline of a Wikipedia page from English to Japanese.
    
    Parameters:
    model_name (str): The model to be used for translation.
    detailed_outline (str): The detailed outline in English.
    
    Returns:
    str: The translated outline in Japanese.
    """
    prompt = [
        {"role": "system", "content": "You need to translate the following English text into Japanese."},
        {"role": "user", "content": detailed_outline},
        {"role": "system", "content": "Please provide the translation of the entire text into Japanese, maintaining the accuracy and context of the original information."}
    ]
    
    response = client.chat.completions.create(
        model=model_name,
        messages=prompt,
        temperature=TEMPERATURE,
    )
    
    translated_outline = response.choices[0].message.content
    
    return translated_outline

# PDFからテキストを抽出する関数
def convert_pdf_to_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text() + '\n'
    return text

def generate_wiki_outline(model_name, scientific_paper):
    # Wikipediaページのアウトラインを生成する関数
    prompt = [
        {"role": "system", "content": "You are an experienced Wikipedia writer and want to edit a specific page."},
        {"role": "system", "content": "Your task is to write an outline of a Wikipedia page based on the content of a scientific article."},
        {"role": "system", "content": "Here is the format of your writing:"},
        {"role": "system", "content": "1. Use '#' Title ' to indicate section title, '##' Title ' to indicate subsection title, '###' Title ' to indicate subsubsection title, and so on."},
        {"role": "system", "content": "2. Do not include other information."},
        {"role": "user", "content": f"Scientific Paper: {scientific_paper}"},
        {"role": "system", "content": "Based on this information, please create a structured outline for the Wikipedia page."},
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=prompt,
        temperature=TEMPERATURE,
    )
    
    outline = response.choices[0].message.content
    
    return outline

def generate_detailed_outline(model_name, outline, scientific_paper):
    """
    Given a summary and related search results, enhance a Wikipedia page outline with detailed descriptions.
    
    Parameters:
    model_name (str): The model to be used for generating the descriptions.
    summary_text (str): Summary text providing a concise overview of the topic.
    related_search_results (str): Text containing related search results or additional contextual information.
    outline (str): The basic outline of the Wikipedia page.
    
    Returns:
    str: The enhanced outline with detailed descriptions for each section.
    """
    prompt = [
        {"role": "system", "content": "You are tasked with enhancing a Wikipedia page outline by integrating detailed descriptions based on a scientific article."},
        {"role": "system", "content": "Here is the basic outline of the page you need to expand with detailed explanations:"},
        {"role": "user", "content": outline},
        {"role": "system", "content": "Use the following scientific article."},
        {"role": "user", "content": f"Scientific Paper: {scientific_paper}"},
        {"role": "system", "content": "Add a comprehensive description of each section based on the content of the scientific article, including an overview, an accurate description of the methodology, and relevant interpretations."}
    ]
    
    response = client.chat.completions.create(
        model=model_name,
        messages=prompt,
        temperature=TEMPERATURE,
    )
    
    detailed_outline = response.choices[0].message.content
    
    return detailed_outline

class StreamingLLMMemory:
    """
    StreamingLLMMemory クラスは、最新のメッセージと特定数のattention sinksを
    メモリに保持するためのクラスです。
    
    attention sinksは、言語モデルが常に注意を向けるべき初期のトークンで、
    モデルが過去の情報を"覚えて"いるのを手助けします。
    """
    def __init__(self, max_length=10, attention_sinks=4):
        """
        メモリの最大長と保持するattention sinksの数を設定
        
        :param max_length: int, メモリが保持するメッセージの最大数
        :param attention_sinks: int, 常にメモリに保持される初期トークンの数
        """
        self.memory = []
        self.max_length = max_length
        self.attention_sinks = attention_sinks
    
    def get(self):
        """
        現在のメモリの内容を返します。
        
        :return: list, メモリに保持されているメッセージ
        """
        return self.memory
    
    def add(self, message):
        """
        新しいメッセージをメモリに追加し、メモリがmax_lengthを超えないように
        調整します。もしmax_lengthを超える場合、attention_sinksと最新のメッセージを
        保持します。
        
        :param message: str, メモリに追加するメッセージ
        """
        self.memory.append(message)
        if len(self.memory) > self.max_length:
            self.memory = self.memory[:self.attention_sinks] + self.memory[-(self.max_length-self.attention_sinks):]
    
    def add_pair(self, user_message, ai_message):
        """
        ユーザーとAIからのメッセージのペアをメモリに追加します。
        
        :param user_message: str, ユーザーからのメッセージ
        :param ai_message: str, AIからのメッセージ
        """
        self.add({"role": "user", "content": user_message})
        self.add({"role": "assistant", "content": ai_message})

memory = StreamingLLMMemory(max_length=16)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('send_message')
def handle_message(data):
    query = data['message']
    summary_enabled = data.get('summary', False)
    
    save_dir = "./embeddings"
    csv_dir = save_dir
    # csv_files = os.listdir(save_dir)
    csv_files = [os.path.join(csv_dir, file) for file in os.listdir(csv_dir) if file.endswith('.csv')]
    # csv_files
    # 質問に対する回答を生成
    response, files, strings = generate_response(query, csv_files, top_n=5)
    
    # linguaを使用するとmarkdown記法を英語と判断してしまうのでナイーブベースで言語取得
    lang = detect(response)
    print(f"language: {lang}")
    
    if lang != "ja":
        translated_response = translate_to_japanese(MODEL_NAME, response)
    else:
        translated_response = response
    
    # QAの結果を送信
    emit('receive_message', {'message': translated_response})
    # 使用したファイルの表示
    
    emit('receive_message', {'message': "この回答に使用した資料"})
    pdf_dir = "./papers"
    # PDFファイルのパスを生成
    pdf_paths = []
    for file in files:
        file_name = os.path.basename(file)  # ファイル名を取得
        pdf_file_name = os.path.splitext(file_name)[0] + ".pdf"  # 拡張子を.csvから.pdfに変更
        # 使用ファイル名の送信
        emit('receive_message', {'message': pdf_file_name})
        pdf_file_path = os.path.join(pdf_dir, pdf_file_name)  # PDFファイルのパスを生成
        pdf_paths.append(pdf_file_path)

    # 詳細説明機能の有効判定
    if summary_enabled:
        # 各PDFファイルのテキストを取得して表示
        pdf_texts = []
        for pdf_path in pdf_paths:
            text = convert_pdf_to_text(pdf_path)
            pdf_texts.append({'file': pdf_path, 'text': text})

        # pdf取得結果のテスト表示
        for pdf in pdf_texts:
            
            outline_text = generate_wiki_outline(MODEL4o_NAME, pdf['text'])
            # 生成したアウトラインに詳細な説明を加える
            detailed_outline = generate_detailed_outline(MODEL4o_NAME, outline_text, outline_text)
            
            lang = detect(detailed_outline)
            if lang != "ja":
                translated_detailed_outline = translate_to_japanese(MODEL_NAME, detailed_outline)
            else:
                translated_detailed_outline = response
            # 詳細説明の送信
            emit('receive_message', {'message': f"説明対象ファイル名: {pdf['file']}\n\n{translated_detailed_outline}"})

if __name__ == '__main__':
    socketio.run(app, debug=True)
