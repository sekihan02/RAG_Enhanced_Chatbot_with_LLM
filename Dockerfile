FROM python:3.9

# 作業ディレクトリを設定
WORKDIR /app

# 必要なPythonパッケージをインストール
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのソースコードをコンテナ内にコピー
COPY . /app

# FLASK_APP環境変数を設定
ENV FLASK_APP=app/app.py

# Flaskアプリケーションを実行
CMD ["flask", "run", "--host=0.0.0.0"]
