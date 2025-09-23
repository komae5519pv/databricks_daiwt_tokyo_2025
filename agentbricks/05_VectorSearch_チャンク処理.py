# Databricks notebook source
# MAGIC %md
# MAGIC [ai_parse_document関数](https://qiita.com/taka_yayoi/items/519a4b789d08290120fd)<br>
# MAGIC [PDFデータソース](https://www.mckinsey.com/jp/~/media/mckinsey/locations/asia/japan/our%20insights/the_economic_potential_of_generative_ai_the_next_productivity_frontier_colormama_4k.pdf)<br>
# MAGIC [非構造化 データパイプライン](https://docs.databricks.com/aws/ja/generative-ai/tutorials/ai-cookbook/quality-data-pipeline-rag)

# COMMAND ----------

# MAGIC %pip install --upgrade langchain langchain-text-splitters
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 . テーブル / chunks 作成
# MAGIC テキストデータをembeddingsモデルに合わせてチャンキングしてテーブル保存します。

# COMMAND ----------

from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
import pandas as pd

# RecursiveCharacterTextSplitterの設定
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,                 # 必要に応じてチャンクサイズを変更してください
    chunk_overlap=20,               # チャンク間の重複（オーバーラップ）部分のサイズ
    keep_separator=True,            # チャンクの前後に区切り文字を残すかどうか
    length_function=len,            # チャンクの長さ判定に使う関数
    is_separator_regex=False        # 区切り文字を正規表現として扱うかどうか
)

# gold_feedbacksテーブルからデータ取得
df = spark.table(f"{catalog}.{schema}.gold_feedbacks").toPandas()

# チャンク処理およびテーブル作成用リスト
rows = []
for idx, row in df.iterrows():
    # commentをchunkに分割
    splitted_texts = text_splitter.split_text(str(row["comment"]))
    for chunk in splitted_texts:
        rows.append({
            "chunk_id": str(uuid4()),  # 一意のIDをchunkごとに生成
            "feedback_id": row["feedback_id"],
            "user_id": row["user_id"],
            "product_id": row["product_id"],
            "rating": row["rating"],
            "date": row["date"],
            "category": row["category"],
            "positive_score": row["positive_score"],
            "summary": row["summary"],
            "chunk": chunk
        })

# 新しいDataFrame作成
df_final = pd.DataFrame(rows)

# Spark DataFrame変換・テーブル保存
df_final_spark = spark.createDataFrame(df_final)
df_final_spark.write.mode('overwrite').format('delta') \
            .option("delta.enableChangeDataFeed", "true") \
            .saveAsTable(f'{catalog}.{schema}.gold_feedbacks_chunks')
