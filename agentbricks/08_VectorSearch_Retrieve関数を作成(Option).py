# Databricks notebook source
# MAGIC %md
# MAGIC ### Vector Search を行う関数`manual_retriever`をカタログへ登録する
# MAGIC [ドキュメント](https://docs.databricks.com/aws/ja/generative-ai/agent-framework/unstructured-retrieval-tools#unity-catalog%E6%A9%9F%E8%83%BD%E3%82%92%E6%8C%81%E3%81%A4%E3%83%99%E3%82%AF%E3%83%88%E3%83%AB%E6%A4%9C%E7%B4%A2%E3%83%AC%E3%83%88%E3%83%AA%E3%83%BC%E3%83%90%E3%83%BC%E3%83%84%E3%83%BC%E3%83%AB)  を参考にカタログへベクトル検索関数を登録しましょう

# COMMAND ----------

# MAGIC %pip install -U databricks-vectorsearch databricks-langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Vector Search を行う関数`manual_retriever`をUCカタログに登録

# COMMAND ----------

create_func = f"""
CREATE OR REPLACE FUNCTION {catalog}.{schema}.manual_retriever (
  query STRING
  COMMENT 'スーパーBricksマートに対するお客様フィードバックを検索するためのクエリ文字列'
) RETURNS TABLE
COMMENT '与えられたクエリに最も関連するテキストを取得する、スーパーBricksマートのお客様フィードバックを返す関数です'
LANGUAGE SQL
  RETURN
  SELECT
    chunk,
    map(
      'feedback_id', CAST(feedback_id AS STRING),
      'user_id', CAST(user_id AS STRING),
      'product_id', CAST(product_id AS STRING),
      'rating', CAST(rating AS STRING),
      'date', CAST(date AS STRING),
      'category', CAST(category AS STRING),
      'positive_score', CAST(positive_score AS STRING),
      'summary', CAST(summary AS STRING)
    ) AS metadata
  FROM
    VECTOR_SEARCH (
      index => '{catalog}.{schema}.gold_feedbacks_index',
      query => query,
      num_results => 3
    )
"""

# print(create_func)
spark.sql(create_func)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. UCカタログに登録した関数をUCFunctionToolkitでラップ
# MAGIC この取得ツールを AI エージェントで使用するには、 UCFunctionToolkitでラップします。<br>
# MAGIC これにより、MLflow ログに RETRIEVER スパンの種類を自動的に生成することで、MLflow による自動トレースが可能になります。<br>
# MAGIC [参考](https://docs.databricks.com/aws/ja/generative-ai/agent-framework/unstructured-retrieval-tools#unity-catalog%E6%A9%9F%E8%83%BD%E3%82%92%E6%8C%81%E3%81%A4%E3%83%99%E3%82%AF%E3%83%88%E3%83%AB%E6%A4%9C%E7%B4%A2%E3%83%AC%E3%83%88%E3%83%AA%E3%83%BC%E3%83%90%E3%83%BC%E3%83%84%E3%83%BC%E3%83%AB)

# COMMAND ----------

import mlflow
from databricks_langchain import VectorSearchRetrieverTool

vs_tool = VectorSearchRetrieverTool(
  index_name=f"{catalog}.{schema}.gold_feedbacks_index",
  tool_name="manual_retriever",
  tool_description="スーパーBricksマートのお客様フィードバックを検索するツールです",
  columns=["chunk_id", "feedback_id", "user_id", "product_id", "rating",
           "date", "category", "positive_score", "summary", "chunk"],
  num_results=3     # 1件だけ返却
)

vs_tool.invoke("店舗設備に関するフィードバックを教えてください")
