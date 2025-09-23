# Databricks notebook source
# カタログ情報
MY_CATALOG = "komae_demo_v4"              # ご自分のカタログ名に変更してください
MY_SCHEMA = "bricksmart"
MY_VOLUME = "raw"

# ベクターサーチエンドポイント
# MY_VECTOR_SEARCH_ENDPOINT = "komae_vs_endpoint"
MY_VECTOR_SEARCH_ENDPOINT = "one-env-shared-endpoint-2"

# Embedding Model Endpoint
EMBEDDING_MODEL_ENDPOINT_NAME = "komae-text-embedding-3-small"
# EMBEDDING_MODEL_ENDPOINT_NAME = "aoai-text-embedding-3-large"

# COMMAND ----------

# カタログ、スキーマ、ボリューム作成
spark.sql(f"CREATE CATALOG IF NOT EXISTS {MY_CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {MY_CATALOG}.{MY_SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {MY_CATALOG}.{MY_SCHEMA}.{MY_VOLUME}")

spark.sql(f"USE CATALOG {MY_CATALOG}")
spark.sql(f"USE SCHEMA {MY_SCHEMA}")

# ボリュームのサブディレクトリ作成
dbutils.fs.mkdirs(f"/Volumes/{MY_CATALOG}/{MY_SCHEMA}/{MY_VOLUME}/binary")
dbutils.fs.mkdirs(f"/Volumes/{MY_CATALOG}/{MY_SCHEMA}/{MY_VOLUME}/binary/image/taxi")
dbutils.fs.mkdirs(f"/Volumes/{MY_CATALOG}/{MY_SCHEMA}/{MY_VOLUME}/binary/image/lunch")
dbutils.fs.mkdirs(f"/Volumes/{MY_CATALOG}/{MY_SCHEMA}/{MY_VOLUME}/binary/pdf")
dbutils.fs.mkdirs(f"/Volumes/{MY_CATALOG}/{MY_SCHEMA}/{MY_VOLUME}/feedbacks")

print(f"MY_CATALOG: {MY_CATALOG}")
print(f"MY_SCHEMA: {MY_SCHEMA}")
print(f"MY_VOLUME: {MY_VOLUME}")
print(f"/Volumes/{MY_CATALOG}/{MY_SCHEMA}/{MY_VOLUME}/binary/image/taxi")
print(f"/Volumes/{MY_CATALOG}/{MY_SCHEMA}/{MY_VOLUME}/binary/image/lunch")
print(f"/Volumes/{MY_CATALOG}/{MY_SCHEMA}/{MY_VOLUME}/binary/pdf")
print(f"/Volumes/{MY_CATALOG}/{MY_SCHEMA}/{MY_VOLUME}/feedbacks")
