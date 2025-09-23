# Databricks notebook source
# MAGIC %md
# MAGIC ## ベクトル検索インデックス作成・クエリ実行

# COMMAND ----------

# MAGIC %md
# MAGIC [ベクトル検索インデックスを作成してクエリを実行する方法](https://docs.databricks.com/aws/ja/generative-ai/create-query-vector-search)<br>
# MAGIC [PDFデータソース](https://www.mckinsey.com/jp/~/media/mckinsey/locations/asia/japan/our%20insights/the_economic_potential_of_generative_ai_the_next_productivity_frontier_colormama_4k.pdf)<br>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. インストール

# COMMAND ----------

# MAGIC %md
# MAGIC ベクトル検索 SDK を使用するには、ノートブックにインストールする必要があります。 次のコードを使用して、パッケージをインストールします。

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch -U mlflow[genai]>=2.9.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 認証

# COMMAND ----------

# dbutils.widgets.text("Personal Access Token", "")

# COMMAND ----------

# ウィジットからPAT取得する場合
# DATABRICKS_TOKEN = dbutils.widgets.get("Personal Access Token")

# シークレットからPAT取得する場合
DATABRICKS_TOKEN = dbutils.secrets.get(scope="komae_scope", key="ws_token")
print(DATABRICKS_TOKEN)

# COMMAND ----------

# VectorSearchClientをインポート
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

# # 認証用のPATトークンを自動生成します
# vsc = VectorSearchClient(
#   workspace_url="https://adb-984752964297111.11.azuredatabricks.net",
#   personal_access_token=DATABRICKS_TOKEN
# )

# # サービスプリンシパルトークンを使って認証します
# vsc = VectorSearchClient(
#   service_principal_client_id=<CLIENT_ID>,
#   service_principal_client_secret=<CLIENT_SECRET>
# )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. ベクトル検索エンドポイントを作成する

# COMMAND ----------

# # Python SDK を使用してベクトル検索エンドポイントを作成します
# vsc.create_endpoint(
#     name=MY_VECTOR_SEARCH_ENDPOINT,
#     endpoint_type="STANDARD" # または "STORAGE_OPTIMIZED"
# )

# # # Python SDK を使用してベクトル検索エンドポイントを削除します
# # vsc.delete_endpoint(name=MY_VECTOR_SEARCH_ENDPOINT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. (Option)埋め込みモデルを提供するエンドポイントを作成して構成する

# COMMAND ----------

# MAGIC %run ./_helper/embedding_model_serving

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. ベクトル検索インデックスを作成する
# MAGIC Python SDK を使用してインデックスを作成する

# COMMAND ----------

print(f"MY_VECTOR_SEARCH_ENDPOINT: {MY_VECTOR_SEARCH_ENDPOINT}")
print(f"EMBEDDING_MODEL_ENDPOINT_NAME: {EMBEDDING_MODEL_ENDPOINT_NAME}")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c
import time

# ベクター検索インデックスのソーステーブル・ベクター検索インデックスのパス
source_table_fullname = f"{catalog}.{schema}.gold_feedbacks_chunks"
vs_index_fullname = f"{catalog}.{schema}.gold_feedbacks_index"

# インデックス存在チェック関数（実装例）
def index_exists(vsc, endpoint_name, index_name):
  try:
    vsc.get_index(endpoint_name, index_name)
    return True
  except Exception:
    return False

# インデックス作成or同期
if not index_exists(vsc, MY_VECTOR_SEARCH_ENDPOINT, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {MY_VECTOR_SEARCH_ENDPOINT}...")
  index = vsc.create_delta_sync_index(
    endpoint_name=MY_VECTOR_SEARCH_ENDPOINT,                        # エンドポイント名
    source_table_name=source_table_fullname,                        # ベクター検索インデックスのソーステーブルパス
    index_name=vs_index_fullname,                                   # 作成するベクター検索インデックスパス
    pipeline_type="TRIGGERED",                                      # 同期モード
    primary_key="chunk_id",                                         # 主キー
    # columns_to_sync=["chunk_id", "<同期する列>"],                     # 同期する列（オプション）
    embedding_source_column="chunk",                                # embeddings対象のチャンクテキスト
    embedding_model_endpoint_name=EMBEDDING_MODEL_ENDPOINT_NAME     # embedding modelのエンドポイント名
  )
else:
  index = vsc.get_index(MY_VECTOR_SEARCH_ENDPOINT, vs_index_fullname)
  index.sync()

# ONLINEになるまで待機
while not index.describe().get('status').get('detailed_state').startswith('ONLINE'):
  print("Waiting for index to be ONLINE...")
  time.sleep(5)
print("Index is ONLINE")
index.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC 作成したインデックスをチェック

# COMMAND ----------

# インデックスの作成を確認
index = vsc.get_index(index_name=f"{catalog}.{schema}.gold_feedbacks_index")
index.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. ベクトル検索インデックスの更新
# MAGIC Delta Sync インデックスの更新:<br>
# MAGIC 連続 同期モードで作成されたインデックスは、ソース Delta テーブルが変更されると自動的に更新されます。<br>
# MAGIC トリガー 同期モードを使用している場合は、UI、Python SDK、または REST API を使用して同期を開始できます。

# COMMAND ----------

# Python SDKを使ってインデックスを同期
index = vsc.get_index(index_name=f"{catalog}.{schema}.gold_feedbacks_index")

index.sync()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. ベクトル検索エンドポイントのクエリ

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7-1. 普通にクエリしてみる

# COMMAND ----------

# 返却されるカラムの設定
all_columns = spark.table(f"{catalog}.{schema}.gold_feedbacks_chunks").columns
print(all_columns)

# 質問
question = "店舗設備について改善ポイントは何ですか？"

results = index.similarity_search(
  query_text=question,
  # columns=["feedback_id", "comment"],                     # 返却されるカラムを指定
  columns=all_columns,                                    # 返却されるカラムを指定
  num_results=3
  )

# print(results)

# カラム名の取得
columns = [col['name'] for col in results['manifest']['columns']]
rows = results['result']['data_array']

# 指定の件数をループして表示
for idx, row in enumerate(rows, start=1):
    print(f"<{idx}件目>")
    for col_name, value in zip(columns, row):
        print(f"{col_name}：{value}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7-2. 特定列でフィルターしつつ、クエリしてみる

# COMMAND ----------

from datetime import date

# 返却されるカラムの設定
all_columns = spark.table(f"{catalog}.{schema}.gold_feedbacks_chunks").columns
print(all_columns)

# 質問
question = "店舗設備について改善ポイントは何ですか？"

# Databricksで計算された埋め込みを使ったDelta Sync Indexの類似検索
results = index.similarity_search(
    query_text=question,
    # columns=["feedback_id", "comment"],                     # 返却されるカラムを指定
    columns=all_columns,                                    # 返却されるカラムを指定
    filters={"rating <=": 2.0},                             # 連番10以上でフィルター
    num_results=3
  )

# print(results)

# カラム名の取得
columns = [col['name'] for col in results['manifest']['columns']]
rows = results['result']['data_array']

# 指定の件数をループして表示
for idx, row in enumerate(rows, start=1):
    print(f"<{idx}件目>")
    for col_name, value in zip(columns, row):
        print(f"{col_name}：{value}")
    print()
