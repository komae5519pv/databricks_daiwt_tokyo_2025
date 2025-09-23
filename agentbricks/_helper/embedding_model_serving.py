# Databricks notebook source
# %md
# # 外部モデル（text-embedding-3-small）をサービングエンドポイントにデプロイします
# * [外部モデルとしてサービングエンドポイントにデプロイ](https://qiita.com/taka_yayoi/items/352bff5f19789f299341)
# * DBR 17.0 ML以降で実行してください
# * 前提
#   * ここでは個別契約しているOpenAI APIを外部モデルとしてModel Serving Endpointに登録します
#   * よって事前に下記が必要です
#     * OpenAI APIの契約（[公式サイト](https://platform.openai.com/settings/organization/general)）
#     * OpenAIのAPI_KEYの発行
#     * AI_KEYの[Databricksシークレット](https://docs.databricks.com/aws/ja/security/secrets)への登録（シークレットスコープ作成&シークレット作成）

# COMMAND ----------

# %pip install -U mlflow[genai]>=2.9.0
# dbutils.library.restartPython()

# COMMAND ----------

OPENAI_API_TOKEN = dbutils.secrets.get(scope="komae_scope", key="openai_api")
print(OPENAI_API_TOKEN)

# COMMAND ----------

# %md
# # モデルサービングエンドポイントにデプロイ
# 外部モデル：`text-embedding-3-small`

# COMMAND ----------

import mlflow.deployments

ENDPOINT_NAME = "komae-text-embedding-3-small"
client = mlflow.deployments.get_deploy_client("databricks")

# エンドポイントが存在するか確認
def endpoint_exists(client, name):
    try:
        client.get_endpoint(name)   # 存在すれば詳細が返る
        return True
    except Exception:
        return False

if not endpoint_exists(client, ENDPOINT_NAME):
    client.create_endpoint(
        name=ENDPOINT_NAME,
        config={
            "served_entities": [{
                "name": "openai-embeddings",
                "external_model": {
                    "name": "text-embedding-3-small",
                    "provider": "openai",
                    "task": "llm/v1/embeddings",
                    "openai_config": {
                        "openai_api_key_plaintext": OPENAI_API_TOKEN
                    }
                }
            }]
        }
    )
else:
    print(f"Endpoint '{ENDPOINT_NAME}' already exists. Skipping creation.")
