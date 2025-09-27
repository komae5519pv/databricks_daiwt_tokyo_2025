# Databricks notebook source
# MAGIC %md
# MAGIC # PlaygroundでAIエージェントを実行

# COMMAND ----------

# MAGIC %md
# MAGIC ## Playground における AIエージェント の初期設定
# MAGIC Playgroundへ移動します。
# MAGIC
# MAGIC #### ツール
# MAGIC - ツールを追加: `MCP Servers`
# MAGIC   - Vector Search: `<カタログ>.bricksmart`
# MAGIC   - Genie Space: `小売スーパー売り上げ分析`
# MAGIC   - Unity Catalog Function: `<カタログ>.bricksmart` (Option)
# MAGIC
# MAGIC #### システムプロンプト
# MAGIC ```
# MAGIC あなたはスーパーBricksマートの売り上げ分析を行うエージェントです。質問に対して日本語で回答してください。
# MAGIC 必要に応じてツールを利用してデータを取得して、簡潔に回答してください。
# MAGIC 1つの質問回答に対して、ツールの利用は3回以内にしてください。
# MAGIC データに基づく場合はその旨を、仮説の場合はその旨を明確に明記にしてください。
# MAGIC ```
# MAGIC
# MAGIC 英語の方が精度は上がりやすい
# MAGIC ```
# MAGIC You are an agent that performs sales analysis for the supermarket Bricks Mart. Please respond to questions in Japanese.
# MAGIC Use tools as needed to retrieve data, and provide concise answers.
# MAGIC Limit tool usage to a maximum of three times per question.
# MAGIC Always answer based on facts. If data cannot be retrieved, respond with: "データが取得できないため回答できません".
# MAGIC ```
# MAGIC
# MAGIC #### 入力例（ユーザープロンプト）
# MAGIC - RAG
# MAGIC   - `フィードバックで良い評価と悪い評価で顕著な内容を教えて`
# MAGIC - Genie
# MAGIC   - `先月の最も売れなかった店舗はどこ？その要因も考えて`
# MAGIC   - `先月の最も売れた商品は？`
# MAGIC   - `それは全店舗合計の金額ですか？`
# MAGIC - UC関数
# MAGIC   - `一番繁盛しているお店で一番人気の商品の在庫状況を教えて` (Option)
