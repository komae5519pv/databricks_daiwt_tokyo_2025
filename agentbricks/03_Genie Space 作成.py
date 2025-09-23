# Databricks notebook source
# MAGIC %md
# MAGIC # PlaygroundででMCPサーバーを活用した Genie Spaceエージェントを実行

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Genie Spaceを作る
# MAGIC <img src='https://github.com/komae5519pv/databricks_quick_demo/blob/main/20250912_aibi_quickworkshop/images/Genie%E3%82%B9%E3%83%98%E3%82%9A%E3%83%BC%E3%82%B9.gif?raw=true' width='88%'/>
# MAGIC
# MAGIC
# MAGIC #### データ
# MAGIC - `gold_feedbacks`
# MAGIC - `gold_user`
# MAGIC - `products`
# MAGIC - `transactions`
# MAGIC
# MAGIC #### 設定
# MAGIC - Title: `小売スーパー売り上げ分析`
# MAGIC - Description: `ブリックスマートの売上要因の深堀りを行うチャットボット`
# MAGIC - Default Warehouse: <任意のSQL Warehouse>
# MAGIC - サンプルSQL:
# MAGIC   - `ユーザーの性別の分布はどうなっていますか？`
# MAGIC   - `食料品の平均レビュースコアはどのくらいですか？`
# MAGIC   - `最近3ヶ月間の月ごとの取引回数はどうなっていますか？`
# MAGIC
# MAGIC #### 指示
# MAGIC <テキスト>
# MAGIC 一般的な指示:
# MAGIC ```
# MAGIC *あなたはブリックスマートの店長アシスタントです。店舗の売上データ分析を行うアシスタントとして、入力者の分析および洞察を導き出す補助をして、日本語で回答してください。
# MAGIC *「年代別」の指示がある時、年齢を10単位でグルーピングしてください
# MAGIC *売上は、単価かける数量で計算してください
# MAGIC *SQLのFROM句でテーブル指定する際、``で括らないでください
# MAGIC 現在を起点とする場合、必ず「SELECT from_utc_timestamp(CURRENT_TIMESTAMP(), 'Asia/Tokyo')::DATE;」で日時特定してください
# MAGIC ```
# MAGIC
# MAGIC <結合>
# MAGIC - 1つ目
# MAGIC   - 左のテーブル: `gold_feedbacks` / 右のテーブル: `products`
# MAGIC   - 結合条件: `product_id` = `product_id`
# MAGIC   - 関係のタイプ: `多対一`
# MAGIC - 2つ目
# MAGIC   - 左のテーブル: `gold_feedbacks` / 右のテーブル: `users`
# MAGIC   - 結合条件: `user_id` = `user_id`
# MAGIC   - 関係のタイプ: `多対一`

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC <SQLクエリー>
# MAGIC クエリーの例：
# MAGIC - `最終評価が3未満/3以上で評価した人の平均ユーザー単価を比較したい`
# MAGIC
# MAGIC ↓↓下記実行結果のテキストをクエリーの例に貼り付けてください

# COMMAND ----------

print(f'''
WITH latest_feedback AS (
  SELECT
    user_id,
    rating,
    date AS feedback_date,
    feedback_id,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY date DESC, feedback_id DESC) AS rn
  FROM {catalog}.{schema}.gold_feedbacks
),
labeled_users AS (
  SELECT
    user_id,
    feedback_date,
    CASE WHEN rating < 3 THEN 'below_3' ELSE 'above_or_equal_3' END AS rating_bucket
  FROM latest_feedback
  WHERE rn = 1
),
tx_after AS (
  SELECT
    t.user_id,
    t.transaction_price,
    t.quantity,
    l.rating_bucket
  FROM {catalog}.{schema}.transactions t
  JOIN labeled_users l
    ON t.user_id = l.user_id
   AND t.transaction_date >= l.feedback_date   -- 最終評価日以降に限定
  WHERE t.quantity > 0
),
per_user AS (
  -- ユーザーごとの「金額合計 ÷ 数量合計」= ユーザー単価
  SELECT
    user_id,
    rating_bucket,
    SUM(transaction_price) / SUM(quantity) AS user_unit_price
  FROM tx_after
  GROUP BY user_id, rating_bucket
)
SELECT
  rating_bucket,
  AVG(user_unit_price)                                  AS avg_user_unit_price,     -- グループ平均（知りたい指標）
  APPROX_PERCENTILE(user_unit_price, 0.5)               AS median_user_unit_price,  -- 参考：中央値（外れ値に強い）
  COUNT(*)                                              AS user_count
FROM per_user
GROUP BY rating_bucket
ORDER BY rating_bucket
;
''')
