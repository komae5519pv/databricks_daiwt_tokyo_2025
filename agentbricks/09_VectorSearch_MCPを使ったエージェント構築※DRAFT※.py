# Databricks notebook source
# MAGIC %md
# MAGIC ### 1. MCPを使ったエージェント構築
# MAGIC [DatabricksにおけるMCPを用いたエージェントの構築および評価](https://qiita.com/taka_yayoi/items/f2c7ad187ff3acbe0cc6)
# MAGIC
# MAGIC 次のスニペットを実行して、MCPサーバーへの接続を検証します。このスニペットは、Unity Catalogツールを一覧表示し、その後ベクトル検索インデックスをクエリします。

# COMMAND ----------

# %pip install -U --quiet databricks-sdk databricks-langchain databricks-agents databricks-vectorsearch bs4==0.0.2 markdownify==0.14.1 pydantic==2.10.1 databricks-mcp mlflow mcp "databricks-sdk[openai]" databricks-agents
# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install -U databricks-agents databricks-mcp databricks-vectorsearch mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC 上記のスニペットを基に、ツールを使う基本的なシングルターンエージェントを定義できます。<br>
# MAGIC 後続のセクションでデプロイできるよう、エージェントのコードを mcp_agent.py という名前でローカルに保存します。

# COMMAND ----------

import asyncio

from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession
from databricks_mcp import DatabricksOAuthClientProvider
from databricks.sdk import WorkspaceClient

# ワークスペースへの認証を構成
workspace_client = WorkspaceClient()
workspace_hostname = workspace_client.config.host
mcp_server_url = f"{workspace_hostname}/api/2.0/mcp/vector-search/{catalog}/{schema}"

# 以下のスニペットは、Unity Catalog の関数 MCP サーバーを使用して Vector Search Index を公開します
async def test_connect_to_server():
    async with streamablehttp_client(
        f"{mcp_server_url}", auth=DatabricksOAuthClientProvider(workspace_client)
    ) as (read_stream, write_stream, _), ClientSession(
        read_stream, write_stream
    ) as session:
        # MCP サーバーからツールを一覧取得し、呼び出す
        await session.initialize()
        tools = await session.list_tools()
        toolnames = [t.name for t in tools.tools]
        print(
            f"MCP サーバー {mcp_server_url} から検出されたツール: {toolnames}"
        )
        result = await session.call_tool(
            toolnames[0], {"query": "Databricksとは何ですか？"}
        )
        print(
            f"{toolnames[0]} ツールを呼び出し、結果を取得: {result.content}"
        )

await test_connect_to_server()

# COMMAND ----------

# MAGIC %%writefile mcp_agent.py
# MAGIC
# MAGIC import os
# MAGIC from contextlib import asynccontextmanager
# MAGIC import json
# MAGIC import uuid
# MAGIC import asyncio
# MAGIC from typing import Any, Callable, List
# MAGIC from pydantic import BaseModel
# MAGIC import threading
# MAGIC
# MAGIC import mlflow
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
# MAGIC
# MAGIC from databricks_mcp import DatabricksOAuthClientProvider
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from mcp.client.session import ClientSession
# MAGIC from mcp.client.streamable_http import streamablehttp_client
# MAGIC
# MAGIC # Databricksノートブック環境でのみnest_asyncioを適用
# MAGIC if os.getenv('DATABRICKS_RUNTIME_VERSION') and 'ipykernel' in os.environ.get('_', ''):
# MAGIC     # Databricksノートブック内
# MAGIC     import nest_asyncio
# MAGIC     nest_asyncio.apply()
# MAGIC     NOTEBOOK_ENV = True
# MAGIC else:
# MAGIC     # Model Servingやその他の環境
# MAGIC     NOTEBOOK_ENV = False
# MAGIC
# MAGIC # 1) エンドポイント/プロファイルの設定
# MAGIC LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
# MAGIC SYSTEM_PROMPT = "あなたは有能なアシスタントです。"
# MAGIC workspace_client = WorkspaceClient()
# MAGIC host = workspace_client.config.host
# MAGIC
# MAGIC # カタログ・データベース名を設定
# MAGIC # catalog = "takaakiyayoi_catalog"
# MAGIC # db = "rag_chatbot_jpn"
# MAGIC import os
# MAGIC catalog = os.environ.get("CATALOG", "default_catalog")
# MAGIC schema = os.environ.get("SCHEMA", "default_schema")
# MAGIC
# MAGIC # 必要に応じてMCPサーバーURLを追加
# MAGIC MCP_SERVER_URLS = [
# MAGIC     f"{host}/api/2.0/mcp/vector-search/{catalog}/{schema}",
# MAGIC ]
# MAGIC
# MAGIC # 2) ResponsesAgent形式の"message dict"をChatCompletions形式に変換するヘルパー
# MAGIC def _to_chat_messages(msg: dict[str, Any]) -> List[dict]:
# MAGIC     """
# MAGIC     ResponsesAgent形式のdictを1つ以上のChatCompletions互換dictに変換
# MAGIC     """
# MAGIC     msg_type = msg.get("type")
# MAGIC     if msg_type == "function_call":
# MAGIC         return [
# MAGIC             {
# MAGIC                 "role": "assistant",
# MAGIC                 "content": None,
# MAGIC                 "tool_calls": [
# MAGIC                     {
# MAGIC                         "id": msg["call_id"],
# MAGIC                         "type": "function",
# MAGIC                         "function": {
# MAGIC                             "name": msg["name"],
# MAGIC                             "arguments": msg["arguments"],
# MAGIC                         },
# MAGIC                     }
# MAGIC                 ],
# MAGIC             }
# MAGIC         ]
# MAGIC     elif msg_type == "message" and isinstance(msg["content"], list):
# MAGIC         return [
# MAGIC             {
# MAGIC                 "role": "assistant" if msg["role"] == "assistant" else msg["role"],
# MAGIC                 "content": content["text"],
# MAGIC             }
# MAGIC             for content in msg["content"]
# MAGIC         ]
# MAGIC     elif msg_type == "function_call_output":
# MAGIC         return [
# MAGIC             {
# MAGIC                 "role": "tool",
# MAGIC                 "content": msg["output"],
# MAGIC                 "tool_call_id": msg["tool_call_id"],
# MAGIC             }
# MAGIC         ]
# MAGIC     else:
# MAGIC         # {"role": ..., "content": "..."}等のプレーンなdictのフォールバック
# MAGIC         return [
# MAGIC             {
# MAGIC                 k: v
# MAGIC                 for k, v in msg.items()
# MAGIC                 if k in ("role", "content", "name", "tool_calls", "tool_call_id")
# MAGIC             }
# MAGIC         ]
# MAGIC
# MAGIC # 3) MCPセッションとツール呼び出しロジック
# MAGIC @asynccontextmanager
# MAGIC async def _mcp_session(server_url: str, ws: WorkspaceClient):
# MAGIC     async with streamablehttp_client(
# MAGIC         url=server_url, auth=DatabricksOAuthClientProvider(ws)
# MAGIC     ) as (reader, writer, _):
# MAGIC         async with ClientSession(reader, writer) as session:
# MAGIC             await session.initialize()
# MAGIC             yield session
# MAGIC
# MAGIC async def _list_tools_async(server_url: str, ws: WorkspaceClient):
# MAGIC     async with _mcp_session(server_url, ws) as sess:
# MAGIC         return await sess.list_tools()
# MAGIC
# MAGIC def _run_async_in_thread(coroutine):
# MAGIC     """
# MAGIC     非同期コルーチンを専用スレッドのイベントループで実行（Model Serving向け）
# MAGIC     """
# MAGIC     result = None
# MAGIC     exception = None
# MAGIC     
# MAGIC     def run_in_thread():
# MAGIC         nonlocal result, exception
# MAGIC         try:
# MAGIC             # 新しいイベントループを作成
# MAGIC             loop = asyncio.new_event_loop()
# MAGIC             asyncio.set_event_loop(loop)
# MAGIC             try:
# MAGIC                 result = loop.run_until_complete(coroutine)
# MAGIC             finally:
# MAGIC                 loop.close()
# MAGIC         except Exception as e:
# MAGIC             exception = e
# MAGIC     
# MAGIC     # 別スレッドで実行
# MAGIC     thread = threading.Thread(target=run_in_thread)
# MAGIC     thread.start()
# MAGIC     thread.join()
# MAGIC     
# MAGIC     if exception:
# MAGIC         raise exception
# MAGIC     return result
# MAGIC
# MAGIC def _run_async_safely(coroutine):
# MAGIC     """
# MAGIC     環境に応じて非同期コルーチンを安全に実行
# MAGIC     """
# MAGIC     if NOTEBOOK_ENV:
# MAGIC         # ノートブック: 既存イベントループを利用（nest_asyncio適用済み）
# MAGIC         try:
# MAGIC             loop = asyncio.get_running_loop()
# MAGIC             return asyncio.run(coroutine)
# MAGIC         except RuntimeError:
# MAGIC             # フォールバック: スレッド方式
# MAGIC             return _run_async_in_thread(coroutine)
# MAGIC     else:
# MAGIC         # Model Serving: 常にスレッド方式
# MAGIC         return _run_async_in_thread(coroutine)
# MAGIC
# MAGIC def _run_async_in_thread(coroutine):
# MAGIC     """
# MAGIC     非同期コルーチンを専用スレッドのイベントループで実行（Model Serving向け）
# MAGIC     """
# MAGIC     result = None
# MAGIC     exception = None
# MAGIC     
# MAGIC     def run_in_thread():
# MAGIC         nonlocal result, exception
# MAGIC         try:
# MAGIC             loop = asyncio.new_event_loop()
# MAGIC             asyncio.set_event_loop(loop)
# MAGIC             try:
# MAGIC                 result = loop.run_until_complete(coroutine)
# MAGIC             finally:
# MAGIC                 loop.close()
# MAGIC         except Exception as e:
# MAGIC             exception = e
# MAGIC     
# MAGIC     thread = threading.Thread(target=run_in_thread)
# MAGIC     thread.start()
# MAGIC     thread.join()
# MAGIC     
# MAGIC     if exception:
# MAGIC         raise exception
# MAGIC     return result
# MAGIC
# MAGIC def _run_async_safely(coroutine):
# MAGIC     """
# MAGIC     環境に応じて非同期コルーチンを安全に実行
# MAGIC     """
# MAGIC     if NOTEBOOK_ENV:
# MAGIC         try:
# MAGIC             loop = asyncio.get_running_loop()
# MAGIC             return asyncio.run(coroutine)
# MAGIC         except RuntimeError:
# MAGIC             return _run_async_in_thread(coroutine)
# MAGIC     else:
# MAGIC         return _run_async_in_thread(coroutine)
# MAGIC
# MAGIC def _list_tools(server_url: str, ws: WorkspaceClient):
# MAGIC     # 安全な非同期実行
# MAGIC     return _run_async_safely(_list_tools_async(server_url, ws))
# MAGIC
# MAGIC def _make_exec_fn(
# MAGIC     server_url: str, tool_name: str, ws: WorkspaceClient
# MAGIC ) -> Callable[..., str]:
# MAGIC     async def call_it_async(**kwargs):
# MAGIC         async with _mcp_session(server_url, ws) as sess:
# MAGIC             resp = await sess.call_tool(name=tool_name, arguments=kwargs)
# MAGIC             return "".join([c.text for c in resp.content])
# MAGIC     
# MAGIC     def exec_fn(**kwargs):
# MAGIC         # 安全な非同期実行
# MAGIC         return _run_async_safely(call_it_async(**kwargs))
# MAGIC
# MAGIC     return exec_fn
# MAGIC
# MAGIC def _sanitize_tool_name(name: str, max_length: int = 64) -> str:
# MAGIC     """
# MAGIC     Databricks要件に合わせてツール名をサニタイズ
# MAGIC     - 英数字、アンダースコア、ハイフンのみ
# MAGIC     - 最大64文字
# MAGIC     - 正規表現 ^[a-zA-Z0-9_-]{1,64}$ に一致
# MAGIC     """
# MAGIC     import re
# MAGIC     
# MAGIC     # 許可されていない文字をアンダースコアに置換
# MAGIC     sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
# MAGIC     # 連続アンダースコアを1つに
# MAGIC     sanitized = re.sub(r'_+', '_', sanitized)
# MAGIC     # 先頭・末尾のアンダースコアを除去
# MAGIC     sanitized = sanitized.strip('_')
# MAGIC     # 空なら"tool"に
# MAGIC     if not sanitized:
# MAGIC         sanitized = "tool"
# MAGIC     # 長さ制限
# MAGIC     if len(sanitized) <= max_length:
# MAGIC         result = sanitized
# MAGIC     else:
# MAGIC         # 長すぎる場合は分割して短縮
# MAGIC         if "_" in sanitized:
# MAGIC             parts = sanitized.split("_")
# MAGIC             last_part = parts[-1]
# MAGIC             first_part = parts[0]
# MAGIC             available_chars = max_length - len(last_part) - 1
# MAGIC             if available_chars > 0 and len(first_part) <= available_chars:
# MAGIC                 result = f"{first_part}_{last_part}"
# MAGIC             elif available_chars > 0:
# MAGIC                 first_part = first_part[:available_chars]
# MAGIC                 result = f"{first_part}_{last_part}"
# MAGIC             else:
# MAGIC                 result = last_part[:max_length]
# MAGIC         else:
# MAGIC             result = sanitized[:max_length]
# MAGIC     # 最終バリデーション
# MAGIC     pattern = r'^[a-zA-Z0-9_-]{1,64}$'
# MAGIC     if not re.match(pattern, result):
# MAGIC         # 最後の手段: 英数字のみ
# MAGIC         result = re.sub(r'[^a-zA-Z0-9]', '', result)
# MAGIC         if not result:
# MAGIC             result = "tool"
# MAGIC         result = result[:max_length]
# MAGIC     return result
# MAGIC
# MAGIC class ToolInfo(BaseModel):
# MAGIC     name: str
# MAGIC     spec: dict
# MAGIC     exec_fn: Callable
# MAGIC
# MAGIC def _fetch_tool_infos(ws: WorkspaceClient, server_url: str) -> List[ToolInfo]:
# MAGIC     print(f"MCPサーバー {server_url} からツール一覧を取得")
# MAGIC     infos: List[ToolInfo] = []
# MAGIC     try:
# MAGIC         mcp_tools_result = _list_tools(server_url, ws)
# MAGIC         mcp_tools = mcp_tools_result.tools
# MAGIC         
# MAGIC         for t in mcp_tools:
# MAGIC             # ツール名をサニタイズ
# MAGIC             original_name = t.name
# MAGIC             sanitized_name = _sanitize_tool_name(t.name, 64)
# MAGIC             # バリデーション
# MAGIC             import re
# MAGIC             pattern = r'^[a-zA-Z0-9_-]{1,64}$'
# MAGIC             is_valid = re.match(pattern, sanitized_name)
# MAGIC             print(f"元名: '{original_name}'")
# MAGIC             print(f"サニタイズ後: '{sanitized_name}' (長さ: {len(sanitized_name)}, valid: {bool(is_valid)})")
# MAGIC             if not is_valid:
# MAGIC                 print(f"エラー: サニタイズ名がパターンに一致しません!")
# MAGIC                 sanitized_name = "vector_search_tool"
# MAGIC             schema = t.inputSchema.copy() if t.inputSchema else {}
# MAGIC             if "properties" not in schema:
# MAGIC                 schema["properties"] = {}
# MAGIC             # 説明が長すぎる場合は切り詰め
# MAGIC             description = t.description
# MAGIC             if len(description) > 500:
# MAGIC                 description = description[:497] + "..."
# MAGIC             spec = {
# MAGIC                 "type": "function",
# MAGIC                 "function": {
# MAGIC                     "name": sanitized_name,
# MAGIC                     "description": description,
# MAGIC                     "parameters": schema,
# MAGIC                 },
# MAGIC             }
# MAGIC             infos.append(
# MAGIC                 ToolInfo(
# MAGIC                     name=original_name,  # 実行時は元名を使う
# MAGIC                     spec=spec,
# MAGIC                     exec_fn=_make_exec_fn(server_url, original_name, ws)
# MAGIC                 )
# MAGIC             )
# MAGIC         print(f"{len(infos)}個のツールを正常にロード")
# MAGIC     except Exception as e:
# MAGIC         print(f"{server_url} からのツール取得エラー: {e}")
# MAGIC     return infos
# MAGIC
# MAGIC # 4) シングルターン型エージェントクラス
# MAGIC class SingleTurnMCPAgent(ResponsesAgent):
# MAGIC     def __init__(self):
# MAGIC         super().__init__()
# MAGIC         self._tool_infos = None
# MAGIC         self._tools_dict = None
# MAGIC         self._workspace_client = None
# MAGIC         
# MAGIC     def _initialize_tools(self):
# MAGIC         """モデルロード時に一度だけツールを初期化"""
# MAGIC         if self._tool_infos is None:
# MAGIC             try:
# MAGIC                 self._workspace_client = WorkspaceClient()
# MAGIC                 self._tool_infos = [
# MAGIC                     tool_info
# MAGIC                     for mcp_server_url in MCP_SERVER_URLS
# MAGIC                     for tool_info in _fetch_tool_infos(self._workspace_client, mcp_server_url)
# MAGIC                 ]
# MAGIC                 self._tools_dict = {tool_info.name: tool_info for tool_info in self._tool_infos}
# MAGIC                 print(f"モデルロード時に{len(self._tool_infos)}個のツールを初期化")
# MAGIC             except Exception as e:
# MAGIC                 print(f"警告: モデルロード時のツール初期化失敗: {e}")
# MAGIC                 self._tool_infos = []
# MAGIC                 self._tools_dict = {}
# MAGIC     
# MAGIC     def _call_llm(self, history: List[dict], ws: WorkspaceClient, tool_infos):
# MAGIC         """
# MAGIC         現在の履歴をLLMに送信し、生のレスポンスdictを返す
# MAGIC         """
# MAGIC         client = ws.serving_endpoints.get_open_ai_client()
# MAGIC         flat_msgs = []
# MAGIC         for msg in history:
# MAGIC             flat_msgs.extend(_to_chat_messages(msg))
# MAGIC
# MAGIC         # Databricksツール形式に変換
# MAGIC         tools_param = None
# MAGIC         if tool_infos:
# MAGIC             tools_param = []
# MAGIC             for ti in tool_infos:
# MAGIC                 function_spec = ti.spec["function"]
# MAGIC                 tool_dict = {
# MAGIC                     "type": "function",
# MAGIC                     "function": {
# MAGIC                         "name": function_spec["name"],
# MAGIC                         "description": function_spec["description"]
# MAGIC                     }
# MAGIC                 }
# MAGIC                 # パラメータが存在し空でなければ追加
# MAGIC                 if function_spec.get("parameters") and function_spec["parameters"].get("properties"):
# MAGIC                     tool_dict["function"]["parameters"] = function_spec["parameters"]
# MAGIC                 else:
# MAGIC                     # 空のパラメータ仕様
# MAGIC                     tool_dict["function"]["parameters"] = {
# MAGIC                         "type": "object",
# MAGIC                         "properties": {}
# MAGIC                     }
# MAGIC                 tools_param.append(tool_dict)
# MAGIC             # ノートブック環境のみデバッグ出力
# MAGIC             if NOTEBOOK_ENV:
# MAGIC                 print(f"LLMに{len(tools_param)}個のツールを送信")
# MAGIC                 for i, tool in enumerate(tools_param):
# MAGIC                     print(f"ツール {i}: {tool['function']['name']}")
# MAGIC                     import json
# MAGIC                     print(f"ツール構造: {json.dumps(tool, indent=2)}")
# MAGIC
# MAGIC         # 複数アプローチで実行
# MAGIC         try:
# MAGIC             # まずtoolsパラメータ付きで実行
# MAGIC             if tools_param:
# MAGIC                 return client.chat.completions.create(
# MAGIC                     model=LLM_ENDPOINT_NAME,
# MAGIC                     messages=flat_msgs,
# MAGIC                     tools=tools_param,
# MAGIC                 )
# MAGIC             else:
# MAGIC                 return client.chat.completions.create(
# MAGIC                     model=LLM_ENDPOINT_NAME,
# MAGIC                     messages=flat_msgs,
# MAGIC                 )
# MAGIC         except Exception as e:
# MAGIC             if NOTEBOOK_ENV:
# MAGIC                 print(f"最初の試行失敗: {e}")
# MAGIC                 print("ツールなしでフォールバック...")
# MAGIC             # フォールバック: ツールなしで実行
# MAGIC             return client.chat.completions.create(
# MAGIC                 model=LLM_ENDPOINT_NAME,
# MAGIC                 messages=flat_msgs,
# MAGIC             )
# MAGIC
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         # 未初期化ならツールを初期化
# MAGIC         self._initialize_tools()
# MAGIC         
# MAGIC         ws = self._workspace_client or WorkspaceClient()
# MAGIC
# MAGIC         # 1) system+userで初期履歴を構築
# MAGIC         history: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
# MAGIC         for inp in request.input:
# MAGIC             history.append(inp.model_dump())
# MAGIC
# MAGIC         # 2) LLMを一度呼び出し
# MAGIC         try:
# MAGIC             # 事前ロード済みツールを利用
# MAGIC             tool_infos = self._tool_infos
# MAGIC             tools_dict = self._tools_dict
# MAGIC             
# MAGIC             if NOTEBOOK_ENV:
# MAGIC                 print(f"事前ロード済みツール数: {len(tool_infos)}")
# MAGIC             
# MAGIC             llm_resp = self._call_llm(history, ws, tool_infos)
# MAGIC             raw_choice = llm_resp.choices[0].message.to_dict()
# MAGIC             raw_choice["id"] = uuid.uuid4().hex
# MAGIC             history.append(raw_choice)
# MAGIC
# MAGIC             tool_calls = raw_choice.get("tool_calls") or []
# MAGIC             if tool_calls:
# MAGIC                 # （この例では単一ツールのみ対応）
# MAGIC                 fc = tool_calls[0]
# MAGIC                 requested_name = fc["function"]["name"]
# MAGIC                 args = json.loads(fc["function"]["arguments"])
# MAGIC                 # サニタイズ名から元名を検索
# MAGIC                 original_name = None
# MAGIC                 for tool_info in tool_infos:
# MAGIC                     if tool_info.spec["function"]["name"] == requested_name:
# MAGIC                         original_name = tool_info.name
# MAGIC                         break
# MAGIC                 if original_name and original_name in tools_dict:
# MAGIC                     try:
# MAGIC                         tool_info = tools_dict[original_name]
# MAGIC                         result = tool_info.exec_fn(**args)
# MAGIC                     except Exception as e:
# MAGIC                         result = f"{original_name}の呼び出しエラー: {e}"
# MAGIC                 else:
# MAGIC                     result = f"ツール {requested_name} が見つかりません"
# MAGIC                 # 4) "tool"出力を履歴に追加
# MAGIC                 history.append(
# MAGIC                     {
# MAGIC                         "type": "function_call_output",
# MAGIC                         "role": "tool",
# MAGIC                         "id": uuid.uuid4().hex,
# MAGIC                         "tool_call_id": fc["id"],
# MAGIC                         "output": result,
# MAGIC                     }
# MAGIC                 )
# MAGIC                 # 5) LLMを再度呼び出し、その返答を最終とする
# MAGIC                 followup = (
# MAGIC                     self._call_llm(history, ws, tool_infos=[]).choices[0].message.to_dict()
# MAGIC                 )
# MAGIC                 followup["id"] = uuid.uuid4().hex
# MAGIC
# MAGIC                 assistant_text = followup.get("content", "")
# MAGIC                 return ResponsesAgentResponse(
# MAGIC                     output=[
# MAGIC                         {
# MAGIC                             "id": uuid.uuid4().hex,
# MAGIC                             "type": "message",
# MAGIC                             "role": "assistant",
# MAGIC                             "content": [{"type": "output_text", "text": assistant_text}],
# MAGIC                         }
# MAGIC                     ],
# MAGIC                     custom_outputs=request.custom_inputs,
# MAGIC                 )
# MAGIC
# MAGIC             # 6) tool_callsがなければ元のassistant返答を返す
# MAGIC             assistant_text = raw_choice.get("content", "")
# MAGIC             return ResponsesAgentResponse(
# MAGIC                 output=[
# MAGIC                     {
# MAGIC                         "id": uuid.uuid4().hex,
# MAGIC                         "type": "message",
# MAGIC                         "role": "assistant",
# MAGIC                         "content": [{"type": "output_text", "text": assistant_text}],
# MAGIC                     }
# MAGIC                 ],
# MAGIC                 custom_outputs=request.custom_inputs,
# MAGIC             )
# MAGIC         
# MAGIC         except Exception as e:
# MAGIC             # エラー処理
# MAGIC             error_message = f"リクエスト処理中のエラー: {str(e)}"
# MAGIC             print(error_message)
# MAGIC             return ResponsesAgentResponse(
# MAGIC                 output=[
# MAGIC                     {
# MAGIC                         "id": uuid.uuid4().hex,
# MAGIC                         "type": "message",
# MAGIC                         "role": "assistant",
# MAGIC                         "content": [{"type": "output_text", "text": error_message}],
# MAGIC                     }
# MAGIC                 ],
# MAGIC                 custom_outputs=request.custom_inputs,
# MAGIC             )
# MAGIC
# MAGIC # MLflowモデルをセット
# MAGIC mlflow.models.set_model(SingleTurnMCPAgent())
# MAGIC
# MAGIC # テスト実行
# MAGIC try:
# MAGIC     print("エージェントリクエスト作成中...")
# MAGIC     req = ResponsesAgentRequest(
# MAGIC         input=[{"role": "user", "content": "Databricksとは？"}]
# MAGIC     )
# MAGIC     
# MAGIC     print("予測実行中...")
# MAGIC     agent = SingleTurnMCPAgent()
# MAGIC     resp = agent.predict(req)
# MAGIC     
# MAGIC     print("レスポンス:")
# MAGIC     for item in resp.output:
# MAGIC         print(item)
# MAGIC         
# MAGIC except Exception as e:
# MAGIC     print(f"実行中のエラー: {e}")
# MAGIC     import traceback
# MAGIC     traceback.print_exc()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. MCPを使用してエージェントをデプロイする

# COMMAND ----------

# MAGIC %md
# MAGIC MCPサーバーに接続するエージェントをデプロイする準備ができたら、[標準のエージェントデプロイメントプロセス](https://docs.databricks.com/aws/ja/generative-ai/agent-framework/deploy-agent)を使用してください。<br>
# MAGIC
# MAGIC [エージェントがアクセスする必要があるすべてのリソースをログイン時に指定する](https://docs.databricks.com/aws/ja/generative-ai/agent-framework/log-agent#authentication-for-databricks-resources)ことを確認してください。<br>
# MAGIC 例えば、エージェントが以下のMCPサーバーURLを使用する場合：<br>
# MAGIC `https://<your-workspace-hostname>/api/2.0/mcp/vector-search/prod/customer_support`<br>
# MAGIC `https://<your-workspace-hostname>/api/2.0/mcp/vector-search/prod/billing`<br>
# MAGIC `https://<your-workspace-hostname>/api/2.0/mcp/functions/prod/billing`<br>
# MAGIC エージェントが必要とするすべてのベクトル検索インデックス、およびすべてのUnity Catalog関数をリソースとして指定する必要があります。<br>
# MAGIC
# MAGIC エージェントが必要とするすべてのベクトル検索インデックス、およびすべてのUnity Catalog関数をリソースとして指定する必要があります。<br>
# MAGIC 例えば、上記で定義されたエージェントをデプロイするには、エージェントコード定義をmcp_agent.pyに保存したと仮定して、次のスニペットを実行できます。<br>
# MAGIC Pythonカーネルを再起動してimportするファイルを認識できるようにします。
# MAGIC

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# DBTITLE 1,環境変数を設定
# 環境変数を設定
import os
os.environ["CATALOG"] = catalog
os.environ["SCHEMA"] = schema

# COMMAND ----------

# DBTITLE 1,MCPエージェントをサービングエンドポイントにデプロイ
import os
from databricks.sdk import WorkspaceClient
from databricks import agents
import mlflow
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint, DatabricksVectorSearchIndex
from mcp_agent import LLM_ENDPOINT_NAME

workspace_client = WorkspaceClient()

# mcp_agent.pyで定義されたエージェントをログ
agent_script = "mcp_agent.py"
resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME),
    # --- エージェントコード内のMCP_SERVER_URLSを介して参照される場合、以下の行をアンコメントしてベクトル検索インデックスや追加のUC関数を指定 ---
    DatabricksVectorSearchIndex(index_name=f"{catalog}.{schema}.gold_feedbacks_index"),
    # DatabricksVectorSearchIndex(index_name="prod.billing.another_index"),
    # DatabricksFunction(f"{catalog}.{schema}.get_store_product_inventory"),
    # DatabricksFunction(f"{catalog}.{schema}.get_store_item_sales_ranking"),
    # DatabricksFunction(f"{catalog}.{schema}.get_store_sales_ranking"),
]

with mlflow.start_run():
    logged_model_info = mlflow.pyfunc.log_model(
        name="mcp_agent",
        python_model=agent_script,
        resources=resources,
    )

# TODO UCモデル名をここに指定
UC_MODEL_NAME = f"{catalog}.{schema}.bricksmart_analysis_mcp_agent"
registered_model = mlflow.register_model(logged_model_info.model_uri, UC_MODEL_NAME)

deployment_info = agents.deploy(
    model_name=UC_MODEL_NAME,
    model_version=registered_model.version,
)


# COMMAND ----------

# DBTITLE 1,UCからモデル削除
from mlflow.tracking import MlflowClient

client = MlflowClient()
for model_name in [f"{catalog}.{schema}.bricksmart_analysis_mcp_agent", f"{catalog}.{schema}.feedback"]:
    try:
        client.get_registered_model(model_name)
        client.delete_registered_model(model_name)
    except Exception:
        pass  # モデルがなければ何もしない
