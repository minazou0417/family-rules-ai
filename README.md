# Family Rules Bot（家族のルールの確認アプリ）

家庭のルールを登録して、子どもの質問に AI がやさしく答えるアプリです。

## デモ
- URL: https://family-rules-ai-r7jsirvcqxtqhcdbencl5k.streamlit.app/
- デモ認証: なし（誰でもアクセス可）

## 要件（概要）
- 目的: 保護者は、気軽にルールを確認できる環境を作れるようになる。子どもは、おうちのルールをすぐ確認できるようになる。
- 対象: 保護者・子ども
- 流れ: ルール登録 → チャットで質問 → ルールに基づき回答（なければ一般的な目安）

## 機能
- チャット回答（関西弁・短文）
- ルールの追加 / 編集 / 削除 / 一覧
- 簡易RAG検索（ベクトル類似度 Top1）
- 画面切替（チャット / ルール管理）

## 使っている技術
- Streamlit
- OpenAI API（gpt-4o-mini, text-embedding-3-small）
- FAISS
- NumPy