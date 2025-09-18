import os, json, numpy as np, streamlit as st, faiss
from openai import OpenAI

st.set_page_config(page_title="Family Rules Bot", page_icon="👪")
st.title("家族のルールの確認アプリ")

# --- 認証 ---
my_api_key = (st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")) or "").strip()
if not my_api_key:
    st.error("OPENAI_API_KEY が未設定です。ローカルは .streamlit/secrets.toml、Cloud は Settings→Secrets に保存してください。")
    st.stop()
client = OpenAI(api_key=my_api_key)

# --- パス・モデル ---
base = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base, "data")
vs_dir   = os.path.join(base, "vectorstore")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(vs_dir,   exist_ok=True)

rules_path = os.path.join(data_dir, "rules.txt")
index_path = os.path.join(vs_dir,   "faiss.index")
meta_path  = os.path.join(vs_dir,   "meta.json")
embed_model = "text-embedding-3-small"

# --- rules.txt 読み込み（先頭が「ルール：」の行だけ採用） ---
rules_texts = []
if os.path.exists(rules_path):
    with open(rules_path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if s and not s.startswith("#") and s.startswith("ルール："):
                content = s[len("ルール："):].strip()
                if content:
                    rules_texts.append(content)
else:
    st.warning("data/rules.txt がありません。例）ルール： おやつは午後3時までだよ。")

# --- インデックス：あれば読む／なければ作る ---
rules, index = [], None
if os.path.exists(index_path) and os.path.exists(meta_path):
    try:
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        if meta.get("model") == embed_model:
            index = faiss.read_index(index_path)
            rules = list(meta.get("rules", []))   # 「ルール：」抜き本文の配列
            st.caption("FAISS: loaded existing index")
    except Exception:
        index = None

if index is None:
    rules = list(rules_texts)
    if rules:
        emb = client.embeddings.create(model=embed_model, input=rules)
        X = np.array([d.embedding for d in emb.data], dtype="float32")
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)   # コサイン用に正規化
        d = X.shape[1]
        index = faiss.IndexFlatIP(d)                              # 内積＝コサイン類似度
        index.add(X)
        faiss.write_index(index, index_path)
        json.dump({"model": embed_model, "rules": rules}, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False)
        st.caption("FAISS: built & saved")

# --- 質問 → 分類 →（必要なら）RAG → 応答 ---
msg = st.chat_input("質問を入力してね（例：おやつはいつ？）")
if msg:
    with st.chat_message("user"):
        st.write(msg)

    # ① 家庭のルールに関する質問？（YES/NOだけ出す）
    cls_system = (
        "あなたは短い判定アシスタントです。"
        "入力が『家庭内のルール（していい/だめ、何時まで、どれくらい等）』に関する質問なら YES、"
        "そうでなければ NO。出力は必ず YES か NO のみ。"
    )
    cls = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role":"system","content":cls_system},{"role":"user","content": msg}],
    ).choices[0].message.content.strip().upper()

    if cls != "YES":
        # ルール質問ではない → 一般的なチャット応答
        gen = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.6,
            messages=[
                {"role":"system","content":"あなたは親切なアシスタントです。子どもにも分かる表現で簡潔に答えてください。"},
                {"role":"user","content": msg},
            ],
        )
        with st.chat_message("assistant"):
            st.write(gen.choices[0].message.content)
        st.caption("(general chat)")
    else:
        # ルール質問 → RAG 可否を判定
        use_fallback = False
        fb_reason = ""
        best_i, best_sim = -1, 0.0

        if not rules or index is None:
            use_fallback = True
            fb_reason = "no rules/index"
        else:
            # ベクトル検索（Top1）
            q = client.embeddings.create(model=embed_model, input=[msg]).data[0].embedding
            q = np.array(q, dtype="float32")[None, :]
            q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
            D, I = index.search(q, 1)
            best_sim, best_i = float(D[0][0]), int(I[0][0])
            TH = 0.65
            if best_sim < TH:
                use_fallback = True
                fb_reason = f"no match, sim: {best_sim:.2f}"

        if use_fallback:
            # RAG不可 or 該当ルールなし → 一般的な目安で回答（共通フォールバック）
            system = (
                "あなたは家庭内ルールをやさしく説明するアシスタントです。"
                "断定は避け、子どもにも分かる短い文で答えてください。"
                "最後に『おうちの人に確認してね』と添えてください。"
            )
            prompt = f"質問: {msg}\n\n一般的な目安を1〜2文で伝えてください。"
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.4,
                messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
            )
            with st.chat_message("assistant"):
                st.write(res.choices[0].message.content)
                if fb_reason:
                    st.caption(f"({fb_reason})")
        else:
            # 該当ルールあり → ルールを根拠に回答
            rule_text = rules[best_i]
            system = "あなたは家庭内ルールをやさしく説明するアシスタントです。子供にもわかる短い文で答えてください。"
            context = f"家庭のルール（該当）:\n- {rule_text}"
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                messages=[
                    {"role":"system","content": system},
                    {"role":"user", "content": f"{context}\n\n質問: {msg}\n\n上のルールに基づいて、子どもにも分かる言葉で1〜2文で答えて。"},
                ],
            )
            with st.chat_message("assistant"):
                st.write(res.choices[0].message.content)
                st.caption(f"(match: {rule_text[:20]}… , sim: {best_sim:.2f})")


st.caption("Powered by OpenAI (RAG: FAISS + text-embedding-3-small)")