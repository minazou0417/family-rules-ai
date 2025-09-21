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

# === ここから 画面切り替え ===
st.divider()
mode = st.radio("モードを選んでください", ["チャット", "ルール管理"], horizontal=True)
st.divider()

if mode == "ルール管理":
    # --- ルール管理（確認・追加・編集・削除） ---
    st.subheader("ルール管理")

    # 一覧表示（確認）
    if rules:
        st.caption(f"登録ルール数: {len(rules)}")
        for i, r in enumerate(rules, start=1):
            st.write(f"{i:>2}. {r}")
    else:
        st.info("まだルールがありません。下の欄から追加できます。")

    st.divider()

    # 追加
    st.write("**追加**")
    new_rule = st.text_input("新しいルール（例：おやつは午後3時までだよ。）", key="new_rule")
    if st.button("追加する", use_container_width=True):
        txt = (new_rule or "").strip()
        if not txt:
            st.warning("ルール文を入力してください。")
        else:
            if not txt.startswith("ルール："):
                line = "ルール： " + txt
            else:
                line = txt
                txt = line[len("ルール："):].strip()

            # rules.txt に追記
            os.makedirs(os.path.dirname(rules_path), exist_ok=True)
            with open(rules_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

            # in-memory へ反映
            content = line[len("ルール："):].strip()
            rules.append(content)

            # 埋め込み1件だけ作成してFAISSに追加（初回は新規作成）
            emb = client.embeddings.create(model=embed_model, input=[content]).data[0].embedding
            v = np.array(emb, dtype="float32")
            v = v / (np.linalg.norm(v) + 1e-9)
            if index is None:
                d = v.shape[0]
                index = faiss.IndexFlatIP(d)
                index.add(v[None, :])
            else:
                index.add(v[None, :])

            # 保存
            faiss.write_index(index, index_path)
            json.dump({"model": embed_model, "rules": rules}, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False)
            st.success("ルールを追加しました。")
            st.rerun()

    st.divider()

    # 編集
    st.write("**編集**")
    if rules:
        edit_i = st.selectbox(
            "編集するルールを選択",
            options=list(range(len(rules))),
            format_func=lambda i: f"{i+1}. {rules[i][:40]}{'…' if len(rules[i])>40 else ''}",
            key="edit_select",
        )
        edit_default = "ルール： " + rules[edit_i]
        edit_text = st.text_input("編集後の内容（先頭の『ルール：』は無くてもOK）", value=edit_default, key="edit_text")

        if st.button("更新する", use_container_width=True):
            t = (edit_text or "").strip()
            if not t:
                st.warning("内容を入力してください。")
            else:
                if t.startswith("ルール："):
                    new_content = t[len("ルール："):].strip()
                else:
                    new_content = t

                # in-memory 更新
                rules[edit_i] = new_content

                # rules.txt 全件書き直し
                with open(rules_path, "w", encoding="utf-8") as f:
                    for r in rules:
                        f.write("ルール： " + r + "\n")

                # 全件再ベクトル化→FAISS再構築（編集は安全のため全再構築）
                emb = client.embeddings.create(model=embed_model, input=rules)
                X = np.array([d.embedding for d in emb.data], dtype="float32")
                X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
                d = X.shape[1]
                index = faiss.IndexFlatIP(d)
                index.add(X)

                # 保存
                faiss.write_index(index, index_path)
                json.dump({"model": embed_model, "rules": rules}, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False)
                st.success("ルールを更新しました。")
                st.rerun()
    else:
        st.caption("（編集対象がありません）")

    st.divider()

    # 削除
    st.write("**削除**")
    if rules:
        del_i = st.selectbox(
            "削除するルールを選択",
            options=list(range(len(rules))),
            format_func=lambda i: f"{i+1}. {rules[i][:40]}{'…' if len(rules[i])>40 else ''}",
            key="del_select",
        )
        if st.button("削除する", use_container_width=True):
            # in-memory から除外
            rules.pop(del_i)

            # rules.txt 全件書き直し
            with open(rules_path, "w", encoding="utf-8") as f:
                for r in rules:
                    f.write("ルール： " + r + "\n")

            # 0件なら index を破棄、あれば全再構築
            if not rules:
                index = None
                try:
                    os.remove(index_path)
                except Exception:
                    pass
                json.dump({"model": embed_model, "rules": []}, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False)
            else:
                emb = client.embeddings.create(model=embed_model, input=rules)
                X = np.array([d.embedding for d in emb.data], dtype="float32")
                X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
                d = X.shape[1]
                index = faiss.IndexFlatIP(d)
                index.add(X)
                faiss.write_index(index, index_path)
                json.dump({"model": embed_model, "rules": rules}, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False)

            st.success("ルールを削除しました。")
            st.rerun()
    else:
        st.caption("（削除対象がありません）")

else:
    # --- チャット（質問 → 分類 → RAG → 応答） ---
    msg = st.chat_input("質問を入力してね（例：おやつはいつ？）")
    if msg:
        with st.chat_message("user"):
            st.write(msg)

        # ① 家庭のルールに関する質問か？（YES/NO）
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
            # 一般チャット
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
            # RAG 可否判定 → 検索 → 応答
            use_fallback = False
            best_i, best_sim = -1, 0.0

            if not rules or index is None:
                use_fallback = True
            else:
                q = client.embeddings.create(model=embed_model, input=[msg]).data[0].embedding
                q = np.array(q, dtype="float32")[None, :]
                q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
                D, I = index.search(q, 1)
                best_sim, best_i = float(D[0][0]), int(I[0][0])
                TH = 0.65
                if best_sim < TH:
                    use_fallback = True

            if use_fallback:
                system = (
                    "あなたは家庭内ルールをやさしく説明するアシスタントです。"
                    "断定は避け、子どもにも分かる短い文で答えてください。"
                    "最後に『おうちの人に確認してね』と添えてください。"
                )
                prompt = f"質問: {msg}\n\n登録された家庭ルールは見つかりませんでした。一般的に無難で安全な目安を1〜2文で伝えてください。"
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.4,
                    messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
                )
                with st.chat_message("assistant"):
                    st.write(res.choices[0].message.content)
            else:
                rule_text = rules[best_i]
                system = (
                    "あなたは家庭内ルールをやさしく
                    説明するアシスタントです。"
                    "子どもにも分かる短い文で答えてください。"
                )
                context = f"家庭のルール（該当）:\n- {rule_text}"
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.2,
                    messages=[
                        {"role":"system","content": system},
                        {"role":"user", "content": f"{context}\n\n質問: {msg}\n\n上のルールに基づいて、1〜2文で答えて。"},
                    ],
                )
                with st.chat_message("assistant"):
                    st.write(res.choices[0].message.content)
# === ここまで 画面切り替え ===
