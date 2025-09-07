import os, streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Family Rules Bot", page_icon="👪")
st.title("家族のルールの確認アプリ")

# ← ここで変数名を my_api_key に
my_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
#my_project = st.secrets.get("OPENAI_PROJECT", os.getenv("OPENAI_PROJECT"))

if not my_api_key:
    st.error("OPENAI_API_KEY が未設定です。ローカルは .streamlit/secrets.toml、Cloudは Settings→Secrets に保存してください。")
    st.stop()
else:
    client = OpenAI(api_key=my_api_key)   # ← 引数名は api_key のまま

msg = st.chat_input("質問を入力してね")
if msg:
    with st.chat_message("user"):
        st.write(msg)

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "あなたは家庭のルールをやさしく説明するアシスタントです。関西弁で答えてね"},
            {"role": "user", "content": msg},
        ],
    )
    ans = res.choices[0].message.content

    with st.chat_message("assistant"):
        st.write(ans)

st.caption("Powered by OpenAI (model: gpt-4o-mini)")
