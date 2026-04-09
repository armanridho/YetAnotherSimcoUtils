import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from openai import OpenAI
from ollamafreeapi import OllamaFreeAPI
from concurrent.futures import ThreadPoolExecutor
import io

st.set_page_config(page_title="Buyer Intelligence • Red Forrest Inc.", layout="wide", page_icon="🔥")

st.title("🔥 Buyer Intelligence • Red Forrest Inc. v3.1")
st.markdown("**AI-Powered • Smart Analytics • Multi AI Backup**")

@st.cache_resource
def get_ai_clients():
    return {
        "primary": OpenAI(base_url="https://qwen.ai.unturf.com/v1", api_key="choose-any-value"),
        "backup": OllamaFreeAPI()
    }

def call_with_timeout(func, timeout_sec=6):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout_sec)
        except:
            return None

def ask_ai_analyst(question, context_data=""):
    clients = get_ai_clients()
    prompt = f"""Kamu adalah Senior Buyer Intelligence Analyst di Red Forrest Inc.
Analisis data buyer, contract sell, profitability, risiko, dan berikan rekomendasi actionable.

Data yang tersedia:
{context_data[:12000]}

Pertanyaan: {question}

Jawab secara profesional, tajam, dan langsung ke poin."""

    def primary_call():
        response = clients["primary"].chat.completions.create(
            model="hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.65,
            max_tokens=1500
        )
        return response.choices[0].message.content

    result = call_with_timeout(primary_call, 6)
    if result:
        return result

    def backup_call():
        response = clients["backup"].chat(
            model="deepseek-r1:latest",
            prompt=prompt,
            temperature=0.7
        )
        return response

    result = call_with_timeout(backup_call, 12)
    if result:
        return result

    return "❌ Kedua AI sedang mengalami kendala. Coba lagi dalam 10-20 detik."

uploaded_files = st.file_uploader(
    "Upload semua CSV sekaligus (sand, transport, power, account, warehouse)",
    type="csv", accept_multiple_files=True
)

if uploaded_files:
    dfs = {}
    for file in uploaded_files:
        df = pd.read_csv(file)
        fn = file.name.lower()
        if "sand" in fn: dfs["sand"] = df
        elif "transport" in fn: dfs["transport"] = df
        elif "power" in fn: dfs["power"] = df
        elif "account" in fn or "history" in fn: dfs["account"] = df
        elif "warehouse" in fn: dfs["warehouse"] = df

    st.success(f"✅ {len(uploaded_files)} file berhasil di-load!")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Buyer Intelligence", "⚡ Quick Contract", "🏆 Profitability", 
        "📈 Daily Operation", "⚡ Power Intelligence", "📦 Warehouse", "🤖 AI Analyst"
    ])

    sell_df = pd.DataFrame()
    if "sand" in dfs:
        sand_df = dfs["sand"].copy()
        sand_df["Timestamp"] = pd.to_datetime(sand_df["Timestamp"], utc=True, errors='coerce')
        sell_df = sand_df[(sand_df["Category"] == "Contract sell") & (sand_df["Resource"] == "Sand")].copy()
        sell_df["Amount"] = sell_df["Amount"].abs()

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📋 Last 48 Hours Contracts")
            if not sell_df.empty:
                recent = sell_df[sell_df["Timestamp"] >= (pd.Timestamp.now(tz='UTC') - timedelta(hours=48))]
                st.dataframe(recent[["Timestamp", "Other Party", "Amount", "Quality"]], use_container_width=True)
        with col2:
            st.subheader("🚨 Smart Alerts")
            if not sell_df.empty:
                buyer_last = sell_df.groupby("Other Party").agg(
                    Last_Contract=("Timestamp", "max"),
                    Last_Qty=("Amount", "last"),
                    Last_Quality=("Quality", "last")
                ).reset_index()
                buyer_last["Hours_Ago"] = (pd.Timestamp.now(tz='UTC') - buyer_last["Last_Contract"]).dt.total_seconds() / 3600
                for _, row in buyer_last[buyer_last["Hours_Ago"] > 24].iterrows():
                    st.error(f"🚨 **{row['Other Party']}** — sudah {int(row['Hours_Ago'])} jam tidak beli!")

    with tab2:
        st.subheader("⚡ Quick Contract Generator")
        if not sell_df.empty:
            buyers = sorted(sell_df["Other Party"].unique())
            selected = st.selectbox("Pilih Buyer", buyers)
            buyer_data = sell_df[sell_df["Other Party"] == selected]
            if not buyer_data.empty:
                top = buyer_data.groupby(["Amount", "Quality"]).size().idxmax()
                qty, quality = top
                contract_text = f"Contract sell Sand {int(qty)} Q{int(quality)} ke {selected}"
                st.code(contract_text, language=None)
                if st.button("Copy Contract"):
                    st.success("✅ Contract copied!")

    with tab3:
        st.subheader("🏆 Buyer Profitability Ranking")
        if not sell_df.empty:
            sell_df["Est_Profit"] = sell_df["Amount"] * 0.42
            ranking = sell_df.groupby("Other Party").agg(
                Total_Sand=("Amount", "sum"),
                Est_Profit=("Est_Profit", "sum")
            ).sort_values("Est_Profit", ascending=False).reset_index()
            ranking["Est_Profit"] = ranking["Est_Profit"].apply(lambda x: f"${x:,.0f}")
            st.dataframe(ranking, use_container_width=True)

    with tab4:
        st.subheader("📈 Daily Operation")
        if "sand" in dfs:
            today = pd.Timestamp.now(tz='UTC').floor('D')
            prod_today = sand_df[(sand_df["Category"] == "Production") & 
                               (sand_df["Timestamp"].dt.date == today.date())]["Amount"].sum()
            st.metric("Sand Produksi Hari Ini", f"{prod_today:,.0f} unit")

    with tab5:
        st.subheader("⚡ Power Intelligence")
        if "power" in dfs:
            power_df = dfs["power"].copy()
            power_df["Timestamp"] = pd.to_datetime(power_df["Timestamp"], utc=True, errors='coerce')
            st.plotly_chart(px.line(power_df, x="Timestamp", y="Amount", color="Category", title="Power Movement"), use_container_width=True)

    with tab6:
        st.subheader("📦 Warehouse & Reports")
        if "warehouse" in dfs:
            st.dataframe(dfs["warehouse"], use_container_width=True)

    with tab7:
        st.subheader("🤖 AI Analyst — Qwen + OllamaFreeAPI Backup")
        question = st.text_area("Tanya AI Analyst:", 
            "Analisis buyer mana yang paling menguntungkan dan rekomendasi kontrak untuk 7 hari ke depan?")
        
        if st.button("🔍 Ask AI Analyst"):
            if not sell_df.empty:
                context = sell_df.to_string()
                with st.spinner("AI sedang menganalisis data..."):
                    answer = ask_ai_analyst(question, context)
                    st.markdown(answer)
            else:
                st.warning("Upload data sand.csv terlebih dahulu")

    st.caption("v3.1 BEAST MODE • AI Backup System • Red Forrest Inc.")
else:
    st.info("👆 Upload semua file CSV untuk memulai analisis")