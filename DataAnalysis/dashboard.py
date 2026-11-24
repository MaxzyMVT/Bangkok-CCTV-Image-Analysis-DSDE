"""Explore the committed rain analysis: streamlit run DataAnalysis/dashboard.py."""

from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "DataAnalysis" / "output"
FIGURES = ROOT / "DataAnalysis" / "figures"


@st.cache_data(show_spinner=False)
def table(name):
    return pd.read_csv(OUTPUT / f"{name}.csv", encoding="utf-8-sig")


def pct(value):
    return f"{value:+.1f}%"


def show_figure(name, caption):
    path = FIGURES / f"{name}.png"
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")


st.set_page_config(page_title="Bangkok rain and traffic", layout="wide")
st.title("ฝนเปลี่ยนการจราจรกรุงเทพฯ อย่างไร")
st.caption("100 กล้อง CCTV · 25–31 ส.ค. 2026 · ผลวิเคราะห์จาก DataAnalysis/rain บน main ล่าสุด")

required = ["camera", "camera_hour", "city_effect", "camera_effect", "natural_experiment"]
missing = [name for name in required if not (OUTPUT / f"{name}.csv").is_file()]
if missing:
    st.error("ไม่พบตารางผลวิเคราะห์: " + ", ".join(missing) + " — รัน python DataAnalysis/rain/run_all.py")
    st.stop()

city = table("city_effect")
overall = city[city["window"] == "all"].set_index("cls")
hours = table("camera_hour")
hours["hour_ts"] = pd.to_datetime(hours["hour_ts"])
hours["date"] = hours["hour_ts"].dt.date

st.info("ผลหลัก: ในชั่วโมงที่ฝนตก รถจักรยานยนต์ในภาพลดลงราว 21% เมื่อเทียบกับกล้องเดิม ชั่วโมงเดิม และวันชนิดเดียวกัน (วันธรรมดา/สุดสัปดาห์) ส่วนรถยนต์ไม่พบการเปลี่ยนแปลงที่ชัดเจน")
cols = st.columns(3)
for col, cls, label in zip(cols, ("total", "car", "motorcycle"), ("รถทั้งหมด", "รถยนต์", "จักรยานยนต์")):
    row = overall.loc[cls]
    col.metric(label, pct(row["effect_pct"]))
    col.caption(f"95% CI {pct(row['ci_low'])} ถึง {pct(row['ci_high'])} · {int(row['n_wet_hours'])} ชั่วโมงกล้องที่ฝนตก")

tab_overview, tab_time, tab_map, tab_hypotheses, tab_data = st.tabs(
    ["ผลหลัก", "ช่วงเวลา", "พื้นที่", "ตรวจสมมติฐาน", "ข้อมูลและข้อจำกัด"])

with tab_overview:
    show_figure("city_effect", "ผลต่างเมื่อฝนตก แยกประเภทรถและช่วงเวลา; ช่วงความเชื่อมั่นคำนวณใน pipeline เดิม")
    st.write("ตารางผลที่ใช้สร้างตัวเลขด้านบน")
    st.dataframe(city, width="stretch", hide_index=True)
    st.caption("รถต่อภาพคือความหนาแน่นที่กล้องมองเห็น ไม่ใช่ความเร็วหรือปริมาณรถที่ผ่านจุดนั้น")

with tab_time:
    dates = sorted(hours["date"].unique())
    chosen = st.multiselect("เลือกวันที่", dates, default=dates)
    metric = st.selectbox("ประเภทรถ", ["total", "car", "motorcycle", "bus", "truck"],
                          format_func=lambda x: {"total": "รถทั้งหมด", "car": "รถยนต์", "motorcycle": "จักรยานยนต์", "bus": "รถบัส", "truck": "รถบรรทุก"}[x])
    subset = hours[hours["date"].isin(chosen)]
    if not subset.empty:
        # Equal weight per active camera in each hour; n_frames remains visible below.
        trend = subset.groupby("hour_ts", as_index=False).agg(
            vehicles_per_frame=(metric, "mean"), mean_rain_mm=("rain_mm", "mean"),
            cameras=("camera_id", "nunique"), frames=("n_frames", "sum"))
        st.write("จำนวนรถเฉลี่ยต่อภาพในแต่ละชั่วโมง (เฉลี่ยค่าของแต่ละกล้องก่อน)")
        st.line_chart(trend.set_index("hour_ts")["vehicles_per_frame"])
        st.write("ฝนเฉลี่ยที่สถานีวัดใกล้กล้องในชั่วโมงเดียวกัน")
        st.line_chart(trend.set_index("hour_ts")["mean_rain_mm"])
        st.dataframe(trend, width="stretch", hide_index=True)
    else:
        st.info("เลือกอย่างน้อยหนึ่งวันที่มีข้อมูล")
    show_figure("rebound", "หลังฝนหยุด: อัตราส่วนเทียบระดับปกติของกล้องและช่วงเวลาเดียวกัน")
    st.caption("เวลาฝนเป็นข้อมูลรายชั่วโมง จึงบอกนาทีที่ฝนหยุดจริงไม่ได้")

with tab_map:
    cameras = table("camera").merge(table("camera_effect"), on="camera_id", how="left")
    labels = st.multiselect("ผลของแต่ละกล้อง", sorted(cameras["label"].dropna().unique()),
                            default=sorted(cameras["label"].dropna().unique()))
    min_wet = st.slider("ชั่วโมงฝนขั้นต่ำ", 0, int(cameras["n_wet_hours"].max()), 0)
    filtered = cameras[cameras["label"].isin(labels) & (cameras["n_wet_hours"] >= min_wet)].copy()
    if not filtered.empty:
        st.map(filtered.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]])
        st.dataframe(filtered.sort_values("effect_pct"), width="stretch", hide_index=True)
    else:
        st.info("ไม่มีจุดที่ตรงกับตัวกรอง")
    show_figure("camera_map", "ผลแต่ละกล้องและตำแหน่งรถไฟฟ้า; การจัดกลุ่มที่ชัดเจนต้องมีฝนอย่างน้อย 3 ชั่วโมง")
    st.caption("ระยะใกล้รถไฟฟ้าในตารางเป็นระยะเส้นตรง ไม่ใช่ระยะเดินทาง")

with tab_hypotheses:
    st.subheader("เปรียบเทียบกล้องที่ฝนตกกับกล้องแห้งใกล้กันในชั่วโมงเดียวกัน")
    show_figure("natural_experiment", "การเปรียบเทียบใกล้กันภายใน 3 กม. ในชั่วโมงเดียวกัน")
    st.dataframe(table("natural_experiment"), width="stretch", hide_index=True)
    st.caption("ผลนี้มาจาก 15 ชั่วโมงเท่านั้น จึงควรอ่านช่วงความเชื่อมั่นประกอบ")

    st.subheader("ที่หลบฝนและสถานีรถไฟฟ้า")
    show_figure("denser_share", "สัดส่วนกล้องที่รถหนาแน่นขึ้นเมื่อฝนตก เทียบพื้นที่ใกล้สถานี/ที่หลบฝน/จุดน้ำท่วม")
    st.dataframe(table("denser_share"), width="stretch", hide_index=True)
    st.caption("มีเพียง 5 กล้องในกลุ่มที่หนาแน่นขึ้น จึงยังไม่มีหลักฐานชัดว่าปัจจัยเหล่านี้อธิบายตำแหน่งรถติด")

    st.subheader("น้ำท่วมและข้อร้องเรียน")
    flood = table("flood_sensor_hours")
    st.metric("ชั่วโมงเซนเซอร์ที่น้ำ ≥ 5 ซม.", len(flood))
    st.dataframe(flood, width="stretch", hide_index=True)
    st.caption("เซนเซอร์น้ำท่วมชุดนี้ไม่มีพิกัด จึงจับคู่กับกล้อง CCTV โดยตรงไม่ได้")

with tab_data:
    counts = pd.read_csv(ROOT / "DataAnalysis/data/counts_v2.csv", usecols=["image_id"])
    cols = st.columns(3)
    cols[0].metric("ภาพที่นับรถ", f"{len(counts):,}")
    cols[1].metric("กล้องในผลวิเคราะห์", f"{int(overall.loc['total', 'n_cameras']):,}")
    cols[2].metric("ชั่วโมงกล้องที่ฝนตก", f"{int(overall.loc['total', 'n_wet_hours']):,}")
    st.write("แหล่งตารางผล: `DataAnalysis/output/*.csv`; วิธีคำนวณและข้อค้นพบเต็มอยู่ใน `DataAnalysis/rain/README.md` และ `DataAnalysis/rain_insight.ipynb`")
    st.warning("ข้อมูลครอบคลุมเพียงหนึ่งสัปดาห์ โดยฝนส่วนใหญ่เกิดช่วงสุดสัปดาห์ ผลต่างจากภาพนิ่งไม่พิสูจน์เหตุและผล และไม่สามารถยืนยันจำนวนผู้โดยสารรถไฟฟ้า การปิดเลน ราคา surge หรือจุดจอดมอเตอร์ไซค์หลบฝนตามผังแนวคิดได้")
    st.write("สำหรับ Power BI Desktop สามารถนำเข้า CSV ใน `DataAnalysis/output/` โดยตรง ตาราง `camera_hour.csv` ใช้สร้างกราฟเวลา, `camera.csv` + `camera_effect.csv` ใช้ทำแผนที่ และ `city_effect.csv` ใช้แสดงผลหลัก")
