# Data Prep & Training — สรุปส่งงาน

Branch `feat/data-prep-yolo-test-set` · 15 พ.ย. 2026 · ทำโดย kan;ts

## ทำไม

โปรเจกต์ดูว่า **ฝนตกทำให้การเดินทางเปลี่ยนไหม** (ฝนตก → คนเลี่ยง BTS/มอไซค์ → รถยนต์เพิ่ม → ติดหนักที่บางจุด)
ทุกการวิเคราะห์ใช้ **จำนวนรถต่อกล้องต่อเวลา** จาก YOLO ซึ่งนับไม่เป๊ะ งานนี้จึงทำ 3 อย่าง:
**วัด** error ของการนับ, **ลด** error ด้วยการ fine-tune, และ **ให้คนตรวจ** label โดยไม่ต้องวาดกรอบเอง

## ส่งอะไรบ้าง

| ส่วน | ไฟล์ | ทำอะไร |
|---|---|---|
| ชุดภาพ Test | `DataEngineering/test_set/candidates.csv` | 1,000 ภาพ กล้องละ 10 (2 ภาพ × 5 ช่วงเวลา) ล็อก seed ไว้ |
| ชุดภาพ Train | `DataEngineering/train_set/train_manifest.csv` | 1,200 ภาพ กล้องละ 12 ไม่ซ้ำ/ไม่ใกล้ภาพ Test (±20 นาที) val = 31 ส.ค. |
| YOLO ตีกรอบล่วงหน้า | `DataEngineering/reviews/{test,train}/prelabels/` | `yolo11x` ตีกรอบไว้ทั้ง 2,200 ภาพแล้ว (ไม่ต้องมี GPU) |
| เครื่องมือตรวจภาพ | `DataEngineering/labeling/review_app.py` | เว็บ `localhost:8090` กด Approve หรือแก้กรอบ/class เอง |
| Merge Test set | `DataEngineering/labeling/merge_test_set.py` | รวมภาพที่ตรวจแล้ว + ออก `count_error.csv` (YOLO นับพลาดกี่คัน) |
| Training pipeline | `Models/training/train_pipeline.py` | build → fine-tune → เก็บ `best.pt` → eval (mAP + count error) |
| App ใช้โมเดลใหม่ | `Models/model_registry.py`, `Apps/app.py` | App โหลด `Models/weights/best.pt` อัตโนมัติถ้ามี |
| Tests | `tests/` | 34 tests (`python -m unittest discover -s tests -t .`) |

คู่มือ: `DataEngineering/labeling/README.md` (คนตรวจ) · `Models/training/README.md` (train)

## ผลตอนนี้

โมเดล `best.pt` = `yolo11s` fine-tune 80 epoch บน 1,027 ภาพ (label จาก `yolo11x`) วัดบน val 173 ภาพ:

| | mAP50 | นับพลาด / ภาพ (car · moto · bus · truck · **รวม**) |
|---|---|---|
| `yolov8n` เดิมของ App | 0.434 | 1.67 · 0.35 · 0.31 · 0.66 · **1.96** |
| `best.pt` ใหม่ | 0.735 | 1.11 · 0.22 · 0.11 · 0.49 · **1.27** |

นับพลาดลดลง ~35% · ⚠️ val ยังเป็น label จาก `yolo11x` → เป็นแค่ sanity check ตัวเลขจริงต้องรอตรวจ Test set

## วิธีใช้ (ทีม)

```bash
conda activate dsde
git pull
python DataEngineering/labeling/review_app.py            # ตรวจภาพที่ http://localhost:8090
git add DataEngineering/reviews && git commit -m "data(reviews): review batch" && git push

python DataEngineering/labeling/merge_test_set.py         # หลังตรวจ Test ครบ → count error จริง
python Models/training/train_pipeline.py all --device 0    # train ใหม่ด้วย label ที่ตรวจแล้ว
```

กติกา class: กระบะ/รถตู้/แท็กซี่/ตุ๊กตุ๊ก/สองแถว/มินิบัส = **car** · รถเมล์ใหญ่ = **bus** · รถขนของ = **truck**

## ยังไม่ได้ทำ / ต้องรู้

- **ยังไม่มีใครตรวจภาพ** (0/2,200) — ตัวเลข error จริงยังไม่มี
- `best.pt` ยังนับ **กระบะเป็น truck** (ติดมาจาก COCO) → จะหายเมื่อ train ใหม่ด้วย label ที่ตรวจแล้ว
- **bus/truck มีน้อย + truck ปนกระบะ**: pre-label ชุด train มี bus แค่ 3 % ของภาพ (171 ภาพ) แต่ truck 16 % (592 ภาพ) ซึ่งสูงเกินจริง เพราะ yolo11x เรียกกระบะว่า truck
  - review tool มี Sort **"bus/truck first"** → ตรวจภาพที่มี bus/truck ก่อน (กระบะ = car)
  - `build` ซ้ำภาพที่มี bus ใน train 2 เท่า (`--oversample-bus 2`) และเพิ่มภาพยืด/บีบแนวนอน ×0.8–1.2 ภาพละ 1 ใบ (`--stretch`) — val ไม่ถูกแตะ
  - **ยังไม่เพิ่มน้ำหนัก truck** จนกว่าจะตรวจ truck เสร็จ
  - val ยังเป็น pseudo-label ของ yolo11x → วัดว่า bus/truck ดีขึ้นจริงไหมได้หลังตรวจ test set 1,000 ภาพแล้วเท่านั้น
- ไฟล์โมเดล (`*.pt`) ไม่อยู่ใน git → แชร์ `Models/weights/best.pt` + `best.json` ผ่าน Drive หรือ train เอง (~40 นาที GPU)
- **กล้องซ้อนกัน**: 21 คู่ห่างกัน < 100 ม. (ส่วนใหญ่คนละทิศ) → ตอนรวมยอดต้องจัดกลุ่มตาม ถนน + ทิศ ไม่บวกกล้องตรง ๆ (งานถัดไป)
- **ชนกับ branch `feat/interactive-camera-map`**: branch นั้นย้าย `Apps/app.py` → `Apps/Detection-Classification/app.py` ต้องย้ายการแก้ `resolve_model()` ไปไฟล์ใหม่ตอน merge
- ภาพใน dataset บางเครื่องอยู่ที่ `dataset/cctv_images/` แทน `dataset/100_cameras/cctv_images/` → สคริปต์รองรับทั้งสองแบบ
