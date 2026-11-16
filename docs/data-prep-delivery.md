# Data Prep & Training — สรุปส่งงาน

Branch `feat/data-prep-yolo-test-set` · 15 พ.ย. 2026 · ทำโดย kan;ts

## ทำไม

โปรเจกต์ดูว่า **ฝนตกทำให้การเดินทางเปลี่ยนไหม** (ฝนตก → คนเลี่ยง BTS/มอไซค์ → รถยนต์เพิ่ม → ติดหนักที่บางจุด)
ทุกการวิเคราะห์ใช้ **จำนวนรถต่อกล้องต่อเวลา** จาก YOLO ซึ่งนับไม่เป๊ะ งานนี้จึงทำ 3 อย่าง:
**วัด** error ของการนับ, **ลด** error ด้วยการ fine-tune, และ **ให้คนตรวจ** label โดยไม่ต้องวาดกรอบเอง

## ส่งอะไรบ้าง

| ส่วน | ไฟล์ | ทำอะไร |
|---|---|---|
| ชุดภาพ Test | `DataEngineering/test_set/candidates.csv` | 1,000 ภาพ กล้องละ 10 ล็อก seed ไว้ · **ต้องตรวจแค่ 300 ภาพ `core`** (กล้องละ 3 ภาพ คนละช่วงเวลา, ช่วงละ 60) |
| ชุดภาพ Train | `DataEngineering/train_set/train_manifest.csv` | 1,200 ภาพ กล้องละ 12 ไม่ซ้ำ/ไม่ใกล้ภาพ Test (±20 นาที) val = 31 ส.ค. |
| YOLO ตีกรอบล่วงหน้า | `DataEngineering/reviews/{test,train}/prelabels/` | `yolo11x` ตีกรอบไว้ทั้ง 2,200 ภาพแล้ว (ไม่ต้องมี GPU) |
| เครื่องมือตรวจภาพ | `DataEngineering/labeling/review_app.py` | เว็บ `localhost:8090` กด Approve หรือแก้กรอบ/class เอง |
| Merge Test set | `DataEngineering/labeling/merge_test_set.py` | รวมภาพที่ตรวจแล้ว + ออก `count_error.csv` (YOLO นับพลาดกี่คัน) |
| Training pipeline | `Models/training/train_pipeline.py` | build → fine-tune → เก็บ `best.pt` → eval (mAP + count error) |
| App ใช้โมเดลใหม่ | `Models/model_registry.py`, `Apps/app.py` | App โหลด `Models/weights/best.pt` อัตโนมัติถ้ามี |
| Tests | `tests/` | 54 tests (`python -m unittest discover -s tests -t .`) |

คู่มือ: `DataEngineering/labeling/README.md` (คนตรวจ) · `Models/training/README.md` (train)

## ผลตอนนี้ (วัดบน Test 294 ภาพที่คนตรวจ, 16 พ.ย.)

Test = 300 ภาพ core (กล้องละ 3 คนละช่วงเวลา) ตรวจครบ, reject 8 → 294 ภาพ 2,422 คัน (เฉลี่ย 8.2 คัน/ภาพ)

| โมเดล | mAP50 | นับพลาด / ภาพ (car · moto · bus · truck · **รวม**) |
|---|---|---|
| `yolov8n` เดิมของ App | 0.29 | 3.12 · 1.68 · 0.40 · 0.35 · **4.26** |
| `yolo11x` (ตัวตีกรอบล่วงหน้า) | – | 1.70 · 1.32 · 0.13 · 0.84 · **2.28** |
| `yolo11s` fine-tune, pseudo-label ล้วน | 0.44 | 2.01 · 1.32 · 0.14 · 0.94 · **2.52** |
| **`best.pt` ปัจจุบัน** = fine-tune + label คน 252 ภาพ | **0.51** | **1.63 · 1.15 · 0.11 · 0.57 · 2.31** |

- นับพลาดลดจาก 4.26 → 2.31 คัน/ภาพ (−46 %) เทียบโมเดลเดิมของ App
- label จากคนแก้ปัญหากระบะ = truck ได้ตรงจุด (truck 0.94 → 0.57)
- ทุกโมเดลยัง**นับขาด** (best.pt ขาดเฉลี่ย 1.2 คัน/ภาพ) มากสุดช่วงเย็นเร่งด่วน → ต้องระบุในรายงาน
- ขยายภาพตอน predict (416 → 640/832) แทบไม่ช่วย (2.31 / 2.34 / 2.24) → ปัญหาไม่ใช่ความละเอียดภาพเป็นหลัก

### Accuracy (แบบอื่นของตัวเลขเดียวกัน, Test 294 ภาพ)

| วัดแบบ | โมเดลใหม่ | yolov8n เดิม |
|---|---|---|
| นับตรงเป๊ะทั้งภาพ | 20 % ของภาพ | 9 % |
| พลาดไม่เกิน 1 คัน | 51 % ของภาพ | 29 % |
| พลาดไม่เกิน 2 คัน | 68 % ของภาพ | 42 % |
| **ความแม่นยำการนับ** (1 − คันที่พลาดรวม ÷ คันจริงรวม) | **72 %** | 48 % |
| **นับได้กี่ % ของรถจริง** (ยอดรวมทุกภาพ) | **85 %** (2,056 / 2,422) | 49 % |

| ประเภท (โมเดลใหม่) | นับตรงเป๊ะ | พลาด ≤ 1 คัน | นับได้กี่ % ของรถจริง |
|---|---|---|---|
| car | 31 % | 62 % | 87 % (1,570 / 1,796) |
| motorcycle | 47 % | 75 % | **51 %** (270 / 531) |
| bus | 91 % | 99 % | 82 % (49 / 60) |
| truck | 65 % | 87 % | **477 %** (167 / 35) ⚠️ |
| car + truck (รถยนต์รวม) | – | – | 95 % (1,737 / 1,831) |

- ตัวเลขสำหรับรายงาน: **ความแม่นยำการนับ 72 %** และ **นับได้ 85 % ของรถจริง**
- **มอไซค์นับได้แค่ครึ่งเดียว** (เล็ก + ถูกบัง) → ใช้ดูแนวโน้มขึ้น/ลงได้ ไม่ใช้จำนวนตรง ๆ
- **truck ยังนับเกิน ~5 เท่า** (Test มี truck จริงแค่ 35 คัน ส่วนใหญ่ที่เกินคือกระบะ) → **ห้ามใช้ยอด truck ตรง ๆ** ให้รวม car + truck เป็น "รถยนต์" (แม่น 95 %)

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

- **ตรวจแล้ว (16 พ.ย.)**: train 248 ภาพ (แก้ 228 = YOLO ผิด 93 %) · test core เพิ่งเริ่ม — ตัวเลข error จริงรอ test core 300 ภาพ
- **ลด Test เหลือ 300 ภาพ** เพราะเหลือคนตรวจคนเดียว: 300 ภาพพอประมาณค่าเฉลี่ยการนับพลาดได้ (กระจายทุกกล้อง ทุกช่วงเวลา) ภาพที่เหลือ 700 ตรวจเพิ่มได้ถ้ามีเวลา
- `best.pt` ยังนับ **กระบะเป็น truck** (ติดมาจาก COCO) → จะหายเมื่อ train ใหม่ด้วย label ที่ตรวจแล้ว
- **bus/truck มีน้อย + truck ปนกระบะ**: pre-label ชุด train มี bus แค่ 3 % ของภาพ (171 ภาพ) แต่ truck 16 % (592 ภาพ) ซึ่งสูงเกินจริง เพราะ yolo11x เรียกกระบะว่า truck
  - review tool มี Sort **"bus/truck first"** → ตรวจภาพที่มี bus/truck ก่อน (กระบะ = car)
  - `build` ซ้ำภาพที่มี bus ใน train 2 เท่า (`--oversample-bus 2`) และเพิ่มภาพยืด/บีบแนวนอน ×0.8–1.2 ภาพละ 1 ใบ (`--stretch`) — val ไม่ถูกแตะ
  - **ยังไม่เพิ่มน้ำหนัก truck** จนกว่าจะตรวจ truck เสร็จ
  - val ยังเป็น pseudo-label ของ yolo11x → วัดว่า bus/truck ดีขึ้นจริงไหมได้หลังตรวจ test set 1,000 ภาพแล้วเท่านั้น
- **กล้องค้าง**: `DataEngineering/quality/frozen_frames.csv` = 598 ภาพ (1 %) ที่เป็นภาพเดิมซ้ำทุกพิกเซล ทั้งที่เวลาต่างกัน (กล้อง 150, 938, 939, 206, 395 ค้างนานสุด ~23 ชม.) → `Models/pipeline.py` **ข้ามให้อัตโนมัติ** (`--keep-frozen` ถ้าอยากนับด้วย) ถ้าใช้ข้อมูลจากที่อื่นต้องตัดเอง · สร้างใหม่ด้วย `python DataEngineering/quality/find_frozen_frames.py`
- โมเดลของ App (`Models/weights/best.pt` + `best.json`, 19 MB) **อยู่ใน git แล้ว** → `git pull` ได้เลย · โมเดลอื่น/`runs/` ไม่ commit · commit `best.pt` ใหม่เฉพาะเมื่อชนะตัวเดิมบน Test (repo โตครั้งละ 19 MB)
- **กล้องซ้อนกัน**: 21 คู่ห่างกัน < 100 ม. (ส่วนใหญ่คนละทิศ) → ตอนรวมยอดต้องจัดกลุ่มตาม ถนน + ทิศ ไม่บวกกล้องตรง ๆ (งานถัดไป)
- **ชนกับ branch `feat/interactive-camera-map`**: branch นั้นย้าย `Apps/app.py` → `Apps/Detection-Classification/app.py` ต้องย้ายการแก้ `resolve_model()` ไปไฟล์ใหม่ตอน merge
- ภาพใน dataset บางเครื่องอยู่ที่ `dataset/cctv_images/` แทน `dataset/100_cameras/cctv_images/` → สคริปต์รองรับทั้งสองแบบ
