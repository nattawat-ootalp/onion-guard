"""What to do with the head after a scan.

WHY THIS IS NOT A SAFETY VERDICT
The model misses roughly 15 of every 100 infected heads (recall 0.846 on
held-out folds). A screen with that miss rate cannot certify anything as safe
to eat, and printing "ปลอดภัย" would be worse than printing nothing: it would
replace the operator's own inspection with a guarantee the system cannot make.

So every message below states an ACTION, never a status. The negative case
says "still inspect it and wash it", not "it is clean". The wording rules the
report fixes for the result label apply here too — this text sits directly
under it and must not undo it.

WHY WASHING IS NOT OFFERED AS A FIX FOR A POSITIVE
Washing removes surface spores. It does not remove mycotoxins, which are
chemically stable and survive normal processing (WHO / FDA guidance on
mycotoxins). Telling someone to wash a head the system flagged and then use
it would be actively wrong advice, so the positive case says to set it aside,
not to clean it up.

PRIORITY
Reliability beats result. A scan whose framing failed or whose features sit
outside the training range has no trustworthy label at all, so that advice is
returned instead of — not alongside — the positive/negative advice. Otherwise
the page would show a confident instruction derived from a number it just
said was meaningless.
"""

# Keys are stable identifiers; the UI styles by `level`, not by matching text.
LEVELS = ("unreliable", "positive", "caution", "negative")

DEFAULT_ADVICE = {
    "unreliable": {
        "level": "unreliable",
        "headline": "ยังสรุปไม่ได้ ควรถ่ายใหม่",
        "steps": [
            "ถ่ายภาพใหม่ให้หัวหอมอยู่กลางกรอบ เห็นเต็มหัว ไม่ชนขอบภาพ",
            "ตรวจว่าปิดไฟห้องแล้ว เปิดเฉพาะไฟ UV และล็อกค่ากล้อง (ISO/shutter/WB) ไว้เหมือนเดิมทุกครั้ง",
            "ถ้าถ่ายใหม่แล้วยังขึ้นแบบเดิม ให้ถือว่ายังไม่ได้ตรวจ และตรวจด้วยตาหรือส่งห้องปฏิบัติการแทน",
        ],
        "note": "อย่าใช้ผลนี้ตัดสินใจ ระบบกำลังบอกว่าภาพนี้อยู่นอกเงื่อนไขที่มันวัดได้",
    },
    "positive": {
        "level": "positive",
        "headline": "แยกหัวนี้ออก และส่งตรวจยืนยัน",
        "steps": [
            "แยกออกจากกองทันที อย่าเก็บรวมกับหัวอื่น เพราะสปอร์แพร่ไปหัวข้างเคียงได้",
            "อย่านำไปบริโภคหรือจำหน่ายจนกว่าจะมีผลตรวจยืนยัน",
            "ส่งเพาะเชื้อ (เช่น CompactDry YM) เพื่อยืนยันว่าพบเชื้อจริงหรือไม่",
            "ตรวจหัวที่เก็บอยู่ใกล้กันเพิ่มเติม และลดความชื้นในที่เก็บ",
        ],
        "note": "การล้างไม่ช่วยในกรณีนี้ — ล้างออกได้เฉพาะสปอร์ที่ผิว แต่สารพิษจากเชื้อรา "
                "(mycotoxin) ทนความร้อนและการล้าง ไม่หายไปด้วยการทำความสะอาด",
    },
    "caution": {
        "level": "caution",
        "headline": "ผลก้ำกึ่ง ให้ถือว่าน่าสงสัยไว้ก่อน",
        "steps": [
            "แยกเก็บต่างหากก่อน จนกว่าจะตรวจซ้ำ",
            "ถ่ายซ้ำอีกมุมหนึ่ง เผื่อรอยอยู่ด้านที่กล้องไม่เห็น",
            "ถ้ายังก้ำกึ่ง ให้ส่งเพาะเชื้อยืนยัน",
        ],
        "note": "ค่าที่ได้อยู่ใกล้เส้นตัดสินมาก การขยับเพียงเล็กน้อยก็เปลี่ยนผลได้ "
                "จึงยังไม่ควรใช้ผลนี้ตัดสินใจอย่างเดียว",
    },
    "negative": {
        "level": "negative",
        "headline": "ยังไม่พบสิ่งผิดปกติ แต่ต้องตรวจด้วยตาก่อนใช้",
        "steps": [
            "ตรวจด้วยตาอีกครั้ง โดยเฉพาะบริเวณราก คอหัว และใต้เปลือกชั้นนอก",
            "ล้างและปอกเปลือกชั้นนอกก่อนนำไปใช้",
            "เก็บในที่แห้ง อากาศถ่ายเท ไม่อับชื้น",
            "ถ้าพบกลิ่นผิดปกติ จุดนิ่ม หรือผงสีคล้ำ ให้แยกออกแม้ระบบจะไม่พบก็ตาม",
        ],
        "note": "ระบบนี้ตรวจไม่พบหัวที่มีเชื้อประมาณ 15 ใน 100 หัว ผลนี้จึงไม่ใช่การรับรอง "
                "ว่าปลอดภัย และไม่ควรใช้แทนการตรวจด้วยตา",
    },
}


def _is_unreliable(ood_status=None, framing_ok=True, scale_status=None,
                   background_status=None):
    """Any signal that the label itself cannot be trusted.

    out_of_range means the measurements fall outside everything the model was
    fitted on, where a tree ensemble stops responding rather than
    extrapolating; a failed detection means the features describe a centred
    crop of the background rather than the onion. Either makes the label
    meaningless, not merely uncertain.
    """
    if ood_status == "out_of_range":
        return "ค่าที่วัดได้อยู่นอกช่วงข้อมูลที่ใช้ฝึกโมเดล"
    if framing_ok is False:
        return "ระบบหาตำแหน่งหัวหอมในภาพไม่สำเร็จ"
    if scale_status == "inconsistent":
        return "ขนาดหัวหอมระหว่างภาพไม่สอดคล้องกัน"
    if background_status == "inconsistent":
        return "ความสว่างพื้นหลังระหว่างภาพต่างกันมาก"
    return None


def build_advice(label, borderline=False, ood_status=None, framing_ok=True,
                 scale_status=None, background_status=None, advice_table=None):
    """Return one advice dict for a finished scan.

    label: 1 = พบความผิดปกติที่สัมพันธ์กับเชื้อรา, 0 = ไม่พบความผิดปกติ.
    advice_table lets config.json override the wording without touching code.
    """
    table = advice_table or DEFAULT_ADVICE

    reason = _is_unreliable(ood_status, framing_ok, scale_status, background_status)
    if reason is not None:
        out = dict(table["unreliable"])
        out["reason"] = reason
        return out

    if int(label) == 1:
        return dict(table["positive"])
    if borderline:
        # A negative this close to the cutoff is treated as suspicious rather
        # than clean: a false negative costs more here than a wasted re-check.
        return dict(table["caution"])
    return dict(table["negative"])
