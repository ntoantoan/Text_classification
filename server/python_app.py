import requests as r

# add review
text = "TAND quận Gò Vấp, TP.HCM, đã thụ lý vụ án tranh chấp hợp đồng vay tiền giữa nguyên.Quận Gò Vấp (TP.HCM) đã thụ lý vụ án tranh chấp hợp đồng vay tiền giữa nguyên đơn là bà Đặng Thùy Trang và bị đơn là hoa hậu Nguyễn Thúc Thùy Tiên."


keys = {"text": text}

prediction = r.get("http://0.0.0.0:4500/predict-text/", params=keys)

results = prediction.json()
print(results["prediction"])
print(results["Probability"])