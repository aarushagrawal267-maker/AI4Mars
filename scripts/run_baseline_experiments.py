"""Reproduce the baseline U-Net vs DeepLabV3 comparison through the REST API (backend must be running)."""
import json, sys, time, urllib.request
B = "http://localhost:8000/api"
def call(method, path, body=None):
    req = urllib.request.Request(B + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())
COMMON = dict(epochs=10, batch_size=8, resolution=192, learning_rate=1e-3, weight_decay=1e-4, dice_weight=0.5,
              class_weighting="none", scheduler="cosine", early_stopping_patience=4, augment=True, seed=42,
              max_train_samples=None, max_val_samples=None)
RUNS = [dict(arch="unet", model_options={"base_channels": 16, "depth": 4}, name="U-Net (base 16) - MSL NavCam subset, 192 px"),
        dict(arch="deeplabv3", model_options={"backbone": "mobilenet_v3_large", "pretrained": "coco"},
             name="DeepLabV3 (MobileNetV3-L, COCO init) - MSL NavCam subset, 192 px")]
for run in RUNS:
    r = call("POST", "/training/start", {**COMMON, **run})
    eid = r["experiment_id"]; print("started", eid, flush=True)
    while True:
        time.sleep(20)
        st = call("GET", "/training/status")
        if not st["active"]:
            print("finished", eid, st["state"], st.get("best_val_miou"), flush=True); break
    if st["state"] not in ("completed",):
        print("run did not complete; stopping pipeline", flush=True); sys.exit(1)
    for split, variant in (("test", "min3"), ("test", "min1"), ("val", None)):
        call("POST", "/evaluate", {"model_id": eid, "split": split, "label_variant": variant})
        while True:
            time.sleep(5)
            j = call("GET", "/evaluate/status")["job"]
            if j["status"] != "running" and j["status"] != "queued":
                print("  eval", split, variant, j["status"], j.get("error"), flush=True); break
print("PIPELINE DONE", flush=True)
