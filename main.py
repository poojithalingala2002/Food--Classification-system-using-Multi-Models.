import redis
import json
import os

# connect to redis
r = redis.Redis(host="localhost", port=6379)

# folders
json_folder = "json results"
model_folder = "model results"   # folder where .h5 files are stored

# ---------------- STORE JSON FILES ----------------
for file in os.listdir(json_folder):

    if file.endswith(".json"):

        path = os.path.join(json_folder, file)

        with open(path, "r") as f:
            data = json.load(f)

        name = file.replace(".json", "")

        r.set(name, json.dumps(data))

        print(name, "JSON stored")


# ---------------- STORE H5 MODEL FILES ----------------
for file in os.listdir(model_folder):

    if file.endswith(".h5"):

        path = os.path.join(model_folder, file)

        with open(path, "rb") as f:   # read binary
            model_data = f.read()

        name = file.replace(".h5", "")

        r.set(name, model_data)

        print(name, "Model stored")
