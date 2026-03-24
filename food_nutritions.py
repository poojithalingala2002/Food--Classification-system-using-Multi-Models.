import redis
import json

# connect to redis
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

# load food nutrients json
with open("food_nutrition.json", "r") as f:
    data = json.load(f)

# store each food item in redis
for food, nutrients in data.items():
    key = f"food:{food}"
    r.set(key, json.dumps(nutrients))
    print(f"{key} stored in Redis")

print("\nAll food nutrients uploaded successfully!")