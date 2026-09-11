import random
import math

alice = [math.pi * random.random()]
bob = [math.pi * random.random()]
alice = alice + [2 * math.pi * random.random() for _ in range(2)]
bob = bob + [2 * math.pi * random.random() for _ in range(2)]

print(f"alice_params = np.array([{alice}])")
print(f"bob_params = np.array([{bob}])")
# print("Alice:",alice)
# print("bob:",bob)
