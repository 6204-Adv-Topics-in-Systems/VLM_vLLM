import matplotlib.pyplot as plt

# Labels and values
stages = ["encoder", "prefill"]
latencies_ms = [14.2, 202]  # replace with your actual values in ms

# Plot
plt.bar(stages, latencies_ms, color=["skyblue", "lightcoral"])
plt.ylabel("Time (ms)")
plt.title("Latencies of different stages of llava-v1.6-vicuna-7b-hf")

# Save instead of show
plt.savefig("latencies.png", dpi=300, bbox_inches="tight")
plt.close()
