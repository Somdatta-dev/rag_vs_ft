import pandas as pd

# Load baseline evaluation results
base_df = pd.read_csv('results/baseline_eval_base_20250809_213747.csv')
ft_df = pd.read_csv('results/baseline_eval_ft_20250809_213822.csv')

print("=" * 60)
print("BASELINE EVALUATION COMPARISON")
print("=" * 60)

print("\n📊 BASE MODEL (Pre-Fine-Tuning) RESULTS:")
base_accuracy = (base_df["correct"].sum() / len(base_df))
base_time = base_df["time_s"].mean()
base_similarity = base_df["similarity"].mean()
base_confidence = base_df["confidence"].mean()

print(f"Accuracy: {base_accuracy:.1%}")
print(f"Avg Time: {base_time:.3f}s")
print(f"Avg Similarity: {base_similarity:.3f}")
print(f"Avg Confidence: {base_confidence:.3f}")
print(f"Questions Evaluated: {len(base_df)}")

print("\n🎯 FINE-TUNED MODEL RESULTS:")
ft_accuracy = (ft_df["correct"].sum() / len(ft_df))
ft_time = ft_df["time_s"].mean()
ft_similarity = ft_df["similarity"].mean()
ft_confidence = ft_df["confidence"].mean()

print(f"Accuracy: {ft_accuracy:.1%}")
print(f"Avg Time: {ft_time:.3f}s")
print(f"Avg Similarity: {ft_similarity:.3f}")
print(f"Avg Confidence: {ft_confidence:.3f}")
print(f"Questions Evaluated: {len(ft_df)}")

print("\n📈 COMPARISON (Fine-tuned vs Base):")
acc_diff = ft_accuracy - base_accuracy
time_diff = ft_time - base_time
sim_diff = ft_similarity - base_similarity
conf_diff = ft_confidence - base_confidence

print(f"Accuracy Change: {acc_diff:+.1%} ({'✅ Improved' if acc_diff > 0 else '❌ Degraded'})")
print(f"Speed Change: {time_diff:+.3f}s ({'❌ Slower' if time_diff > 0 else '✅ Faster'})")
print(f"Similarity Change: {sim_diff:+.3f} ({'✅ Improved' if sim_diff > 0 else '❌ Degraded'})")
print(f"Confidence Change: {conf_diff:+.3f} ({'✅ Improved' if conf_diff > 0 else '❌ Degraded'})")

print("\n" + "=" * 60)
