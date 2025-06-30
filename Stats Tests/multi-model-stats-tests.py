import statsmodels.stats.proportion as smp

# Example
wins_A = 520
total_A = 1000

wins_B = 501
total_B = 1000

# Proportion (win rate)
prop_A = wins_A / total_A
prop_B = wins_B / total_B

# Confidence intervals
ci_A = smp.proportion_confint(wins_A, total_A, alpha=0.05, method='wilson')
ci_B = smp.proportion_confint(wins_B, total_B, alpha=0.05, method='wilson')

print(f"Model A: {prop_A:.2%} (95% CI: {ci_A[0]:.2%} to {ci_A[1]:.2%})")
print(f"Model B: {prop_B:.2%} (95% CI: {ci_B[0]:.2%} to {ci_B[1]:.2%})")