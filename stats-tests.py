from scipy.stats import binom
import statsmodels.stats.proportion as smp

# Count wins
total = 1000
wins = 517
losses = total - wins

# Binomial test: Is the win rate > 50%?
p_value = binom.sf(wins-1, n=total, p=0.5)

# Confidence interval (Wilson score)
ci_low, ci_upp = smp.proportion_confint(wins, total, alpha=0.05, method='wilson')

# Print results
print(f"Win count: {wins} / {total}")
print(f"Loss count: {losses}")
print(f"Binomial test p-value: {p_value:.5f}")
print(f"95% CI (Wilson score): [{ci_low:.2%}, {ci_upp:.2%}]")