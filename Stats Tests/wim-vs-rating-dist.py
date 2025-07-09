import re
import numpy as np
import matplotlib.pyplot as plt

# Extract the responses
with open("Completions/Ratings/rating-vs-response.txt", "r", encoding="utf-8") as f:
    raw = f.read()
raw_responses = re.findall(r"^Response:\s*(.*)", raw, flags=re.MULTILINE)
raw_ratings = re.findall(r"^Rating:\s*(.*)", raw, flags=re.MULTILINE)

# Cast to floats
ratings = [float(idx) for idx in raw_ratings]
responses = [float(idx) for idx in raw_responses]

# Change the scale to 1->10
ratings = [((idx * 5) + 5) for idx in ratings]
responses = [((idx * 5) + 5) for idx in responses]

# Grouping the ratings to a single output
response_group = []
rating_group = []
for i in range(0, len(responses)-1, 2):
    response_group.append((responses[i], responses[i+1]))
for i in range(0, len(ratings)-1, 2):
    rating_group.append((ratings[i], ratings[i+1]))

print(f'Rating Mean: {np.mean(ratings)}')
print(f'Rating Std: {np.std(ratings)}')
print(f'Rating Var: {np.var(ratings)}')
print(f'Responses Mean: {np.mean(responses)}')
print(f'Responses Std: {np.std(responses)}')
print(f'Responses Var: {np.var(responses)}')
print()

# Finding repeated pairings for the ratings
same_rating_count = 0
different_rating_groups = []
for i, rating_couple in enumerate(rating_group):
    if rating_couple[0] == rating_couple[1]:
        same_rating_count += 1
    else:
        different_rating_groups.append((i, rating_couple))
print(f'Percentage the same rating: {(float(same_rating_count) / float(len(rating_group))) * 100}%')

# Seeing how often the ratings agree
agreement_count = 0
for rating_couple in different_rating_groups:
    response_at_idx = response_group[rating_couple[0]]
    rating_at_idx = rating_couple[1]

    first_response_higher = True if response_at_idx[0] >= response_at_idx[1] else False
    first_rating_higher = True if rating_at_idx[0] >= rating_at_idx[1] else False

    if first_rating_higher == first_response_higher:
        agreement_count += 1
print(f'Agreement percentage: {(float(agreement_count) / float(len(different_rating_groups))) * 100}%')

# Seeing the distribution in rating pairs
response_pair_delta = []
rating_pair_delta = []
for i in range(len(rating_group)):
    response_at_idx = response_group[i]
    rating_at_idx = rating_group[i]

    # Deltas
    response_pair_delta.append(abs(response_at_idx[0] - response_at_idx[1]))
    rating_pair_delta.append(abs(rating_at_idx[0] - rating_at_idx[1]))

# Find averages
print(f'Average response delta: {np.average(response_pair_delta)}')
print(f'Average rating delta: {np.average(rating_pair_delta)}')
print()

print(f'Agreement percentage: {(float(agreement_count) / float(len(different_rating_groups))) * 100}%')

show_graphs = True
if show_graphs:
    xmin = 0
    xmax = 10
    plt.figure(1)
    plt.hist(ratings)
    plt.ylabel('Count')
    plt.xlabel('Rating')
    plt.xlim(xmin, xmax)
    plt.hist(responses)
    plt.title('Histogram of Adjusted WIM Ratings')
    plt.show()