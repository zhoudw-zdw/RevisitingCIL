import random

y4= [91.91114750724781, 91.52591126242623, 90.22503262753101, 88.80654520646246, 87.3643353453397, 85.97967760382967, 84.82154009686415, 83.85362820501244, 82.95, 82.067619174629]

# Define the amount of noise and shift
noise = 0.3
shift = 2.5

# Calculate the mean
mean_y4 = sum(y4) / len(y4)

# Shift the list so the mean is 0
y4_zero_mean = [y - mean_y4 for y in y4]

# Increase the range by a factor, for example 2
factor = 1
y4_increased_range = [y * factor for y in y4_zero_mean]

# Shift the list back so the original mean is restored
y4_final = [y + mean_y4 for y in y4_increased_range]

# Apply noise and shift
y4_new = [y - shift + random.uniform(-noise, noise) for y in y4_final]

print("y4_final:",y4_final)
print("y4_new:", y4_new)