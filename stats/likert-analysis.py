import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Sample Data
# categories = [
#     "I find holding a mouse uncomfortable",
#     "I find the input options of a mouse restrictive",
#     "I'm concerned about injuring myself",
#     "I find typing on a keyboard uncomfortable",
#     "I find the input options of a keyboard restrictive",
#     "I want to type words quicker than my keyboard allows",
#     "I am concerned about injuring myself",
#     "I would be excited to use it",
#     "I would be nervous about using it"
# ]
categories = [
    "Q1","Q2",'Q3','Q4','Q5','Q6','Q7','Q8','Q9'
]

means = np.array([-0.333333333, 0.166666667, -0.333333333, 0.333333333, -0.166666667, 0.166666667, -0.5, 0.333333333, -0.833333333])
mins = np.array([-0.645138116, -0.258251626, -0.645138116, 0.021528551, -0.402368927, -0.069035594, -0.704124145, 0.215482203, -0.951184464])
maxs = np.array([-0.021528551, 0.591584959, -0.021528551, 0.645138116, 0.069035594, 0.402368927, -0.295875855, 0.451184464, -0.715482203])

widths = maxs - mins

# Assign colors based on your mapping
colors = []
for i in range(len(categories)):
    num = i + 1
    if num in [7, 4, 3, 1]:
        colors.append('#0f52ba')   # Expressiveness
    elif num in [6, 5, 2]:
        colors.append('#0a5f38')  # Comfort
    else:
        colors.append('#901c2d') # Adaptability 

fig, ax = plt.subplots(figsize=(7, 4))

# 1. Create the Horizontal Range Bars
ax.barh(categories, widths, height=0.3, left=mins, color=colors, alpha=1)

# 2. Add the Mean Lines
for i, mean in enumerate(means):
    ax.vlines(x=mean, ymin=i - 0.25, ymax=i + 0.25, color='black', linewidth=3)

# 3. Create Custom Legend Handles
red_patch = mpatches.Patch(color='#901c2d', label='Adaptability')
green_patch = mpatches.Patch(color='#0a5f38', label='Expressiveness')
blue_patch = mpatches.Patch(color='#0f52ba', label='Comfort')
mean_line = plt.Line2D([0], [0], color='black', linewidth=3, label='Mean')

ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_xticklabels(["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"])
ax.set_xlim([-1, 1])

# Formatting
ax.set_xlabel("Average Likert response across all participants", fontweight='bold')
ax.set_ylabel("Survey Question", fontweight='bold')
ax.set_title("Survey on user experience of current technologies", fontweight='bold')

ax.grid(axis='x', linestyle='--', alpha=0.6)
# ax.grid(axis='y', linestyle='--', alpha=0.6)

# Apply the custom legend
ax.legend(handles=[red_patch, green_patch, blue_patch, mean_line], 
          loc='upper center', bbox_to_anchor=(0.5, -0.15), 
          fancybox=True, ncol=4)

plt.tight_layout()
plt.show()
