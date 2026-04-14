import matplotlib.pyplot as plt
import numpy as np

categories = [
'Focus',
'Difficulty',
'Confidence',
'Connection',
'Adaptability',
'Enjoyment'
]

# --- Series 1 ---
means1 = np.array([2.571428571	,3.875,	2.875,	1.5	,1.25	,1.875])
mins1  = np.array([2.207212892	,3.709640543	,2.411487595	,1.146446609	,0.835421901	,1.348365639])
maxs1  = np.array([2.935644251	,4 ,3.338512405	,1.853553391	,1.664578099	,2.401634361])

# --- Series 2 ---
means2 = np.array([3.625	,2.625	,3.25	,2.875	,3.75	,3.375])
mins2  = np.array([3.382938541	,2.382938541	,2.919281086	,2.484687625	,3.533493649	,3.132938541])
maxs2  = np.array([3.867061459	,2.867061459	,3.580718914	,3.265312375	,3.966506351	,3.617061459])

# Error bars
yerr1 = np.vstack([means1 - mins1, maxs1 - means1])
yerr2 = np.vstack([means2 - mins2, maxs2 - means2])

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))

# --- Series 1 (RED) ---
ax.bar(x - width/2, means1, width,
       color='#d62728',
       label='Mouse (Control)',
       yerr=yerr1,
       capsize=4)

# --- Series 2 (BLUE) ---
ax.bar(x + width/2, means2, width,
       color='#1f77b4',
       label='HoldMyHand',
       yerr=yerr2,
       capsize=4)

# X axis
ax.set_xticks(x)
ax.set_xticklabels(categories)

# Y axis (Likert scale)
ax.set_ylim(0, 4.2)
ax.set_yticks([0, 1, 2, 3, 4])
ax.set_yticklabels([
    "Strongly Disagree",
    "Disagree",
    "Neutral",
    "Agree",
    "Strongly Agree"
])

# Styling
ax.set_ylabel("Likert response")
ax.set_xlabel("Survey Question")
ax.set_title("Survey comparison on 5-point Likert scale")

ax.axhline(2, color='black', linewidth=0.8, alpha=0.5)  # neutral line
ax.grid(axis='y', linestyle='--', alpha=0.3)

ax.legend()

plt.tight_layout()
plt.show()
