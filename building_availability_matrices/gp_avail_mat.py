import matplotlib.pyplot as plt
import pickle
from building_availability_matrices.gaussian_process_models.utils import exp1


res = exp1(freq0=0.8)

with open("gp_avail_mat.pkl", "wb") as f:
    pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)

fig, axs = plt.subplots(2, 1, figsize=(6, 12))

# Plot each array using imshow
for i, ty in enumerate(res.keys()):
    axs[i].imshow(res[ty], cmap="binary")
    axs[i].set_title(ty)

# Adjust layout
plt.tight_layout()
plt.savefig("gp_avail_mat.png")