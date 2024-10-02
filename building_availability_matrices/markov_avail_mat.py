from markov_models.utils import gen_avail_mat
import matplotlib.pyplot as plt

CLIENTS = {
    'C0' : (0.4, 0.6),
    'C1' : (0.4, 0.6),
    'C2' : (0.4, 0.6),
    'C3' : (0.4, 0.6),
    'C4' : (0.4, 0.6),
    'C5' : (0.4, 0.6),
    'C6' : (0.4, 0.6)
}

# CLIENTS_CORR = {
#     'C0' : (0.04, 0.06),
#     'C1' : (0.04, 0.06),
#     'C2' : (0.04, 0.06),
#     'C3' : (0.04, 0.06),
#     'C4' : (0.04, 0.06),
#     'C5' : (0.04, 0.06),
#     'C6' : (0.04, 0.06)
# }

CLIENTS_CORR = {
    'C0' : (0.94, 0.96),
    'C1' : (0.94, 0.96),
    'C2' : (0.94, 0.96),
    'C3' : (0.94, 0.96),
    'C4' : (0.94, 0.96),
    'C5' : (0.94, 0.96),
    'C6' : (0.94, 0.96)
}


avail_mat = gen_avail_mat(CLIENTS_CORR, 100)

plt.imshow(avail_mat)
plt.savefig('tmp_eig_1.png')

