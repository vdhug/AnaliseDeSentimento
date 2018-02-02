import numpy as np
import matplotlib.pyplot as plt


vira_latas = 500
labradores = 500

vira_lata_altura = 28 + 4 * np.random.randn(vira_latas)
labradores_altura = 24 + 4 * np.random.randn(labradores)

plt.hist([vira_lata_altura, labradores_altura], stacked=True, color=['r', 'b'])
plt.show()