"""
2KathleenBlyth(2019)forMicropropulsion
3Themainfilefordeterminingthebestoperationalenvelopebycomparinga
4rangeofinitialpoints
5"""
import time

import matplotlib.pyplot as plt
import numpy as np
from Axessetup import AxesConfig
from Behaviour import ThrusterBehaviour

# =============================================================================
# ----------------------------- UserInputs  -----------------------------------
# =============================================================================

efficiency = 0.6  # Thrusterefficiency[-]
initialpressure = [1.1, 1.22, 1.25, 0.365]  # Initialpressure[bar]
initialvolume = [0.12, 0.115, 0.165, 0.385]  # Initialvolume[V/Vtube]
dt = 0.1  # Timestep[s]

# =============================================================================
# -------------------  Determining operational envelope -----------------------
# =============================================================================

axissetup = AxesConfig()
behaviour = ThrusterBehaviour()
starttime = time.time()

initialpressure = [x * 1E5 for x in initialpressure]
envelopes = {}

for i in range(4):
    key = "Point" + str(i + 1)
    envelopes[key] = behaviour.envelope(initialvolume[i], initialpressure[i], efficiency, dt, 2)

# =============================================================================
# -------------------------- Generat ing p l o t s --------------------------------
# =============================================================================

# --> De f ining the 4 axes
fig, axs = plt.subplots(3, 2, figsize=(10, 12))
ax1, ax2, ax3, ax4, ax5, ax6 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1], axs[2, 0], axs[2, 1]
# --> Pl o t t ing the data f o r a l l s c e n a r i o s
for i in range(4):
    key = "Point " + str(i + 1)
    burn_time = np.arange(0, envelopes[key]["burn_time"], dt)
    ax1.plot(burn_time, [x * 1000 for x in envelopes[key]["thrust"]])
    ax2.plot(burn_time, envelopes[key]["power"])
    ax3.plot(burn_time, [x * 1000 for x in envelopes[key][" thrus t t o powe r "]])
    ax4.plot(burn_time, [x * 1000 for x in envelopes[key]["mass"]])
    ax5.plot(burn_time, [x / 100000 for x in envelopes[key][" pr e s sur e "]])
    ax6.plot(burn_time, envelopes[key][" temperature "])
# --> Formatting the pl o t a x i s
labels = ['Previous point', 'Max burn time', 'Max average thrust',
          'Max thrus t to power ']
axissetup.axesconfig(axs, ax1, labels, "upper right", "Time [ s ]", "Thrust [mN]", "Thrust ")
axissetup.axesconfig(
    axs, ax2, labels, "upper right ", "Time [ s ] ", "Power [W] ", " Input Power")
axissetup.axesconfig(axs, ax3, labels, " lower right ", "Time [ s ] ",
                     "Thrust to power [mN/W] ", "Thrust to power")
axissetup.axesconfig(axs, ax4, labels, "upper right ", "Time [ s ] ", "Mass [ g ] ",
                     " Pr ope l l ant remaining ")
axissetup.axesconfig(axs, ax5, labels, "upper right ", "Time [ s ] ",
                     " Pressure [bar]", "Chamber Pressure")
axissetup.axesconfig(axs, ax6, labels, "upper right ", "Time [ s ] ",
                     "Temperature [K] ", "Chamber temperature")
plt.show()
