# understand the wake expansion

import numpy as np
import matplotlib.pyplot as plt

# input parameters
D = 126.0 # rotor diameter
x = np.linspace(0,40*D)
a = 0.3
Ct = 4*a*(1-a)
I0 = 0.06
A = 0.6
c1 = 1.5  # (Page 3)
c2 = 0.8  # (Page 3)
U_inf = 8.0
Vin = 8.0

# park model
k = 0.04
Dw_park = D + 2 * k * x

V_park = U_inf * ( 1 - 2 * a * (D/Dw_park)**2 )

# turbo park model
# Solve for the wake diameter
# (Equation 6 (in steps))
# Computed values
alpha = c1 * I0  # (Page 4)
beta = c2 * I0 / np.sqrt(Ct)
term1 = np.sqrt((alpha + (beta * x / D)) ** 2 + 1)
term2 = np.sqrt(1 + alpha ** 2)
term3 = (term1 + 1) * alpha
term4 = (term2 + 1) * (alpha + (beta * x / D))
Dwx = D + ((A * I0 * D) / beta) * (term1 - term2 - np.log(term3 / term4))

V_turbo = U_inf * (1 - ( 1 - (Vin/U_inf) * np.sqrt( 1 - Ct )) * (D / Dwx)**2)

plt.figure()
plt.plot(x/D,(Dw_park/D)/2,label='Park')
plt.plot(x/D,(Dwx/D)/2,label='Turbo')
plt.legend()
plt.title('D + 2kx vs. Dwx')
plt.grid()

plt.figure()
plt.plot(x/D,(V_park/U_inf)**3,label='Park')
plt.plot(x/D,(V_turbo/U_inf)**3,label='Turbo')
plt.xlabel('x (m)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid()

plt.show()
