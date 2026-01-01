NEURON {
  SUFFIX hhplustoggle
  USEION na WRITE ina
  USEION  k WRITE ik
  USEION cl READ ecl WRITE icl
  NONSPECIFIC_CURRENT ipump
  RANGE ena, ek, kbath, gcl, kbath0, kbath1, t_toggle, gk, gkleak, gna, gnaleak, rho
}

STATE { n dki dkg nai nao ki ko t }

CONSTANT {
  : Nernst scaling factor = R T / q F at T = 37 Celsius and q = 1
  C = 26.64 (mV)
  : Area factor, computed in main script.
  A = 808.078016193 (um2)
  : corresponding volume
  V = 2160.0 (um3)
  : Faraday
  F = 9.648533e4 (A s / mol)
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

ASSIGNED {
  ena (mV)
  ek (mV)
  kbath
}

PARAMETER {
  : declare membrane potential
  v (mV)
  : Relaxation time constant of n
  tau_n    =      0.25  (ms)
  : Conductance constants
  gcl      =      7.5   (nS)
  gkleak   =      0.12  (nS)
  gk       =      22.0  (nS)
  gnaleak  =      0.02  (nS)
  gna      =      40.0  (nS)
  : Ionic concentration constants
  ki0      =  140.0      (mM)
  ko0      =    4.8      (mM)
  kbath0   =    9.5      (mM)
  kbath1   =    9.5      (mM)
  nai0     =   16.0      (mM)
  nao0     =  138.0      (mM)
  : Volume ratio Vi/Vo
  beta     =    3.0
  rho      =    250.0 (pA)
  : rate of diffusion
  epsilon  =    0.01   (/ms)
  : convert from current to flux
  gamma = 0.04 (mmol um3 / C)
  : time at which we toggle kbath to kbath1
  t_toggle = 2000 (ms)
}

INITIAL {
  n = n_inf(v)
  dki = -0.6
  dkg =  0.8
  kbath = kbath0
  t = 0

  ena = 0
  ek = 0
}

BREAKPOINT {
  SOLVE dS METHOD cnexp

  LOCAL gnabar, gkbar, gclbar, gpump, g_to_S_cm2

  if (t > t_toggle) {
       kbath = kbath1
  }

  : Concentrations
  ki  = ki0  + dki
  ko  = ko0  - beta*dki + dkg
  nai = nai0 - dki
  nao = nao0 + beta*dki

  : Reversal potentials
  ena = C*log(nao/nai)
  ek  = C*log(ko/ki)

  : Convert nS to S/cm2 since we want i in mA/cm2 and i = g U
  : [U] = mV
  : [A] = um2
  : [g*] = nS
  : [g / A] = nS / um2 = 1e-9 S / 1e-8 cm2 = 0.1 S/cm2
  g_to_S_cm2 = 0.1 / A

  : conductivties
  gnabar = g_to_S_cm2*(gnaleak + gna*m_inf(v)*h(n))
  gkbar  = g_to_S_cm2*(gkleak  + gk*n)
  gclbar = g_to_S_cm2*gcl

  : pumping current into mA / cm2
  : [rho] = pA
  : [A] = um2
  : [rho/A] = pA / um2 = 1e-9 mA / 1e-8 cm2 = 0.1 mA/cm2
  gpump  = rho * g_to_S_cm2

  : currents
  ina   = gnabar*(v - ena)
  ik    = gkbar*(v - ek)
  icl   = gclbar*(v - ecl)
  ipump = gpump/((1.0 + exp(10.5 - 0.5*nai))*(1.0 + exp(5.5 - ko)))
}

DERIVATIVE dS {
  LOCAL lki, lko, lnai, lik, lipump

  lki  = ki0  + dki
  lko  = ko0  - beta*dki + dkg
  lnai = nai0 - dki

  lik    = (gkleak  + gk*n)*(v - C*log(lko/lki))
  lipump = rho/((1.0 + exp(10.5 - 0.5*lnai))*(1.0 + exp(5.5 - lko)))

  : derivative units are [X'] = [X]/ms
  n'   = (n_inf(v) - n) / tau_n
  dki' = gamma * (2*lipump - lik) / V
  dkg' = epsilon * (kbath - lko)

  : for keeping time
  t' = 1 : ms
}

FUNCTION n_inf(u) { n_inf = 1.0 / (1.0 + exp(-(u + 19.0) / 18.0)) }
FUNCTION m_inf(u) { m_inf = 1.0 / (1.0 + exp(-(24.0 + u) / 12.0)) }
FUNCTION h(x)     { h = 1.1 - 1.0 / (1.0 + exp(3.2 - 8.0 * x)) }
