import numpy as np
from numba import guvectorize, float64

import logging
import tvb.simulator.lab as lab
from tvb.contrib.cosimulation.cosim_monitors import RawCosim as CoMonitor
from tvb.contrib.cosimulation.cosimulator import CoSimulator
from tvb.simulator.models.wong_wang import ReducedWongWang, Final, List
from tvb.basic.logger.builder import set_loggers_level

set_loggers_level(logging.ERROR)

# NOTE wat? lab is not a module
Linear = lab.coupling.Linear
Connectivity = lab.connectivity.Connectivity
Euler = lab.integrators.EulerDeterministic
Monitor = lab.monitors.Raw


@guvectorize([(float64[:],) * 12], "(n),(m)" + ",()" * 8 + "->(n),(n)", nopython=True)
def _numba_dfun(S, c, a, b, d, g, ts, w, j, io, dx, h):
    """Gufunc for reduced Wong-Wang model equations.(modification for saving the firing rate h)"""
    x = w[0] * j[0] * S[0] + io[0] + j[0] * c[0]
    h[0] = (a[0] * x - b[0]) / (1 - np.exp(-d[0] * (a[0] * x - b[0])))
    dx[0] = -(S[0] / ts[0]) + (1.0 - S[0]) * h[0] * g[0]


@guvectorize([(float64[:],) * 5], "(n),(m)" + ",()" * 2 + "->(n)", nopython=True)
def _numba_dfun_proxy(s, h, g, ts, dx):
    """Gufunc for reduced Wong-Wang model equations for proxy node."""
    dx[0] = -(s[0] / ts[0]) + (1.0 - s[0]) * h[0] * g[0]


class ReducedWongWangProxy(ReducedWongWang):
    state_variables = "S H".split()
    _nvar = 2
    state_variable_range = Final(
        label="State variable ranges [lo, hi]",
        default={"S": np.array([0.0, 1.0]), "H": np.array([0.0, 0.0])},
        doc="Population firing rate",
    )
    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("S", "H"),
        default=("S", "H"),
        doc="""default state variables to be monitored""",
    )
    _coupling_variable = None
    non_integrated_variables = ["H"]
    H_save = None

    def __init__(self):
        super().__init__()
        self.a = np.array([0.27])
        self.b = np.array([0.108])
        self.d = np.array([154])
        self.gamma = np.array([0.641])
        self.tau_s = np.array([100])
        self.J_N = np.array([0.2609])
        self.I_o = np.array(
            [0.3]
        )  # --> Slightly different from TVB default's value (0.33)
        self.w = np.array(
            [1]
        )  # --> That's the one important here because TVB's default value is 0.6.

    def update_state_variables_before_integration(self, *_):
        return None

    def update_state_variables_after_integration(self, X):
        X[1, :] = self.H_save  # only works for Euler integrator
        return X

    def dfun(self, x, c, local_coupling=0.0):
        x_ = x.reshape(x.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T + local_coupling * x[0]
        deriv, H = _numba_dfun(
            x_,
            c_,
            self.a,
            self.b,
            self.d,
            self.gamma,
            self.tau_s,
            self.w,
            self.J_N,
            self.I_o,
        )
        self.H_save = H
        return deriv


class TvbSim:
    def __init__(
        self,
        connectivity,
        proxy,
        dt,
        min_delay,
    ):
        self.nb_node = connectivity.weights.shape[0] - len(proxy)

        model = ReducedWongWangProxy()
        integrator = Euler(
            dt=dt,
            bounded_state_variable_indices=np.array([0]),
            state_variable_boundaries=np.array([[0.0, 1.0]]),
        )
        self.sim = CoSimulator(
            voi=np.array([0]),  # variables of interest?
            synchronization_time=min_delay,
            cosim_monitors=(CoMonitor(),),
            proxy_inds=np.asarray(proxy, dtype=np.int_),
            model=model,
            connectivity=connectivity,
            coupling=Linear(a=np.array(0.096)),  # TODO make this configurable
            integrator=integrator,
            monitors=(Monitor(variables_of_interest=np.array(0)),),
        )
        self.sim.configure()
        self.dt = self.sim.integrator.dt
        self.current_state = None

    def __call__(self, time, proxy_data=None):
        if proxy_data is not None:
            print(proxy_data[0].shape, proxy_data[1].shape)
            proxy_data[1] = np.reshape(
                proxy_data[1],
                (
                    proxy_data[1].shape[0],
                    self.sim.voi.shape[0],
                    self.sim.proxy_inds.shape[0],
                    self.sim.model.number_of_modes,
                ),
            )
        result_delayed = self.sim.run(cosim_updates=proxy_data)
        result = self.sim.loop_cosim_monitor_output()
        return (
            result[0][0],
            result[0][1][:, 1].squeeze(),
            result_delayed[0][1][:, 1].squeeze(),
        )
