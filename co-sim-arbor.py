#!/usr/bin/env python3

from traceback import print_exception
import arbor as A
from arbor import units as U
import numpy as np
import polars as pl
from mpi4py import MPI
from cell import make_point_cell, single_cell_recipe
from argparse import ArgumentParser
from pathlib import Path
from reduced_wong_wang import TvbSim, Connectivity

args = ArgumentParser(
    "cosim-arbor",
    description="Co-simulation with a mixture of healthy and pathological cells.",
)

args.add_argument(
    "-T", "--final-time", default=10000, help="Stop time [ms]", type=float
)
args.add_argument("-t", "--time-step", default=0.01, help="Time step [ms]", type=float)
args.add_argument("-N", "--cell-count", default=10, help="Cell count", type=int)
args.add_argument(
    "-p",
    "--pathological-fraction",
    default=1.0,
    help="Fraction of cells in pathological state.",
    type=float,
)
args.add_argument("-w", "--weight", default=0.5, help="Connection weight.", type=float)
args.add_argument("-l", "--lag", default=0.5, help="Connection lag [ms]", type=float)
args.add_argument(
    "-b",
    "--k-bath-bad",
    default=17.0,
    help="kbath for pathological cells [mM]",
    type=float,
)
args.add_argument(
    "-o", "--k-bath-ok", default=7.5, help="kbath for healthy cells [mM]", type=float
)

args.add_argument(
    "-x",
    "--proxy-region",
    default=0,
    help="Where to put the Arbor region.",
    type=int,
)

args.add_argument(
    "-s",
    "--output",
    default="cosim",
    help="Sub-directory of where to put the data.",
    type=str,
)

args.add_argument(
    "-i",
    "--beta",
    default=1,
    help="Beta parameter of simulation.",
    type=float,
)

args.add_argument(
    "-z",
    "--toggle",
    default=None,
    help="Pathological cell start healthy and switch behaviours at [ms]",
    type=float,
)

args.add_argument(
    "-a",
    "--skip-arbor-output",
    default=True,
    help="Store spikes and membrane potentials.",
    action="store_false",
)

opts = args.parse_args()
T = opts.final_time
dt = opts.time_step
k_bath_ok = opts.k_bath_ok
k_bath_bad = opts.k_bath_bad
N = opts.cell_count
n_ok = N - int(opts.pathological_fraction * N)
delay = opts.lag
weight = opts.weight
ext_delay = 1.0
proxy_id = opts.proxy_region
beta = opts.beta / N
toggle = opts.toggle

G = 100

here = Path(__file__).parent.resolve()
outd = here / "data" / opts.output
outd.mkdir(exist_ok=True, parents=True)

rng = np.random.default_rng()


raw = Connectivity.from_file(str(here / "connectivity_mouse.zip"))
for i in range(len(raw.weights)):
    raw.weights[i, i] = 0
raw.tract_lengths[raw.tract_lengths < ext_delay] = ext_delay
raw.configure()
connectivity = raw

n_nodes = connectivity.weights.shape[0]
tvb_nodes = [id for id in range(n_nodes) if not id == proxy_id]


class TVB:
    def __init__(self, *, proxy_id=None, dt=0.0025, connectivity=None, min_delay=10000):
        if connectivity is None:
            raise RuntimeError("No connectivity given.")
        if proxy_id is None:
            raise RuntimeError("No proxy id given.")
        self.connectivity = connectivity
        self.n_nodes = n_nodes
        self.proxy = proxy_id
        assert 0 <= self.proxy < self.n_nodes, (
            f"Proxy node <{self.proxy}> oob for [0, {self.n_nodes - 1}]"
        )
        self.tvb_nodes = tvb_nodes
        self.dt = dt
        self.min_delay = 0.5 * min(min_delay, self.connectivity.tract_lengths.min())
        print(self.connectivity.tract_lengths)
        assert 0 < self.dt <= self.min_delay, (
            f"Time step <{self.dt}> oob for (0, {self.min_delay}]"
        )
        self.sim = TvbSim(self.connectivity, [self.proxy], self.dt, self.min_delay)
        self.rng = np.random.default_rng()

    def step(self, rates):
        return self.sim(self.min_delay, proxy_data=rates)


class recipe(single_cell_recipe):
    def __init__(
        self,
        ext_gids,
        *,
        N=N,
        weight=weight,
        delay=delay,
        ext_weight=weight,
        ext_delay=ext_delay,
        n_healthy,
        k_bath_ok,
        k_bath_bad,
    ):
        single_cell_recipe.__init__(self, k_bath_ok)
        self.count = N
        self.n_healthy = n_healthy
        if toggle:
            self.pathological = make_point_cell(
                k_bath=k_bath_ok, t_toggle=toggle, k_bath1=k_bath_bad
            )
        else:
            self.pathological = make_point_cell(k_bath=k_bath_bad)
        self.healthy = make_point_cell(k_bath=k_bath_ok)

        self.delay = delay
        self.weight = weight
        self.ext_gids = ext_gids
        self.ext_weight = ext_weight
        self.ext_delay = ext_delay

    def num_cells(self):
        return self.count

    def cell_description(self, gid):
        if gid < self.n_healthy:
            return self.healthy
        else:
            return self.pathological

    def connections_on(self, gid):
        return [
            A.connection(
                (src, "det"),
                "syn",
                weight=self.weight,
                delay=self.delay * U.ms,
            )
            for src in range(0, self.count)
            if not src == gid
        ]

def convert_arb_spikes_ca(ca_0, dt, time, t0, spikes):
    """convert Arbor spikes into rates-like variable for TVB."""
    n_bin = int(np.ceil(time / dt))
    ca_like_activity = np.zeros(n_bin)
    spike_times = np.array([s.time for s in spikes])

    tau = 100.0
    time_steps = np.arange(0, time, dt) + t0

    ca = ca_0
    for ix in range(n_bin):
        tlo = time_steps[ix]
        thi = tlo + dt
        rg = spike_times < thi
        lf = spike_times >= tlo
        ca += beta * (rg & lf).sum()
        ca *= np.exp(-dt / tau)
        ca_like_activity[ix] = ca

    return [time_steps, ca_like_activity * G], ca


def gen_tvb_spikes(times, rates, min_delay, dt, tvb_nodes):
    # TODO Customization point: Poisson distribution is assumed.
    n_bin, _ = rates.shape
    spikes = []
    for ix in tvb_nodes:
        for iy in range(n_bin):
            time = times[iy]
            rate = rates[iy, ix]
            if not np.isfinite(rate):
                continue
            n_spike = int(np.ceil(min_delay * rate))
            count = 0
            while count < n_spike:
                ts = dt * rng.uniform(size=n_spike)
                for s in ts:
                    if s > dt:
                        continue
                    spikes.append(A.remote.arb_spike(ix, 0, s + time))
                    count += 1
                    if count >= n_spike:
                        break
    return spikes


tag = ""
if toggle is not None:
    tag += f"-toggle={toggle}"

wrd = MPI.COMM_WORLD
assert wrd.size == 2, "Must use exactly 2 MPI tasks."
if wrd.rank == 0:
    ldr = 1
else:
    ldr = 0
grp = wrd.Split(wrd.rank)
itr = grp.Create_intercomm(0, wrd, ldr, 42)

if wrd.rank == 0:  # running Arbor
    try:
        # TODO adjust external gids
        rec = recipe(
            N=N,
            weight=weight,
            delay=delay,
            k_bath_ok=k_bath_ok,
            k_bath_bad=k_bath_bad,
            ext_gids=tvb_nodes,
            ext_delay=ext_delay,
            ext_weight=weight,
            n_healthy=n_ok,
        )
        ctx = A.context(mpi=grp, inter=itr)
        sim = A.simulation(rec, ctx)
        hdls = []

        sim.run(T * U.ms, dt * U.ms)
    except Exception as e:
        print("PANIC!!!! Arbor broke with:")
        print_exception(e)
        wrd.Abort()
else:  # running TVB
    try:
        t = 0
        sim = TVB(dt=dt, min_delay=delay, connectivity=connectivity, proxy_id=proxy_id)
        ca_0 = 0.0
        from_arb_rates = None
        dfs = []
        while True:
            msg = A.remote.exchange_ctrl(A.remote.msg_epoch(t, t + sim.min_delay), itr)
            if isinstance(msg, A.remote.msg_abort):
                print(f"PANIC!!!! Arbor sent an abort {msg.reason}")
                wrd.Abort()
            elif isinstance(msg, A.remote.msg_done):  # Arbor finished
                break
            elif isinstance(
                msg, A.remote.msg_epoch
            ):  # getting msg from Arbor line 245, if Arbor is still running
                times, rates, rates_old = sim.step(
                    from_arb_rates
                )  # we feed a different input to TVB
                to_arb_spikes = gen_tvb_spikes(times, rates, delay, dt, sim.tvb_nodes)
                n_to_arb = len(to_arb_spikes)
                from_arb_spikes = A.remote.gather_spikes(to_arb_spikes, itr)

                n_from_arb = len(from_arb_spikes)  # modify to something else
                from_arb_rates, ca_0 = convert_arb_spikes_ca(
                    ca_0, dt, sim.min_delay, t, from_arb_spikes
                )

                rates[:, proxy_id] = from_arb_rates[1].squeeze() / G
                dfs.append(rates)
    except Exception as e:
        print("PANIC!!!! TVB broke with")
        print_exception(e)
        wrd.Abort()
    fn = f"activity{tag}-N={N}-T={T}-f={opts.pathological_fraction}-k={k_bath_ok}|{k_bath_bad}.parquet"
    print(outd, fn)
    df = pl.DataFrame(np.concatenate(dfs, axis=0)) / G
    df.write_parquet(outd / fn)
