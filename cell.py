import arbor as A
from arbor import units as U
import numpy as np
import subprocess as sp
from pathlib import Path

# Spherical geometry from the paper
#  - V = 4/3 pi r^3 = pi r^3
#  - A = 4 pi r^2
# now compute the equivalent square
# cylinder mantle:
# - A = 2 pi l^2
VOLUME = 2160
RADIUS = (0.75 * VOLUME / np.pi) ** (1 / 3) * U.um
AREA = 4 * np.pi * RADIUS * RADIUS


def compile_catalogue(cnm="local", mod="mod"):
    here = Path(__file__).resolve().parent
    cat = (here / f"{cnm}-catalogue.so").resolve()
    mod = (here / mod).resolve()

    print(f"Compiling catalogue '{cnm}' from '{mod}' to {cat}")
    recompile = not cat.exists()
    for src in mod.glob("*.mod"):
        if recompile:
            break
        src = Path(src).resolve()
        recompile |= src.stat().st_mtime > cat.stat().st_mtime
    if recompile:
        sp.run(
            f"arbor-build-catalogue {cnm} {mod}",
            shell=True,
            check=True,
            capture_output=False,
        )
        sp.run(f"mv {cnm}-catalogue.so {cat}", shell=True, check=True)
    return A.load_catalogue(cat)


all = "(all)"
soma = "(tag 1)"
dend = "(tag 3)"
ctr = "(on-components 0.5 (tag 1))"
end = "(on-components 0.5 (tag 3))"


def make_point_cell(k_bath, *, t_toggle=None, k_bath1=17.0):
    l = np.sqrt(0.5 * AREA.value / np.pi)

    tree = A.segment_tree()
    root = A.mnpos
    root = tree.append(root, (0, 0, 0, l), (0, 0, l, l), tag=1)
    mrf = A.morphology(tree)

    dec = A.decor()
    if t_toggle:
        dec.paint(
            soma,
            A.density("hhplustoggle", kbath0=k_bath, t_toggle=t_toggle, kbath1=k_bath1),
        )
    else:
        dec.paint(soma, A.density("hhplus", kbath=k_bath))

    dec.set_property(Vm=-78 * U.mV, cm=0.00115 * U.F / U.m2)
    dec.place(ctr, A.threshold_detector(-25 * U.mV), "det")
    dec.place(ctr, A.synapse("expsyn"), "syn")

    cvp = A.cv_policy("(join (single (tag 1)) (max-extent 5 (tag 3)))")

    return A.cable_cell(mrf, dec, A.label_dict(), cvp)


class single_cell_recipe(A.recipe):
    def __init__(self, k_bath):
        A.recipe.__init__(self)
        self.cable_props = A.neuron_cable_properties()
        self.cable_props.set_ion(
            ion="cl",
            valence=-1,
            int_con=5.0 * U.mM,
            ext_con=112 * U.mM,
            rev_pot=-26.64 * np.log(112.0 / 5.0) * U.mV,
        )
        self.cable_props.set_ion(
            ion="na",
            valence=1,
            int_con=16.0 * U.mM,
            ext_con=138.0 * U.mM,
            rev_pot=26.64 * np.log(138 / 16) * U.mV,
        )
        self.cable_props.set_ion(
            ion="k",
            valence=1,
            int_con=140.0 * U.mM,
            ext_con=4.8 * U.mM,
            rev_pot=26.64 * np.log(4.8 / 140.0) * U.mV,
        )
        self.cable_props.catalogue.extend(compile_catalogue(), "")
        self.the_cell = make_point_cell(k_bath)

    def num_cells(self):
        return 1

    def cell_kind(self, _):
        return A.cell_kind.cable

    def cell_description(self, _):
        return self.the_cell

    def global_properties(self, _):
        return self.cable_props
