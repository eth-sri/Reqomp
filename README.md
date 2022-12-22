# Reqomp

Reqomp is a procedure to automatically synthesize uncomputation in a given
quantum circuit with space constraints.

This repository integrates Reqomp into Qiskit, allowing the programmer to mark
quantum bits as ancilla variables and give a maximal number of ancilla qubits for the  in circuit. Ancilla variables are then safely uncomputed and recomputed on 
these limited qubits by Reqomp.

## Getting Started Guide

### Installation

You can install this project using [pip](https://pypi.org/project/pip/). For
example, to install via [conda](https://docs.conda.io/en/latest/), use

```bash
conda create --name reqomp --yes python=3.10
conda activate reqomp
pip install .
```

For development of Reqomp, install it in editable mode as follows:

```bash
pip install -e .
```

## Start Programming

In the following, we provide some examples using Reqomp. As Reqomp extends
Qiskit, we refer to the [Qiskit
tutorials](https://qiskit.org/documentation/tutorials/circuits/1_getting_started_with_qiskit.html)
for a more thorough introduction into building circuits and using custom gates.

The following code snippet creates a simple circuit applying an H gate on qubit t controlled by qubits o, p, q and r:

```python
from qiskit.circuit import QuantumRegister, QuantumCircuit, AncillaRegister
from qiskit.circuit.library.standard_gates import HGate
from reqomp.ancilla_circuit import AncillaCircuit

x = QuantumRegister(1, name = 'x')
y = QuantumRegister(1, name = 'y')
b = QuantumRegister(1, name = 'b')
c = AncillaRegister(1, name = 'c')

o = QuantumRegister(1, 'o')
p = QuantumRegister(1, 'p')
q = QuantumRegister(1, 'q')
r = QuantumRegister(1, 'r')
t = QuantumRegister(1, 't')
a = AncillaRegister(1, 'a')
b = AncillaRegister(1, 'b')
c = AncillaRegister(1, 'c')

circ = AncillaCircuit(o, p, q, r, a, b, c, t)

circ.ccx(o, p, a) # all gates can be used on an AncillaCircuit directly
circ.ccx(q, a, b)
circ.ccx(r, b, c)
circ.append(HGate().control(1), [c, t]) 

print(circ) # AncillaCircuit contains no uncomputation
                     
# o: ──■─────────────────
#      │                 
# p: ──■─────────────────
#      │                 
# q: ──┼────■────────────
#      │    │            
# r: ──┼────┼────■───────
#    ┌─┴─┐  │    │       
# a: ┤ X ├──■────┼───────
#    └───┘┌─┴─┐  │       
# b: ─────┤ X ├──■───────
#         └───┘┌─┴─┐     
# c: ──────────┤ X ├──■──
#              └───┘┌─┴─┐
# t: ───────────────┤ H ├
#                   └───┘


# Uncomputation is added using the provided number of ancilla qubits, here 3. Uncomputed CCX gates are further replaced by the Margolus RCCX gate. The AncillaCircuit is converted to a qiskit QuantumCircuit, which does not distinguish ancillas from other qubits.
# Here 3 ancilla qubits are used, so all ancilla variables can be computed and uncomputed exactly once.
circ3 = circ.uncompute(3)
print(circ3)

#     ┌───────┐                                         ┌───────┐
#  o: ┤0      ├─────────────────────────────────────────┤0      ├
#     │       │                                         │       │
#  p: ┤1      ├─────────────────────────────────────────┤1      ├
#     │       │┌───────┐                       ┌───────┐│       │
#  q: ┤       ├┤0      ├───────────────────────┤0      ├┤       ├
#     │       ││       │┌───────┐     ┌───────┐│       ││       │
#  r: ┤       ├┤       ├┤0      ├─────┤0      ├┤       ├┤       ├
#     │  Rccx ││       ││       │┌───┐│       ││       ││  Rccx │
#  t: ┤       ├┤       ├┤       ├┤ H ├┤       ├┤       ├┤       ├
#     │       ││  Rccx ││  Rccx │└─┬─┘│  Rccx ││  Rccx ││       │
# a1: ┤       ├┤       ├┤2      ├──■──┤2      ├┤       ├┤       ├
#     │       ││       ││       │     │       ││       ││       │
# a0: ┤       ├┤2      ├┤1      ├─────┤1      ├┤2      ├┤       ├
#     │       ││       │└───────┘     └───────┘│       ││       │
# a2: ┤2      ├┤1      ├───────────────────────┤1      ├┤2      ├
#     └───────┘└───────┘                       └───────┘└───────┘

# Now only 2 ancilla qubits are used, so ancilla variable a is uncomputed early and recomputed, to allow ancilla variable c to use the same ancilla qubit a4.
circ2 = circ.uncompute(2)
print(circ2)

#     ┌───────┐         ┌───────┐                       ┌───────┐         ┌───────┐
#  o: ┤0      ├─────────┤0      ├───────────────────────┤0      ├─────────┤0      ├
#     │       │         │       │                       │       │         │       │
#  p: ┤1      ├─────────┤1      ├───────────────────────┤1      ├─────────┤1      ├
#     │       │┌───────┐│       │                       │       │┌───────┐│       │
#  q: ┤       ├┤0      ├┤       ├───────────────────────┤       ├┤0      ├┤       ├
#     │       ││       ││       │┌───────┐     ┌───────┐│       ││       ││       │
#  r: ┤  Rccx ├┤       ├┤  Rccx ├┤0      ├─────┤0      ├┤  Rccx ├┤       ├┤  Rccx ├
#     │       ││       ││       ││       │┌───┐│       ││       ││       ││       │
#  t: ┤       ├┤  Rccx ├┤       ├┤       ├┤ H ├┤       ├┤       ├┤  Rccx ├┤       ├
#     │       ││       ││       ││  Rccx │└─┬─┘│  Rccx ││       ││       ││       │
# a3: ┤       ├┤2      ├┤       ├┤1      ├──┼──┤1      ├┤       ├┤2      ├┤       ├
#     │       ││       ││       ││       │  │  │       ││       ││       ││       │
# a4: ┤2      ├┤1      ├┤2      ├┤2      ├──■──┤2      ├┤2      ├┤1      ├┤2      ├
#     └───────┘└───────┘└───────┘└───────┘     └───────┘└───────┘└───────┘└───────┘  
```

### Composing circuits

To compose AncillaCircuits, they can be transformed into an AncillaGate. When appending such a gate to an AncillaCircuit, all needed ancillas are appended automatically.

```python
from qiskit import QuantumRegister, AncillaRegister
from reqomp.ancilla_circuit import AncillaCircuit

x = QuantumRegister(1, name = 'x')
y = QuantumRegister(1, name = 'y')
b = QuantumRegister(1, name = 'b')
c = AncillaRegister(1, name = 'c')

circ = AncillaCircuit(x, y, b, c)
circ.ccx(b, x, c)
circ.cx(b, x)
circ.cx(c, y)

gate = circ.to_ancilla_gate() # control qubits should come first, then target qubit then ancillae.

z = QuantumRegister(1, name = 'z')
t = QuantumRegister(1, name = 't')
d = QuantumRegister(1, name = 'd')
circ2 = AncillaCircuit(z, t, d)

# append gate to circ2, adding d to (z, t), the ancilla c is appended automatically
circ2.append(gate, [z, t, d])
# appending it again, a new ancilla is added again
circ2.append(gate, [z, t, d])

print(circ2)

# Output:
#     ┌─────────────┐┌─────────────┐
#  z: ┤0            ├┤0            ├
#     │             ││             │
#  t: ┤1            ├┤1            ├
#     │  circuit-85 ││             │
#  d: ┤2            ├┤2 circuit-85 ├
#     │             ││             │
# a0: ┤3            ├┤             ├
#     └─────────────┘│             │
# a1: ───────────────┤3            ├
#                    └─────────────┘

circ2 = circ2.uncompute(1) # the two ancillas are now allocated on the same qubit

print(circ2)

# Output:
#     ┌───────┐     ┌───────┐┌───┐┌───────┐     ┌───────┐┌───┐
#  z: ┤1      ├─────┤1      ├┤ X ├┤1      ├─────┤1      ├┤ X ├
#     │       │┌───┐│       │└─┬─┘│       │┌───┐│       │└─┬─┘
#  t: ┤       ├┤ X ├┤       ├──┼──┤       ├┤ X ├┤       ├──┼──
#     │  Rccx │└─┬─┘│  Rccx │  │  │  Rccx │└─┬─┘│  Rccx │  │  
#  d: ┤0      ├──┼──┤0      ├──■──┤0      ├──┼──┤0      ├──■──
#     │       │  │  │       │     │       │  │  │       │     
# a3: ┤2      ├──■──┤2      ├─────┤2      ├──■──┤2      ├─────
#     └───────┘     └───────┘     └───────┘     └───────┘     

```

## Step-by-Step Instructions to Reproduce Evaluation

In the following, we describe how to reproduce the evaluation results from the Reqomp paper.

### Organization

The Reqomp implementation is made of the files in the folder [reqomp](reqomp/).

All examples presented in the submitted paper are implemented using Reqomp and
can be found in [reqomp/examples/](reqomp/examples/).

### Paper Claims

To execute Reqomp on the small examples, run

```bash
python run_evaluation.py
```

This writes its results to [evaluation_results](evaluation_results). While this command should not crash, some error and warning messages are expected.

To run both big and small examples, run

```bash
python run_evaluation.py -a
```

Running the Unqomp baseline requires setting up a new environment as described in [unqomp/README.md](unqomp/README.md) (we run Unqomp with Qiskit 0.31.0, the latest Qiskit version it is compatible with):

```bash
cd unqomp
conda create --name unqomp --yes python=3.8
conda activate unqomp
pip install .
cd -
```

Then, run both big and small examples in Unqomp using:

```bash
python run_evaluation.py -u -a
```

Complete results for both Reqomp and Unqomp are already present in [evaluation_results](evaluation_results).

All results can be parsed with:

```bash
python plots.py
```

This produces the data for the evaluation and appendix tables, and plots run times, gate counts and depth for selected examples. Note that as Reqomp does not optimize for depth, depth can slightly change from one run to another.
