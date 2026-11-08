# Quantum_Poincare_Return_Map



"""
Quantum_Poincare_Return_Map_V1.py

Python 3.11
Qiskit 1.4.4

OSNOVNA IDEJA
------------
1. Istorija kola predstavlja diskretnu putanju.
2. Iz treninga se odredjuje orijentisani presek.
3. Registruju se samo uzlazni prelasci kroz presek.
4. Uzastopni prelasci formiraju povratnu mapu.
5. Lokalna regresija prethodnih povrataka daje karakteristike
   ocekivanog sledeceg povratka.
6. Kvantna slicnost povezuje aktuelni kontekst sa istorijskim.
7. Qiskit operator daje Bornove raspodele izlaznih stanja.
8. Uce se pocetna raspodela i sest uslovnih prelaza
   izmedju susednih pozicija u kombinaciji.
9. Tacan globalni MAP odredjuje se nad svim legalnim kombinacijama.

KVANTNI DEO
-----------
Istorijski kontekst kodira se amplitudama sest kubita.

    F(a, b) = |<phi(a)|phi(b)>|^2

Batched racunanje preklopa je tacna algebraicka simulacija.
Qiskit kolo nezavisno proverava preklop preko pripreme
jednog stanja i inverzne pripreme drugog.

Izlazni operator:
    U(tau) = exp(-i * tau * H)

H povezuje susedna brojna bazna stanja 1..39.
Ovaj operator je moja prediktivna dopuna.
Ne predstavlja gravitacioni Hamiltonijan.

Qiskit simulira svih 39 relevantnih kolona operatora.
Njihove amplitude odredjuju sve izlazne verovatnoce.

Lokalni izlaz je mesavina kvantnih stanja sa tezinama
odredjenim istorijskim kontekstom.

KLASICNI DEO
------------
Rekonstrukcija preseka, lokalna regresija povratne mape,
tezine lokalne regresije gustine i dinamicko programiranje.

Nema klasifikatora.
Nema zaokruzivanja regresionih srednjih vrednosti u NEXT.
Nema slucajnih kombinacija ili kozmetickih filtera.

SVI PODACI
----------
Svaki trening koristi svaki dostupan istorijski prelaz.
Lokalne tezine su pozitivne za sve trening primere.
Parametar susedstva odredjuje prostornu skalu, a ne odseca istoriju.

Validacija i test su hronoloski.
Presek, skaliranje i trening primeri zamrznuti su unutar evaluacije.
Ranija evaluaciona kola mogu biti poznati ulaz kasnijih kola.
Evaluacioni ciljevi ne dodaju se u trening.

Zavrsni model koristi celu istoriju svakog CSV-a zasebno.

SIMULACIJA
----------
Sedam sestokubitnih izlaznih registara predstavlja se
preko uslovnih raspodela.
Gust vektor od 42 kubita se ne pravi.
Provera dvoregistarskog kvantnog operatora koristi 12 kubita.
"""


import csv
import hashlib
import itertools
import random
from pathlib import Path

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, UnitaryGate
from qiskit.quantum_info import Statevector
from scipy.linalg import expm
from scipy.special import logsumexp


random.seed(39)
np.random.seed(39)


FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4716_k89.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4716_k89_loto_2980.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4716_k89_loto_plus_1736.csv"),
)

N = 39
K = 7
QUBITS = 6
DIM = 1 << QUBITS

WINDOWS = (1, 3)
NEIGHBOR_SCALES = (32, 128)
WALK_TIMES = (0.5, 1.5)

KERNEL_POWER = 4.0

TRAIN_FRACTION = 0.70
VALIDATION_END_FRACTION = 0.85

PROGRESS_EVERY = 100


class DistributionError(ArithmeticError):
    pass


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def load_csv(path):
    raw = path.read_bytes()
    rows = []

    for line_number, row in enumerate(
        csv.reader(raw.decode("utf-8-sig").splitlines()),
        start=1,
    ):
        if not row or all(not cell.strip() for cell in row):
            continue

        require(
            len(row) == K,
            f"{path.name}, red {line_number}: potrebno je sedam kolona.",
        )

        try:
            values = [int(cell.strip()) for cell in row]
        except ValueError as exc:
            raise RuntimeError(
                f"{path.name}, red {line_number}: necelobrojna vrednost."
            ) from exc

        require(
            len(set(values)) == K
            and all(1 <= value <= N for value in values)
            and values == sorted(values),
            f"{path.name}, red {line_number}: nelegalna kombinacija.",
        )
        rows.append(values)

    require(len(rows) >= 5, f"{path.name}: nedovoljno kola.")

    return {
        "path": path,
        "data": np.asarray(rows, dtype=np.int64) - 1,
        "digest": hashlib.sha256(raw).hexdigest(),
    }


def amplitude_states(features):
    """Realne karakteristike -> normalizovane sestokubitne amplitude."""
    features = np.asarray(features, dtype=np.float64)

    require(
        features.ndim == 2
        and features.shape[1] + 1 <= DIM
        and np.isfinite(features).all(),
        "Neispravne karakteristike za amplitudno kodiranje.",
    )

    states = np.zeros((len(features), DIM), dtype=np.float64)
    states[:, 0] = 1.0
    states[:, 1:features.shape[1] + 1] = features
    states /= np.linalg.norm(states, axis=1, keepdims=True)

    require(
        np.allclose(
            np.sum(states ** 2, axis=1),
            1.0,
            atol=2e-12,
            rtol=0.0,
        ),
        "Ulazna kvantna stanja nisu normalizovana.",
    )
    return states


def qiskit_fidelity(left, right):
    circuit = QuantumCircuit(QUBITS)
    circuit.append(StatePreparation(right), range(QUBITS))
    circuit.append(StatePreparation(left).inverse(), range(QUBITS))

    actual = float(
        abs(Statevector.from_instruction(circuit).data[0]) ** 2
    )
    expected = float(abs(np.vdot(left, right)) ** 2)

    require(
        np.isclose(actual, expected, atol=2e-9, rtol=2e-9),
        "Qiskit i algebraicki kvantni preklop se razlikuju.",
    )
    return actual


def local_weights(states, query, neighbors):
    """
    Sve trening tacke dobijaju pozitivnu tezinu.
    k-ti sused odredjuje samo skalu lokalnog jezgra.

    Jezgro nije Gaussian:
        w_i proporcionalno (1 + distance_i / radius)^(-power).
    """
    require(
        states.ndim == 2
        and len(states) > 0
        and query.shape == (DIM,)
        and neighbors >= 1,
        "Neispravno lokalno kvantno susedstvo.",
    )

    fidelity = np.abs(states @ query.conj()) ** 2
    distances = np.clip(1.0 - fidelity, 0.0, 1.0)

    position = min(int(neighbors), len(distances)) - 1
    radius = float(np.partition(distances, position)[position])

    if radius <= 0.0:
        positive = distances[distances > 0.0]
        radius = float(positive.min()) if len(positive) else 1.0

    log_weights = -KERNEL_POWER * np.log1p(distances / radius)
    log_weights -= logsumexp(log_weights)
    weights = np.exp(log_weights)

    require(
        np.isfinite(weights).all()
        and np.all(weights > 0.0)
        and np.isclose(weights.sum(), 1.0, atol=2e-12),
        "Neispravne lokalne tezine.",
    )

    effective = float(1.0 / np.dot(weights, weights))
    return weights, effective, radius


def fit_section(data, training_transitions):
    """
    Presek se odredjuje samo iz istorijskih ulaza treninga.

    Za trening ciljeve 1..T ulazni redovi su 0..T-1.
    """
    require(
        1 <= training_transitions < len(data),
        "Neispravna granica za ucenje preseka.",
    )

    training = data[:training_transitions].astype(np.float64)

    mean = training.mean(axis=0)
    scale = training.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)

    standardized = (training - mean) / scale
    covariance = standardized.T @ standardized / len(training)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    if eigenvalues[-1] <= 1e-14:
        direction = np.zeros(K)
        direction[0] = 1.0
    else:
        direction = eigenvectors[:, -1].copy()

    largest = int(np.argmax(np.abs(direction)))

    if direction[largest] < 0.0:
        direction *= -1.0

    level = float(np.median(standardized @ direction))

    return {
        "mean": mean,
        "scale": scale,
        "direction": direction,
        "level": level,
    }


def section_events(data, section):
    """
    Orijentisani diskretni presek sa linearnom interpolacijom.

    Prelaz je poznat tek kada je poznat njegov desni red.
    """
    coordinates = (
        data.astype(np.float64) - section["mean"]
    ) / section["scale"]

    height = (
        coordinates @ section["direction"] - section["level"]
    )

    right_indices = []
    crossing_times = []
    points = []

    for right in range(1, len(data)):
        left_height = float(height[right - 1])
        right_height = float(height[right])

        if left_height < 0.0 <= right_height:
            fraction = -left_height / (right_height - left_height)

            point = (
                (1.0 - fraction) * coordinates[right - 1]
                + fraction * coordinates[right]
            )

            require(
                0.0 <= fraction <= 1.0
                and np.isclose(
                    point @ section["direction"],
                    section["level"],
                    atol=2e-10,
                ),
                "Interpolirani prelaz nije na preseku.",
            )

            right_indices.append(right)
            crossing_times.append(right - 1 + fraction)
            points.append(point)

    return {
        "right": np.asarray(right_indices, dtype=np.int64),
        "time": np.asarray(crossing_times, dtype=np.float64),
        "points": np.asarray(points, dtype=np.float64).reshape(-1, K),
    }


def causal_return_features(
    data, training_transitions, window, neighbors, events
):
    """
    Karakteristike za ciljne redove t=1..len(data).
    t=len(data) predstavlja NEXT.

    Povratna regresija za vreme t koristi samo parove ciji je
    zavrsni prelaz poznat do min(t-1, granica_treninga).
    """
    count = len(data)
    lag_columns = K * window + window - 1
    result = np.zeros((count, lag_columns + 3 * K + 5))

    points = events["points"]
    event_right = events["right"]
    event_times = events["time"]

    if len(points) >= 2:
        start_states = amplitude_states(points[:-1])
        jumps = points[1:] - points[:-1]
        durations = np.diff(event_times)
        pair_end_right = event_right[1:]
    else:
        start_states = np.empty((0, DIM))
        jumps = np.empty((0, K))
        durations = np.empty(0)
        pair_end_right = np.empty(0, dtype=np.int64)

    spreads = np.full(count, np.nan)
    effective_returns = np.zeros(count)
    used_pairs = np.zeros(count, dtype=np.int64)

    for row, time in enumerate(range(1, count + 1)):
        for lag in range(1, window + 1):
            source = time - lag

            if source >= 0:
                start = K * (lag - 1)
                result[row, start:start + K] = (
                    data[source].astype(float) + 1.0 - 20.0
                ) / 19.0

                if lag >= 2:
                    result[row, K * window + lag - 2] = 1.0

        latest = int(
            np.searchsorted(event_right, time - 1, side="right")
        ) - 1

        if latest < 0:
            continue

        base = lag_columns
        result[row, base:base + K] = points[latest]

        age = max(0.0, time - 1 - event_times[latest])
        last_duration = 0.0

        if latest >= 1:
            result[row, base + K:base + 2 * K] = (
                points[latest] - points[latest - 1]
            )
            last_duration = float(
                event_times[latest] - event_times[latest - 1]
            )

        limit = min(time - 1, training_transitions)
        completed = int(np.searchsorted(
            pair_end_right, limit, side="right"
        ))

        result[row, base + 3 * K] = np.log1p(age)
        result[row, base + 3 * K + 1] = np.log1p(last_duration)
        result[row, base + 3 * K + 3] = 1.0

        if completed == 0:
            continue

        require(
            pair_end_right[completed - 1] <= limit,
            "Buduci povratak je usao u karakteristike.",
        )

        query = amplitude_states(points[latest:latest + 1])[0]
        weights, effective, _ = local_weights(
            start_states[:completed], query, neighbors
        )

        predicted_jump = weights @ jumps[:completed]
        predicted_duration = float(weights @ durations[:completed])

        result[row, base + 2 * K:base + 3 * K] = predicted_jump
        result[row, base + 3 * K + 2] = np.log1p(predicted_duration)
        result[row, base + 3 * K + 4] = 1.0

        squared_residuals = np.sum(
            (jumps[:completed] - predicted_jump) ** 2,
            axis=1,
        )

        spreads[row] = float(np.sqrt(weights @ squared_residuals))
        effective_returns[row] = effective
        used_pairs[row] = completed

    require(
        np.isfinite(result).all(),
        "Nekonacne karakteristike povratne mape.",
    )

    return result, {
        "spread": spreads,
        "effective": effective_returns,
        "pairs": used_pairs,
    }


def build_context(data, training_transitions, window, neighbors):
    require(window in WINDOWS, "Neispravan prozor istorije.")

    section = fit_section(data, training_transitions)
    events = section_events(data, section)

    raw, return_info = causal_return_features(
        data, training_transitions, window, neighbors, events
    )

    training = raw[:training_transitions]
    mean = training.mean(axis=0)
    scale = training.std(axis=0)
    keep = scale > 1e-12

    features = (
        (raw[:, keep] - mean[keep]) / scale[keep]
        if np.any(keep)
        else np.empty((len(raw), 0))
    )
    states = amplitude_states(features)

    training_event_count = int(np.searchsorted(
        events["right"], training_transitions, side="right"
    ))

    return {
        "cutoff": training_transitions,
        "window": window,
        "neighbors": neighbors,
        "section": section,
        "events": events,
        "raw": raw,
        "states": states,
        "training_states": states[:training_transitions],
        "targets": data[1:training_transitions + 1],
        "return_info": return_info,
        "training_return_pairs": max(0, training_event_count - 1),
    }


def make_quantum_walk(time):
    """
    U je blok dijagonalan:
        kvantni operator na 39 brojnih stanja,
        identitet na preostalih 25 baznih stanja.
    """
    hamiltonian = np.zeros((DIM, DIM), dtype=np.complex128)

    for value in range(N - 1):
        hamiltonian[value, value + 1] = 1.0
        hamiltonian[value + 1, value] = 1.0

    unitary = expm(-1j * float(time) * hamiltonian)

    require(
        np.allclose(
            unitary.conj().T @ unitary,
            np.eye(DIM),
            atol=2e-10,
            rtol=0.0,
        ),
        "Izlazni operator nije unitaran.",
    )

    gate = UnitaryGate(unitary, label=f"Return_U_{time}")
    amplitudes = np.empty((DIM, N), dtype=np.complex128)

    for source in range(N):
        circuit = QuantumCircuit(QUBITS)

        for bit in range(QUBITS):
            if (source >> bit) & 1:
                circuit.x(bit)

        circuit.append(gate, range(QUBITS))
        state = Statevector.from_instruction(circuit)
        amplitudes[:, source] = state.data

    require(
        np.allclose(
            amplitudes, unitary[:, :N],
            atol=2e-10, rtol=2e-10,
        )
        and np.allclose(
            amplitudes[N:], 0.0,
            atol=2e-12, rtol=0.0,
        ),
        "Qiskit izlazne amplitude nisu dosledne.",
    )

    probabilities = np.abs(amplitudes[:N]) ** 2
    column_sums = probabilities.sum(axis=0)

    require(
        np.allclose(
            column_sums, 1.0,
            atol=2e-10, rtol=0.0,
        ),
        "Qiskit izlazne kolone nisu normalizovane.",
    )

    # Korekcija iskljucivo za odstupanje normalizacije zbog zaokruzivanja.
    probabilities /= column_sums[None, :]

    return {
        "time": time,
        "unitary": unitary,
        "gate": gate,
        "probabilities": probabilities,
    }


def quantum_pair_probabilities(walk, first, second):
    circuit = QuantumCircuit(2 * QUBITS)

    for bit in range(QUBITS):
        if (first >> bit) & 1:
            circuit.x(bit)
        if (second >> bit) & 1:
            circuit.x(QUBITS + bit)

    circuit.append(walk["gate"], range(QUBITS))
    circuit.append(
        walk["gate"], range(QUBITS, 2 * QUBITS)
    )

    state = Statevector.from_instruction(circuit).data
    amplitudes = state.reshape(DIM, DIM).T
    return np.abs(amplitudes[:N, :N]) ** 2


def chain_from_weights(targets, weights, walk):
    """
    Regresija lokalne mesavine kvantnih izlaznih stanja.

    Za par pozicija:
        M_j = Q * B_j * Q^T

    B_j je lokalno ponderisana dvoregistarska bazna gustina.
    Q sadrzi Bornove verovatnoce Qiskit operatora.

    Iz M_j izvode se uslovni prelazi:
        C_j(v | u) = M_j(u, v) / sum_v M_j(u, v).

    Ovo zadrzava zavisnost susednih pozicija.
    """
    require(
        targets.shape == (len(weights), K)
        and np.all(weights > 0.0)
        and np.isclose(weights.sum(), 1.0, atol=2e-12),
        "Neispravni podaci lokalne regresije gustine.",
    )

    quantum = walk["probabilities"]

    initial_basis_mass = np.bincount(
        targets[:, 0], weights=weights, minlength=N
    )
    initial = quantum @ initial_basis_mass
    initial /= initial.sum()

    transitions = []

    for slot in range(1, K):
        pair_indices = targets[:, slot - 1] * N + targets[:, slot]
        basis_mass = np.bincount(
            pair_indices,
            weights=weights,
            minlength=N * N,
        ).reshape(N, N)

        joint = quantum @ basis_mass @ quantum.T
        joint /= joint.sum()
        row_mass = joint.sum(axis=1)

        conditional = np.zeros_like(joint)
        np.divide(
            joint,
            row_mass[:, None],
            out=conditional,
            where=row_mass[:, None] > 0.0,
        )
        transitions.append(conditional)

    transitions = np.asarray(transitions)

    # Provera normalizacije pre uslovljavanja na legalnu kombinaciju.
    propagated = initial.copy()

    for conditional in transitions:
        propagated = propagated @ conditional

    require(
        np.isfinite(initial).all()
        and np.isfinite(transitions).all()
        and np.all(initial >= 0.0)
        and np.all(transitions >= 0.0)
        and np.isclose(propagated.sum(), 1.0, atol=2e-9),
        "Lokalni uslovni lanac nije normalizovan.",
    )

    return initial, transitions


def safe_log(values):
    values = np.asarray(values, dtype=np.float64)
    result = np.full(values.shape, -np.inf)
    positive = values > 0.0
    result[positive] = np.log(values[positive])
    return result


def chain_distribution(initial, transitions):
    """
    Tacna raspodela lanca uslovljena na strogo rastuci niz.

    Razlikuje se od proizvoda sedam nezavisnih registara:
    svaka sledeca pozicija zavisi od prethodne vrednosti.
    """
    initial = np.asarray(initial, dtype=np.float64)
    transitions = np.asarray(transitions, dtype=np.float64)

    require(
        initial.ndim == 1 and transitions.ndim == 3,
        "Neispravne dimenzije uslovnog lanca.",
    )

    states = len(initial)
    slots = len(transitions) + 1

    require(
        transitions.shape == (slots - 1, states, states)
        and 1 <= slots <= states
        and np.isfinite(initial).all()
        and np.isfinite(transitions).all()
        and np.all(initial >= 0.0)
        and np.all(transitions >= 0.0),
        "Neispravan uslovni lanac.",
    )

    first_logs = safe_log(initial)
    edge_logs = safe_log(transitions)

    indices = np.arange(states)
    allowed = (
        (indices[None, :] >= np.arange(slots)[:, None])
        & (
            indices[None, :]
            <= states - slots + np.arange(slots)[:, None]
        )
    )

    first_logs[~allowed[0]] = -np.inf

    for slot in range(slots - 1):
        legal_edge = (
            (indices[:, None] < indices[None, :])
            & allowed[slot, :, None]
            & allowed[slot + 1, None, :]
        )
        edge_logs[slot, ~legal_edge] = -np.inf

    forward = np.full((slots, states), -np.inf)
    backward = np.full((slots, states), -np.inf)
    maximum = np.full((slots, states), -np.inf)
    parent = np.full((slots, states), -1, dtype=np.int64)

    forward[0] = first_logs
    maximum[0] = first_logs

    for slot in range(1, slots):
        forward[slot] = logsumexp(
            forward[slot - 1, :, None] + edge_logs[slot - 1],
            axis=0,
        )

        candidates = (
            maximum[slot - 1, :, None] + edge_logs[slot - 1]
        )
        parent[slot] = candidates.argmax(axis=0)
        maximum[slot] = candidates[
            parent[slot], np.arange(states)
        ]

    log_z = float(logsumexp(forward[-1]))

    if not np.isfinite(log_z):
        raise DistributionError("Nema pozitivne legalne mase.")

    backward[-1] = 0.0

    for slot in range(slots - 2, -1, -1):
        backward[slot] = logsumexp(
            edge_logs[slot] + backward[slot + 1][None, :],
            axis=1,
        )

    marginals = np.exp(forward + backward - log_z)

    path = np.empty(slots, dtype=np.int64)
    path[-1] = int(np.argmax(maximum[-1]))

    for slot in range(slots - 1, 0, -1):
        path[slot - 1] = parent[slot, path[slot]]

    log_mode = float(maximum[-1, path[-1]])

    require(
        np.all(np.diff(path) > 0)
        and log_mode <= log_z + 2e-9
        and log_z <= 2e-9
        and np.allclose(
            marginals.sum(axis=1), 1.0,
            atol=2e-9, rtol=0.0,
        ),
        "Neispravna legalna raspodela ili globalni maksimum.",
    )

    return {
        "path": path,
        "marginals": marginals,
        "log_z": log_z,
        "log_mode_probability": log_mode - log_z,
        "first_logs": first_logs,
        "edge_logs": edge_logs,
    }


def target_nll(distribution, target):
    target = np.asarray(target, dtype=np.int64)

    require(
        len(target) == len(distribution["path"])
        and np.all(np.diff(target) > 0),
        "Neispravna ciljna kombinacija.",
    )

    log_mass = float(distribution["first_logs"][target[0]])

    for slot in range(1, len(target)):
        log_mass += float(
            distribution["edge_logs"][
                slot - 1, target[slot - 1], target[slot]
            ]
        )

    loss = distribution["log_z"] - log_mass

    require(
        not np.isnan(loss) and loss >= -2e-8,
        "Neispravan uslovni NLL.",
    )
    return max(0.0, loss)


def predict(context, walk, time, neighbors=None):
    require(
        1 <= time <= len(context["states"]),
        "Neispravno vreme predikcije.",
    )

    if neighbors is None:
        neighbors = context["neighbors"]

    query = context["states"][time - 1]

    weights, effective, radius = local_weights(
        context["training_states"], query, neighbors
    )
    initial, transitions = chain_from_weights(
        context["targets"], weights, walk
    )
    distribution = chain_distribution(initial, transitions)

    return {
        "distribution": distribution,
        "initial": initial,
        "transitions": transitions,
        "weights": weights,
        "effective": effective,
        "radius": radius,
    }


def evaluate(context, walk, data, times, label):
    times = list(times)
    require(len(times) > 0, "Prazan evaluacioni skup.")
    require(
        min(times) > context["cutoff"],
        "Evaluacija se preklapa sa trening ciljevima.",
    )

    losses = []
    hits = []

    for index, time in enumerate(times, start=1):
        prediction = predict(context, walk, time)
        distribution = prediction["distribution"]

        losses.append(target_nll(distribution, data[time]))
        hits.append(
            len(
                set(distribution["path"].tolist())
                & set(data[time].tolist())
            )
        )

        if index % PROGRESS_EVERY == 0 or index == len(times):
            print(
                f"{label}: obradjeno {index}/{len(times)}",
                flush=True,
            )

    losses = np.asarray(losses)

    return {
        "nll": float(losses.mean()),
        "hits": np.asarray(hits, dtype=np.int64),
        "zero_targets": int(np.isposinf(losses).sum()),
    }


def print_metrics(label, metrics):
    print(label)
    print(f"Srednji uslovni NLL: {metrics['nll']:.9f}")
    print(f"Srednji broj pogodaka: {metrics['hits'].mean():.6f}")
    print(f"Ciljevi sa nultom masom: {metrics['zero_targets']}")
    print(
        "Broj kola po pogocima 0..7: "
        + str(np.bincount(metrics["hits"], minlength=K + 1).tolist())
    )


def verify_final_qiskit(context, prediction, time):
    query = context["states"][time - 1]
    count = len(context["training_states"])

    selected = np.unique([
        0,
        count // 2,
        count - 1,
        int(np.argmax(prediction["weights"])),
    ])

    for index in selected:
        qiskit_fidelity(
            context["training_states"][index], query
        )

    path = prediction["distribution"]["path"]

    for slot in range(K):
        probabilities = (
            prediction["initial"]
            if slot == 0
            else prediction["transitions"][slot - 1, path[slot - 1]]
        )

        require(
            np.isclose(probabilities.sum(), 1.0, atol=2e-9),
            "Podrzana uslovna grana nije normalizovana.",
        )

        amplitudes = np.zeros(DIM, dtype=np.complex128)
        amplitudes[:N] = np.sqrt(probabilities)
        amplitudes /= np.linalg.norm(amplitudes)

        circuit = QuantumCircuit(QUBITS)
        circuit.append(
            StatePreparation(amplitudes), range(QUBITS)
        )
        actual = Statevector.from_instruction(circuit).data

        require(
            np.allclose(
                np.abs(actual[:N]) ** 2,
                probabilities,
                atol=2e-9, rtol=2e-9,
            ),
            "Qiskit uslovna amplitudna grana nije dosledna.",
        )

    print("Zavrsne Qiskit provere preklopa i sedam uslovnih grana su prosle.")


def self_checks(walks):
    features = np.array([
        [0.2, -0.4, 0.7],
        [-0.3, 0.1, 0.6],
    ])
    states = amplitude_states(features)
    qiskit_fidelity(states[0], states[1])

    weights, _, _ = local_weights(states, states[0], 32)
    require(
        np.all(weights > 0.0)
        and np.isclose(weights.sum(), 1.0),
        "Provera lokalnih tezina nije prosla.",
    )

    # Kvantni izlaz para i linearnost mesavine.
    sources = [(0, 10), (8, 24), (20, 38)]
    mixture_weights = np.array([0.2, 0.3, 0.5])

    for walk in walks.values():
        actual_mixture = np.zeros((N, N))
        basis_mixture = np.zeros((N, N))

        for weight, (first, second) in zip(
            mixture_weights, sources
        ):
            actual_mixture += weight * quantum_pair_probabilities(
                walk, first, second
            )
            basis_mixture[first, second] += weight

        expected = (
            walk["probabilities"]
            @ basis_mixture
            @ walk["probabilities"].T
        )

        require(
            np.allclose(
                actual_mixture, expected,
                atol=2e-10, rtol=2e-9,
            ),
            "Qiskit dvoregistarska mesavina nije dosledna.",
        )

    # Potpuna provera malog uslovnog lanca.
    toy_initial = np.exp(
        0.4 * np.sin(np.arange(5, dtype=float))
    )
    toy_initial /= toy_initial.sum()

    toy_transitions = np.exp(
        0.3 * np.cos(np.arange(50, dtype=float).reshape(2, 5, 5))
    )
    toy_transitions /= toy_transitions.sum(axis=2, keepdims=True)

    toy = chain_distribution(toy_initial, toy_transitions)
    combinations = list(itertools.combinations(range(5), 3))
    scores = []

    for combination in combinations:
        score = np.log(toy_initial[combination[0]])

        for slot in range(1, 3):
            score += np.log(toy_transitions[
                slot - 1, combination[slot - 1], combination[slot]
            ])

        scores.append(score)

    scores = np.asarray(scores)
    brute_marginals = np.zeros((3, 5))

    for combination, mass in zip(
        combinations, np.exp(scores - logsumexp(scores))
    ):
        brute_marginals[np.arange(3), combination] += mass

    require(
        np.isclose(toy["log_z"], logsumexp(scores), atol=1e-12)
        and np.isclose(
            toy["log_mode_probability"],
            scores.max() - logsumexp(scores),
            atol=1e-12,
        )
        and np.allclose(
            toy["marginals"], brute_marginals,
            atol=1e-12, rtol=0.0,
        ),
        "Potpuna provera uslovnog lanca nije prosla.",
    )

    # Jedina legalna kombinacija.
    sole_initial = np.zeros(5)
    sole_initial[0] = 1.0
    sole_transitions = np.zeros((2, 5, 5))
    sole_transitions[0, 0, 2] = 1.0
    sole_transitions[1, 2, 4] = 1.0

    sole = chain_distribution(sole_initial, sole_transitions)

    require(
        np.array_equal(sole["path"], [0, 2, 4])
        and np.isclose(sole["log_z"], 0.0),
        "Provera jedine legalne kombinacije nije prosla.",
    )

    # Normalizovan lanac bez legalne rastuce putanje.
    impossible_initial = np.zeros(5)
    impossible_initial[4] = 1.0
    impossible_transitions = np.zeros((2, 5, 5))
    impossible_transitions[0, 4, 0] = 1.0
    impossible_transitions[1, 0, 1] = 1.0

    try:
        chain_distribution(
            impossible_initial, impossible_transitions
        )
    except DistributionError:
        pass
    else:
        raise RuntimeError("Nulta legalna masa nije prijavljena.")

    # Hronologija preseka i povratne mape.
    shifts = np.array([
        0, 3, 1, 4, 0, 2, 4, 1,
        3, 0, 4, 2, 1, 4, 0, 3,
    ])
    fixture = (
        np.array([0, 5, 10, 15, 20, 25, 30])[None, :]
        + shifts[:, None]
    )

    cutoff = 10
    context = build_context(fixture, cutoff, 3, 32)

    changed = fixture.copy()
    changed[cutoff + 1:] = changed[cutoff + 1:][::-1]
    changed_context = build_context(changed, cutoff, 3, 32)

    require(
        np.allclose(
            context["section"]["direction"],
            changed_context["section"]["direction"],
        )
        and np.allclose(
            context["raw"][:cutoff + 1],
            changed_context["raw"][:cutoff + 1],
        )
        and np.allclose(
            context["states"][:cutoff + 1],
            changed_context["states"][:cutoff + 1],
        ),
        "Buduci podaci uticu na ranije kontekste.",
    )

    first_walk = walks[WALK_TIMES[0]]
    prediction = predict(context, first_walk, cutoff + 1)
    changed_prediction = predict(
        changed_context, first_walk, cutoff + 1
    )

    require(
        np.allclose(
            prediction["distribution"]["marginals"],
            changed_prediction["distribution"]["marginals"],
            atol=2e-10,
        ),
        "Buduci podaci uticu na raniju predikciju.",
    )

    print("Ugradjene Qiskit, presek, hronologija i DP provere su prosle.")


def longest_consecutive(values):
    longest = 1
    current = 1

    for difference in np.diff(values):
        current = current + 1 if difference == 1 else 1
        longest = max(longest, current)

    return longest


def process_dataset(dataset, walks):
    data = dataset["data"]
    path = dataset["path"]
    transitions = len(data) - 1

    train_end = int(TRAIN_FRACTION * transitions)
    validation_end = int(VALIDATION_END_FRACTION * transitions)

    require(
        0 < train_end < validation_end < transitions,
        "Nedovoljno podataka za hronolosku podelu.",
    )

    print()
    print("=" * 72)
    print(path.name)
    print(f"Stvarni broj kola: {len(data)}")
    print(f"SHA256: {dataset['digest']}")
    print(f"Trening prelazi: 1..{train_end}")
    print(f"Validacioni prelazi: {train_end + 1}..{validation_end}")
    print(f"Test prelazi: {validation_end + 1}..{transitions}")

    best = None

    for window in WINDOWS:
        for neighbors in NEIGHBOR_SCALES:
            print()
            print(
                f"Priprema preseka i povratne mape: "
                f"prozor={window}, skala={neighbors}",
                flush=True,
            )
            context = build_context(
                data, train_end, window, neighbors
            )
            print(
                f"Zatvorenih trening povrataka: "
                f"{context['training_return_pairs']}"
            )

            if context["training_return_pairs"] == 0:
                print("Napomena: trening nema zatvoren povratak kroz presek.")

            for walk_time in WALK_TIMES:
                label = (
                    f"VALIDACIJA prozor={window}, "
                    f"skala={neighbors}, tau={walk_time}"
                )

                try:
                    metrics = evaluate(
                        context,
                        walks[walk_time],
                        data,
                        range(train_end + 1, validation_end + 1),
                        label,
                    )
                except DistributionError as exc:
                    print(f"Kandidat odbijen: {exc}")
                    continue

                print_metrics(label, metrics)

                if np.isfinite(metrics["nll"]) and (
                    best is None or metrics["nll"] < best["nll"]
                ):
                    best = {
                        "window": window,
                        "neighbors": neighbors,
                        "walk_time": walk_time,
                        "nll": metrics["nll"],
                    }

    require(best is not None, "Nema validnog kandidata.")

    print()
    print(
        f"Izabrano: prozor={best['window']}, "
        f"skala={best['neighbors']}, tau={best['walk_time']}"
    )

    walk = walks[best["walk_time"]]

    # Test koristi nov trening sa ukljucenom validacijom.
    # Test ciljevi ne ulaze u model ili skaliranje.
    test_context = build_context(
        data,
        validation_end,
        best["window"],
        best["neighbors"],
    )
    test_metrics = evaluate(
        test_context,
        walk,
        data,
        range(validation_end + 1, len(data)),
        "ODVOJENI TEST",
    )
    print_metrics("ODVOJENI HRONOLOSKI TEST", test_metrics)

    # Zavrsno ucenje koristi sve poznate prelaze.
    final_context = build_context(
        data,
        transitions,
        best["window"],
        best["neighbors"],
    )
    next_time = len(data)
    prediction = predict(final_context, walk, next_time)
    verify_final_qiskit(final_context, prediction, next_time)

    distribution = prediction["distribution"]
    combination = distribution["path"] + 1

    require(
        len(combination) == K
        and len(set(combination.tolist())) == K
        and np.all((combination >= 1) & (combination <= N)),
        "Nelegalna NEXT kombinacija.",
    )

    info = final_context["return_info"]
    next_row = next_time - 1

    print()
    print(
        f"Zatvorenih povrataka cele istorije: "
        f"{final_context['training_return_pairs']}"
    )
    print(
        f"Povrataka dostupnih za aktuelnu lokalnu mapu: "
        f"{info['pairs'][next_row]}"
    )
    print(
        f"Efektivan broj primera za izlaznu regresiju: "
        f"{prediction['effective']:.6f}"
    )
    print(f"Radijus kvantnog susedstva: {prediction['radius']:.9f}")

    if np.isfinite(info["spread"][next_row]):
        print(
            f"Lokalno rasipanje povratne mape: "
            f"{info['spread'][next_row]:.9f}"
        )
        print(
            f"Efektivan broj prethodnih povrataka: "
            f"{info['effective'][next_row]:.6f}"
        )

    print(f"Log legalne mase: {distribution['log_z']:.9f}")
    print(
        f"Log uslovne mase NEXT: "
        f"{distribution['log_mode_probability']:.9f}"
    )

    # Provera lokalne strukture na drugoj skali.
    alternative = predict(
        final_context,
        walk,
        next_time,
        neighbors=2 * best["neighbors"],
    )
    scale_change = float(np.max(np.abs(
        distribution["marginals"]
        - alternative["distribution"]["marginals"]
    )))
    print(
        f"Promena marginala pri izlaznoj skali 2k: "
        f"{scale_change:.9e}"
    )

    if np.array_equal(combination, data[-1] + 1):
        print("Upozorenje: NEXT ponavlja poslednje kolo.")

    run = longest_consecutive(combination)

    if run >= 5:
        print(f"Upozorenje: niz od {run} uzastopnih brojeva.")

    span = int(combination[-1] - combination[0])
    historical_spans = data[:, -1] - data[:, 0]
    percentile = 100.0 * float(np.mean(historical_spans <= span))

    print(f"Raspon NEXT kombinacije: {span}")
    print(f"Udeo istorijskih raspona <= NEXT raspon: {percentile:.3f}%")

    if percentile <= 1.0:
        print("Upozorenje: izrazito grupisana NEXT kombinacija.")

    return {
        "name": path.name,
        "digest": dataset["digest"],
        "next": combination,
        "initial": prediction["initial"],
        "transitions": prediction["transitions"],
        "marginals": distribution["marginals"],
    }


def main():
    print("Quantum Poincare Return Map V1")
    print(f"Qiskit: {qiskit.__version__}")
    print("Orijentisani presek i lokalna povratna mapa.")
    print("Kvantna slicnost sestokubitnih istorijskih stanja.")
    print("Qiskit izlazni operator i Bornove raspodele.")
    print("Sedam sestokubitnih uslovnih izlaznih registara.")
    print("Dvoregistarska provera koristi 12 kubita.")
    print("Lokalna regresija i DP su klasicni delovi.")
    print("Svaki trening prelaz ima pozitivnu lokalnu tezinu.")
    print("Svaki CSV se obradjuje zasebno.")
    print()

    walks = {}

    for walk_time in WALK_TIMES:
        print(
            f"Priprema Qiskit operatora tau={walk_time}",
            flush=True,
        )
        walks[walk_time] = make_quantum_walk(walk_time)

    self_checks(walks)

    results = [
        process_dataset(load_csv(path), walks)
        for path in FILES
    ]

    print()
    print("=" * 72)
    print("POREDJENJE NEZAVISNIH REZULTATA")

    for first, second in itertools.combinations(results, 2):
        if first["digest"] == second["digest"]:
            print(
                f"Upozorenje: isti sadrzaj fajlova "
                f"{first['name']} i {second['name']}."
            )

        if np.array_equal(first["next"], second["next"]):
            print()
            print(
                f"Upozorenje: isti NEXT za "
                f"{first['name']} i {second['name']}."
            )
            print(
                "Najveca razlika pocetnih raspodela: "
                f"{np.max(np.abs(first['initial'] - second['initial'])):.9e}"
            )
            print(
                "Najveca razlika uslovnih prelaza: "
                f"{np.max(np.abs(first['transitions'] - second['transitions'])):.9e}"
            )
            print(
                "Najveca razlika legalnih marginala: "
                f"{np.max(np.abs(first['marginals'] - second['marginals'])):.9e}"
            )

    print()
    print("=" * 72)

    for result in results:
        numbers = " ".join(
            str(int(value)) for value in result["next"]
        )
        print(f"{result['name']} NEXT: {numbers}")


if __name__ == "__main__":
    main()



"""
Quantum Poincare Return Map V1
Qiskit: 1.4.4
Orijentisani presek i lokalna povratna mapa.
Kvantna slicnost sestokubitnih istorijskih stanja.
Qiskit izlazni operator i Bornove raspodele.
Sedam sestokubitnih uslovnih izlaznih registara.
Dvoregistarska provera koristi 12 kubita.
Lokalna regresija i DP su klasicni delovi.
Svaki trening prelaz ima pozitivnu lokalnu tezinu.
Svaki CSV se obradjuje zasebno.

Priprema Qiskit operatora tau=0.5
Priprema Qiskit operatora tau=1.5
Ugradjene Qiskit, presek, hronologija i DP provere su prosle.

========================================================================
loto7_4716_k89.csv
Stvarni broj kola: 4716
SHA256: deb02ea741f500e3270402730ed89565f1c9b4618d53e5a08fea2b7c2c2af31a
Trening prelazi: 1..3300
Validacioni prelazi: 3301..4007
Test prelazi: 4008..4715

Priprema preseka i povratne mape: prozor=1, skala=32
Zatvorenih trening povrataka: 823
VALIDACIJA prozor=1, skala=32, tau=0.5: obradjeno 100/707
VALIDACIJA prozor=1, skala=32, tau=0.5: obradjeno 200/707
VALIDACIJA prozor=1, skala=32, tau=0.5: obradjeno 300/707
VALIDACIJA prozor=1, skala=32, tau=0.5: obradjeno 400/707
VALIDACIJA prozor=1, skala=32, tau=0.5: obradjeno 500/707
VALIDACIJA prozor=1, skala=32, tau=0.5: obradjeno 600/707
VALIDACIJA prozor=1, skala=32, tau=0.5: obradjeno 700/707
VALIDACIJA prozor=1, skala=32, tau=0.5: obradjeno 707/707
VALIDACIJA prozor=1, skala=32, tau=0.5
Srednji uslovni NLL: 16.733660327
Srednji broj pogodaka: 1.216407
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [157, 301, 193, 51, 5, 0, 0, 0]
VALIDACIJA prozor=1, skala=32, tau=1.5: obradjeno 100/707
VALIDACIJA prozor=1, skala=32, tau=1.5: obradjeno 200/707
VALIDACIJA prozor=1, skala=32, tau=1.5: obradjeno 300/707
VALIDACIJA prozor=1, skala=32, tau=1.5: obradjeno 400/707
VALIDACIJA prozor=1, skala=32, tau=1.5: obradjeno 500/707
VALIDACIJA prozor=1, skala=32, tau=1.5: obradjeno 600/707
VALIDACIJA prozor=1, skala=32, tau=1.5: obradjeno 700/707
VALIDACIJA prozor=1, skala=32, tau=1.5: obradjeno 707/707
VALIDACIJA prozor=1, skala=32, tau=1.5
Srednji uslovni NLL: 16.770592681
Srednji broj pogodaka: 1.239038
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [151, 300, 199, 50, 7, 0, 0, 0]

Priprema preseka i povratne mape: prozor=1, skala=128
Zatvorenih trening povrataka: 823
VALIDACIJA prozor=1, skala=128, tau=0.5: obradjeno 100/707
VALIDACIJA prozor=1, skala=128, tau=0.5: obradjeno 200/707
VALIDACIJA prozor=1, skala=128, tau=0.5: obradjeno 300/707
VALIDACIJA prozor=1, skala=128, tau=0.5: obradjeno 400/707
VALIDACIJA prozor=1, skala=128, tau=0.5: obradjeno 500/707
VALIDACIJA prozor=1, skala=128, tau=0.5: obradjeno 600/707
VALIDACIJA prozor=1, skala=128, tau=0.5: obradjeno 700/707
VALIDACIJA prozor=1, skala=128, tau=0.5: obradjeno 707/707
VALIDACIJA prozor=1, skala=128, tau=0.5
Srednji uslovni NLL: 16.726005028
Srednji broj pogodaka: 1.226308
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [156, 293, 204, 50, 4, 0, 0, 0]
VALIDACIJA prozor=1, skala=128, tau=1.5: obradjeno 100/707
VALIDACIJA prozor=1, skala=128, tau=1.5: obradjeno 200/707
VALIDACIJA prozor=1, skala=128, tau=1.5: obradjeno 300/707
VALIDACIJA prozor=1, skala=128, tau=1.5: obradjeno 400/707
VALIDACIJA prozor=1, skala=128, tau=1.5: obradjeno 500/707
VALIDACIJA prozor=1, skala=128, tau=1.5: obradjeno 600/707
VALIDACIJA prozor=1, skala=128, tau=1.5: obradjeno 700/707
VALIDACIJA prozor=1, skala=128, tau=1.5: obradjeno 707/707
VALIDACIJA prozor=1, skala=128, tau=1.5
Srednji uslovni NLL: 16.768379010
Srednji broj pogodaka: 1.254597
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [144, 308, 193, 55, 7, 0, 0, 0]

Priprema preseka i povratne mape: prozor=3, skala=32
Zatvorenih trening povrataka: 823
VALIDACIJA prozor=3, skala=32, tau=0.5: obradjeno 100/707
VALIDACIJA prozor=3, skala=32, tau=0.5: obradjeno 200/707
VALIDACIJA prozor=3, skala=32, tau=0.5: obradjeno 300/707
VALIDACIJA prozor=3, skala=32, tau=0.5: obradjeno 400/707
VALIDACIJA prozor=3, skala=32, tau=0.5: obradjeno 500/707
VALIDACIJA prozor=3, skala=32, tau=0.5: obradjeno 600/707
VALIDACIJA prozor=3, skala=32, tau=0.5: obradjeno 700/707
VALIDACIJA prozor=3, skala=32, tau=0.5: obradjeno 707/707
VALIDACIJA prozor=3, skala=32, tau=0.5
Srednji uslovni NLL: 16.715744123
Srednji broj pogodaka: 1.195191
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [168, 293, 189, 54, 3, 0, 0, 0]
VALIDACIJA prozor=3, skala=32, tau=1.5: obradjeno 100/707
VALIDACIJA prozor=3, skala=32, tau=1.5: obradjeno 200/707
VALIDACIJA prozor=3, skala=32, tau=1.5: obradjeno 300/707
VALIDACIJA prozor=3, skala=32, tau=1.5: obradjeno 400/707
VALIDACIJA prozor=3, skala=32, tau=1.5: obradjeno 500/707
VALIDACIJA prozor=3, skala=32, tau=1.5: obradjeno 600/707
VALIDACIJA prozor=3, skala=32, tau=1.5: obradjeno 700/707
VALIDACIJA prozor=3, skala=32, tau=1.5: obradjeno 707/707
VALIDACIJA prozor=3, skala=32, tau=1.5
Srednji uslovni NLL: 16.766635689
Srednji broj pogodaka: 1.280057
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [135, 310, 200, 54, 7, 1, 0, 0]

Priprema preseka i povratne mape: prozor=3, skala=128
Zatvorenih trening povrataka: 823
VALIDACIJA prozor=3, skala=128, tau=0.5: obradjeno 100/707
VALIDACIJA prozor=3, skala=128, tau=0.5: obradjeno 200/707
VALIDACIJA prozor=3, skala=128, tau=0.5: obradjeno 300/707
VALIDACIJA prozor=3, skala=128, tau=0.5: obradjeno 400/707
VALIDACIJA prozor=3, skala=128, tau=0.5: obradjeno 500/707
VALIDACIJA prozor=3, skala=128, tau=0.5: obradjeno 600/707
VALIDACIJA prozor=3, skala=128, tau=0.5: obradjeno 700/707
VALIDACIJA prozor=3, skala=128, tau=0.5: obradjeno 707/707
VALIDACIJA prozor=3, skala=128, tau=0.5
Srednji uslovni NLL: 16.713727491
Srednji broj pogodaka: 1.189533
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [165, 301, 187, 50, 4, 0, 0, 0]
VALIDACIJA prozor=3, skala=128, tau=1.5: obradjeno 100/707
VALIDACIJA prozor=3, skala=128, tau=1.5: obradjeno 200/707
VALIDACIJA prozor=3, skala=128, tau=1.5: obradjeno 300/707
VALIDACIJA prozor=3, skala=128, tau=1.5: obradjeno 400/707
VALIDACIJA prozor=3, skala=128, tau=1.5: obradjeno 500/707
VALIDACIJA prozor=3, skala=128, tau=1.5: obradjeno 600/707
VALIDACIJA prozor=3, skala=128, tau=1.5: obradjeno 700/707
VALIDACIJA prozor=3, skala=128, tau=1.5: obradjeno 707/707
VALIDACIJA prozor=3, skala=128, tau=1.5
Srednji uslovni NLL: 16.766001680
Srednji broj pogodaka: 1.263083
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [140, 311, 195, 54, 5, 2, 0, 0]

Izabrano: prozor=3, skala=128, tau=0.5
ODVOJENI TEST: obradjeno 100/708
ODVOJENI TEST: obradjeno 200/708
ODVOJENI TEST: obradjeno 300/708
ODVOJENI TEST: obradjeno 400/708
ODVOJENI TEST: obradjeno 500/708
ODVOJENI TEST: obradjeno 600/708
ODVOJENI TEST: obradjeno 700/708
ODVOJENI TEST: obradjeno 708/708
ODVOJENI HRONOLOSKI TEST
Srednji uslovni NLL: 16.666118722
Srednji broj pogodaka: 1.329096
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [123, 307, 207, 64, 7, 0, 0, 0]
Zavrsne Qiskit provere preklopa i sedam uslovnih grana su prosle.

Zatvorenih povrataka cele istorije: 1167
Povrataka dostupnih za aktuelnu lokalnu mapu: 1167
Efektivan broj primera za izlaznu regresiju: 4503.123565
Radijus kvantnog susedstva: 0.768852095
Lokalno rasipanje povratne mape: 1.666743576
Efektivan broj prethodnih povrataka: 326.475137
Log legalne mase: -0.395825026
Log uslovne mase NEXT: -13.299544488
Promena marginala pri izlaznoj skali 2k: 5.095267749e-05
Raspon NEXT kombinacije: 31
Udeo istorijskih raspona <= NEXT raspon: 56.255%

========================================================================
loto7_4716_k89_loto_2980.csv
Stvarni broj kola: 2979
SHA256: f1cd81cd04660d50d9058675924da34346fc006ce8dabd84d52266c66afe17d1
Trening prelazi: 1..2084
Validacioni prelazi: 2085..2531
Test prelazi: 2532..2978

Priprema preseka i povratne mape: prozor=1, skala=32
Zatvorenih trening povrataka: 523
VALIDACIJA prozor=1, skala=32, tau=0.5: obradjeno 100/447
VALIDACIJA prozor=1, skala=32, tau=0.5: obradjeno 200/447
VALIDACIJA prozor=1, skala=32, tau=0.5: obradjeno 300/447
VALIDACIJA prozor=1, skala=32, tau=0.5: obradjeno 400/447
VALIDACIJA prozor=1, skala=32, tau=0.5: obradjeno 447/447
VALIDACIJA prozor=1, skala=32, tau=0.5
Srednji uslovni NLL: 16.751476380
Srednji broj pogodaka: 1.217002
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [115, 163, 134, 27, 8, 0, 0, 0]
VALIDACIJA prozor=1, skala=32, tau=1.5: obradjeno 100/447
VALIDACIJA prozor=1, skala=32, tau=1.5: obradjeno 200/447
VALIDACIJA prozor=1, skala=32, tau=1.5: obradjeno 300/447
VALIDACIJA prozor=1, skala=32, tau=1.5: obradjeno 400/447
VALIDACIJA prozor=1, skala=32, tau=1.5: obradjeno 447/447
VALIDACIJA prozor=1, skala=32, tau=1.5
Srednji uslovni NLL: 16.757088381
Srednji broj pogodaka: 1.261745
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [103, 174, 126, 38, 6, 0, 0, 0]

Priprema preseka i povratne mape: prozor=1, skala=128
Zatvorenih trening povrataka: 523
VALIDACIJA prozor=1, skala=128, tau=0.5: obradjeno 100/447
VALIDACIJA prozor=1, skala=128, tau=0.5: obradjeno 200/447
VALIDACIJA prozor=1, skala=128, tau=0.5: obradjeno 300/447
VALIDACIJA prozor=1, skala=128, tau=0.5: obradjeno 400/447
VALIDACIJA prozor=1, skala=128, tau=0.5: obradjeno 447/447
VALIDACIJA prozor=1, skala=128, tau=0.5
Srednji uslovni NLL: 16.744086987
Srednji broj pogodaka: 1.225951
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [116, 159, 135, 29, 8, 0, 0, 0]
VALIDACIJA prozor=1, skala=128, tau=1.5: obradjeno 100/447
VALIDACIJA prozor=1, skala=128, tau=1.5: obradjeno 200/447
VALIDACIJA prozor=1, skala=128, tau=1.5: obradjeno 300/447
VALIDACIJA prozor=1, skala=128, tau=1.5: obradjeno 400/447
VALIDACIJA prozor=1, skala=128, tau=1.5: obradjeno 447/447
VALIDACIJA prozor=1, skala=128, tau=1.5
Srednji uslovni NLL: 16.756466185
Srednji broj pogodaka: 1.259508
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [105, 176, 118, 41, 7, 0, 0, 0]

Priprema preseka i povratne mape: prozor=3, skala=32
Zatvorenih trening povrataka: 523
VALIDACIJA prozor=3, skala=32, tau=0.5: obradjeno 100/447
VALIDACIJA prozor=3, skala=32, tau=0.5: obradjeno 200/447
VALIDACIJA prozor=3, skala=32, tau=0.5: obradjeno 300/447
VALIDACIJA prozor=3, skala=32, tau=0.5: obradjeno 400/447
VALIDACIJA prozor=3, skala=32, tau=0.5: obradjeno 447/447
VALIDACIJA prozor=3, skala=32, tau=0.5
Srednji uslovni NLL: 16.732163450
Srednji broj pogodaka: 1.214765
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [114, 170, 122, 35, 6, 0, 0, 0]
VALIDACIJA prozor=3, skala=32, tau=1.5: obradjeno 100/447
VALIDACIJA prozor=3, skala=32, tau=1.5: obradjeno 200/447
VALIDACIJA prozor=3, skala=32, tau=1.5: obradjeno 300/447
VALIDACIJA prozor=3, skala=32, tau=1.5: obradjeno 400/447
VALIDACIJA prozor=3, skala=32, tau=1.5: obradjeno 447/447
VALIDACIJA prozor=3, skala=32, tau=1.5
Srednji uslovni NLL: 16.754456561
Srednji broj pogodaka: 1.221477
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [110, 179, 118, 30, 9, 1, 0, 0]

Priprema preseka i povratne mape: prozor=3, skala=128
Zatvorenih trening povrataka: 523
VALIDACIJA prozor=3, skala=128, tau=0.5: obradjeno 100/447
VALIDACIJA prozor=3, skala=128, tau=0.5: obradjeno 200/447
VALIDACIJA prozor=3, skala=128, tau=0.5: obradjeno 300/447
VALIDACIJA prozor=3, skala=128, tau=0.5: obradjeno 400/447
VALIDACIJA prozor=3, skala=128, tau=0.5: obradjeno 447/447
VALIDACIJA prozor=3, skala=128, tau=0.5
Srednji uslovni NLL: 16.730790611
Srednji broj pogodaka: 1.228188
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [106, 179, 121, 36, 5, 0, 0, 0]
VALIDACIJA prozor=3, skala=128, tau=1.5: obradjeno 100/447
VALIDACIJA prozor=3, skala=128, tau=1.5: obradjeno 200/447
VALIDACIJA prozor=3, skala=128, tau=1.5: obradjeno 300/447
VALIDACIJA prozor=3, skala=128, tau=1.5: obradjeno 400/447
VALIDACIJA prozor=3, skala=128, tau=1.5: obradjeno 447/447
VALIDACIJA prozor=3, skala=128, tau=1.5
Srednji uslovni NLL: 16.754349020
Srednji broj pogodaka: 1.228188
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [108, 179, 119, 32, 9, 0, 0, 0]

Izabrano: prozor=3, skala=128, tau=0.5
ODVOJENI TEST: obradjeno 100/447
ODVOJENI TEST: obradjeno 200/447
ODVOJENI TEST: obradjeno 300/447
ODVOJENI TEST: obradjeno 400/447
ODVOJENI TEST: obradjeno 447/447
ODVOJENI HRONOLOSKI TEST
Srednji uslovni NLL: 16.702495221
Srednji broj pogodaka: 1.241611
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [88, 207, 110, 40, 2, 0, 0, 0]
Zavrsne Qiskit provere preklopa i sedam uslovnih grana su prosle.

Zatvorenih povrataka cele istorije: 745
Povrataka dostupnih za aktuelnu lokalnu mapu: 745
Efektivan broj primera za izlaznu regresiju: 2752.662935
Radijus kvantnog susedstva: 0.723638567
Lokalno rasipanje povratne mape: 1.724898023
Efektivan broj prethodnih povrataka: 223.856011
Log legalne mase: -0.395486555
Log uslovne mase NEXT: -12.745646146
Promena marginala pri izlaznoj skali 2k: 1.076186924e-04
Raspon NEXT kombinacije: 19
Udeo istorijskih raspona <= NEXT raspon: 3.927%

========================================================================
loto7_4716_k89_loto_plus_1736.csv
Stvarni broj kola: 1735
SHA256: 56225ed0f0be7e2d3e30779804d8fa45e96ecfb9d14441eb2a731dd7fff38ae6
Trening prelazi: 1..1213
Validacioni prelazi: 1214..1473
Test prelazi: 1474..1734

Priprema preseka i povratne mape: prozor=1, skala=32
Zatvorenih trening povrataka: 311
VALIDACIJA prozor=1, skala=32, tau=0.5: obradjeno 100/260
VALIDACIJA prozor=1, skala=32, tau=0.5: obradjeno 200/260
VALIDACIJA prozor=1, skala=32, tau=0.5: obradjeno 260/260
VALIDACIJA prozor=1, skala=32, tau=0.5
Srednji uslovni NLL: 16.854646890
Srednji broj pogodaka: 1.326923
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [51, 102, 80, 25, 2, 0, 0, 0]
VALIDACIJA prozor=1, skala=32, tau=1.5: obradjeno 100/260
VALIDACIJA prozor=1, skala=32, tau=1.5: obradjeno 200/260
VALIDACIJA prozor=1, skala=32, tau=1.5: obradjeno 260/260
VALIDACIJA prozor=1, skala=32, tau=1.5
Srednji uslovni NLL: 16.755283125
Srednji broj pogodaka: 1.357692
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [46, 109, 77, 23, 4, 1, 0, 0]

Priprema preseka i povratne mape: prozor=1, skala=128
Zatvorenih trening povrataka: 311
VALIDACIJA prozor=1, skala=128, tau=0.5: obradjeno 100/260
VALIDACIJA prozor=1, skala=128, tau=0.5: obradjeno 200/260
VALIDACIJA prozor=1, skala=128, tau=0.5: obradjeno 260/260
VALIDACIJA prozor=1, skala=128, tau=0.5
Srednji uslovni NLL: 16.843851082
Srednji broj pogodaka: 1.365385
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [49, 99, 83, 26, 3, 0, 0, 0]
VALIDACIJA prozor=1, skala=128, tau=1.5: obradjeno 100/260
VALIDACIJA prozor=1, skala=128, tau=1.5: obradjeno 200/260
VALIDACIJA prozor=1, skala=128, tau=1.5: obradjeno 260/260
VALIDACIJA prozor=1, skala=128, tau=1.5
Srednji uslovni NLL: 16.754431136
Srednji broj pogodaka: 1.319231
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [50, 111, 72, 21, 5, 1, 0, 0]

Priprema preseka i povratne mape: prozor=3, skala=32
Zatvorenih trening povrataka: 311
VALIDACIJA prozor=3, skala=32, tau=0.5: obradjeno 100/260
VALIDACIJA prozor=3, skala=32, tau=0.5: obradjeno 200/260
VALIDACIJA prozor=3, skala=32, tau=0.5: obradjeno 260/260
VALIDACIJA prozor=3, skala=32, tau=0.5
Srednji uslovni NLL: 16.836312002
Srednji broj pogodaka: 1.350000
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [51, 98, 83, 25, 3, 0, 0, 0]
VALIDACIJA prozor=3, skala=32, tau=1.5: obradjeno 100/260
VALIDACIJA prozor=3, skala=32, tau=1.5: obradjeno 200/260
VALIDACIJA prozor=3, skala=32, tau=1.5: obradjeno 260/260
VALIDACIJA prozor=3, skala=32, tau=1.5
Srednji uslovni NLL: 16.748597034
Srednji broj pogodaka: 1.250000
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [53, 114, 73, 15, 5, 0, 0, 0]

Priprema preseka i povratne mape: prozor=3, skala=128
Zatvorenih trening povrataka: 311
VALIDACIJA prozor=3, skala=128, tau=0.5: obradjeno 100/260
VALIDACIJA prozor=3, skala=128, tau=0.5: obradjeno 200/260
VALIDACIJA prozor=3, skala=128, tau=0.5: obradjeno 260/260
VALIDACIJA prozor=3, skala=128, tau=0.5
Srednji uslovni NLL: 16.833674364
Srednji broj pogodaka: 1.361538
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [50, 98, 83, 26, 3, 0, 0, 0]
VALIDACIJA prozor=3, skala=128, tau=1.5: obradjeno 100/260
VALIDACIJA prozor=3, skala=128, tau=1.5: obradjeno 200/260
VALIDACIJA prozor=3, skala=128, tau=1.5: obradjeno 260/260
VALIDACIJA prozor=3, skala=128, tau=1.5
Srednji uslovni NLL: 16.748217003
Srednji broj pogodaka: 1.253846
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [54, 112, 72, 18, 4, 0, 0, 0]

Izabrano: prozor=3, skala=128, tau=1.5
ODVOJENI TEST: obradjeno 100/261
ODVOJENI TEST: obradjeno 200/261
ODVOJENI TEST: obradjeno 261/261
ODVOJENI HRONOLOSKI TEST
Srednji uslovni NLL: 16.734521727
Srednji broj pogodaka: 1.283525
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [50, 111, 78, 21, 0, 1, 0, 0]
Zavrsne Qiskit provere preklopa i sedam uslovnih grana su prosle.

Zatvorenih povrataka cele istorije: 445
Povrataka dostupnih za aktuelnu lokalnu mapu: 445
Efektivan broj primera za izlaznu regresiju: 1569.511058
Radijus kvantnog susedstva: 0.747314722
Lokalno rasipanje povratne mape: 1.775933866
Efektivan broj prethodnih povrataka: 166.254483
Log legalne mase: -1.401813050
Log uslovne mase NEXT: -14.639166765
Promena marginala pri izlaznoj skali 2k: 1.356386737e-04
Raspon NEXT kombinacije: 38
Udeo istorijskih raspona <= NEXT raspon: 100.000%


========================================================================
POREDJENJE NEZAVISNIH REZULTATA

========================================================================

loto7_4716_k89.csv NEXT: 8 32 34 36 37 38 39

loto7_4716_k89_loto_2980.csv NEXT: 20 31 32 34 36 37 39

loto7_4716_k89_loto_plus_1736.csv NEXT: 1 7 14 20 26 33 39
"""

 

"""
Henri Poincare

lokalna pravilnost i povratne mape

- Rekonstruisati istorijsko stanje iz nekoliko prethodnih kola.
- Tražiti lokalnu pravilnost: da li slični istorijski konteksti imaju slične raspodele narednog kola.
- Učiti lokalnu povratnu mapu kroz regresiju raspodele, pa proveriti da li poboljšanje opstaje na kasnijim, odvojenim kolima.

Poincaréov presek zahteva praćenje prolazaka dinamičkog sistema kroz izabranu površinu. 
Moji CSV-ovi sadrže završne kombinacije, pa bi takav presek bio konstrukcija nad rekonstruisanom istorijom. 

lokalna regresija povratne mape iz sličnih istorijskih stanja

Osnovna ideja jeste provera da li lokalna struktura omogućava predikciju koju globalni model propušta.

Qiskit je osnova kvantnog dela V1: amplitudno kodiranje istorijskih stanja, poređenje njihovih kvantnih stanja i kvantni operator za izlaznu raspodelu. 
Lokalnu regresiju i računanje povratne mape jasno kao klasični delovi hibridnog postupka.

Tri koraka: orijentisani prelazi kroz presek, lokalna povratna mapa i Qiskit raspodela narednog kola. 
Izlaz uključuje zavisnosti između susednih pozicija u kombinaciji. 
Širinu lokalnog susedstva, prozor istorije i kvantni operator biraće hronološka validacija.

Qiskit za kvantnu sličnost istorijskih stanja i izlazni operator nad amplitudama. 
Čuva orijentisani presek, povratnu mapu i lokalna susedstva iz razmatrane oblasti. 
Ideja lokalne pravilnosti

Lokalna regresija i rekonstrukcija preseka predstavljaju klasične delove hibridnog postupka. 
Završna raspodela uključuje zavisnosti između susednih pozicija kombinacije.
"""
