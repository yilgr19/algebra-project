import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SAFE_DICT = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
SAFE_DICT.update({"e": math.e, "pi": math.pi})


def parse_value(expr: str) -> float:
    try:
        return float(eval(expr, {"__builtins__": {}}, SAFE_DICT))
    except Exception as ex:
        raise ValueError(f"No se pudo evaluar '{expr}': {ex}")


def get_dimension() -> int:
    while True:
        d = input("Selecciona dimensión (1, 2 o 3): ")
        if d in ("1", "2", "3"):
            return int(d)
        print("Por favor ingresa 1, 2 o 3.")


def get_num_vectors() -> int:
    while True:
        m = input("¿Cuántos vectores quieres (1, 2 o 3)? ")
        if m in ("1", "2", "3"):
            return int(m)
        print("Por favor ingresa 1, 2 o 3.")


def get_vector(n: int, idx: int) -> np.ndarray:
    while True:
        try:
            coords = input(
                f"Ingresa las {n} coordenadas del vector {idx} separadas por espacios: "
            )
            parts = coords.strip().split()
            if len(parts) != n:
                print(f"Debes ingresar exactamente {n} valores.")
                continue
            return np.array([parse_value(x) for x in parts])
        except ValueError as ve:
            print(ve)


def angle_data(v1: np.ndarray, v2: np.ndarray) -> dict:
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    cos_theta = dot / (norm1 * norm2)
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    theta_rad = math.acos(cos_theta)
    theta_deg = math.degrees(theta_rad)
    return {
        "dot": dot,
        "norm1": norm1,
        "norm2": norm2,
        "rad": theta_rad,
        "deg": theta_deg,
    }


def plot_vectors(vectors: list, dim: int):
    colors = ["r", "b", "g"]
    origin = np.zeros(dim)
    if dim == 1:
        fig, ax = plt.subplots()
        for i, v in enumerate(vectors):
            ax.arrow(
                0,
                0,
                v[0],
                0,
                length_includes_head=True,
                head_width=0.05,
                color=colors[i],
                label=f"v{i+1} = {v[0]:.3f}",
            )
        xs = [v[0] for v in vectors] + [0]
        xmin, xmax = min(xs) * 1.1, max(xs) * 1.1
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-1, 1)
        ax.axhline(0, color="k")
        ax.set_yticks([])
        ax.set_xlabel("X")
        ax.legend()
        plt.title("Vectores en R¹")
        plt.show()
    elif dim == 2:
        fig, ax = plt.subplots()
        xs = [v[0] for v in vectors] + [0]
        ys = [v[1] for v in vectors] + [0]
        for i, v in enumerate(vectors):
            ax.arrow(
                0,
                0,
                v[0],
                v[1],
                length_includes_head=True,
                head_width=0.05,
                color=colors[i],
                label=f"v{i+1} = ({v[0]:.3f}, {v[1]:.3f})",
            )
        xmin, xmax = min(xs) * 1.1, max(xs) * 1.1
        ymin, ymax = min(ys) * 1.1, max(ys) * 1.1
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        plt.title("Vectores en R²")
        plt.show()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        coords = np.vstack(vectors + [origin])
        xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
        for i, v in enumerate(vectors):
            ax.quiver(
                0,
                0,
                0,
                v[0],
                v[1],
                v[2],
                length=1.0,
                normalize=False,
                color=colors[i],
                label=f"v{i+1} = ({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})",
            )
        xmin, xmax = xs.min() * 1.1, xs.max() * 1.1
        ymin, ymax = ys.min() * 1.1, ys.max() * 1.1
        zmin, zmax = zs.min() * 1.1, zs.max() * 1.1
        if zmin == zmax:
            zmin -= 1
            zmax += 1
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.title("Vectores en R³")
        plt.show()


def main():
    print("Cálculo de información de vectores en R^n (1 ≤ n ≤ 3)")
    print("Puedes usar constantes pi, e y funciones trig, exp, log, etc.")
    dim = get_dimension()
    m = get_num_vectors()
    vectors = [get_vector(dim, i + 1) for i in range(m)]

    for i, v in enumerate(vectors, start=1):
        print(f"Vector v{i}: {v}")
        print(f"  Norma: {np.linalg.norm(v):.4f}")

    if m < 2:
        print("No hay pares para comparar ángulos.")
    else:
        for i in range(m):
            for j in range(i + 1, m):
                data = angle_data(vectors[i], vectors[j])
                print(f"\nEntre v{i+1} y v{j+1}:")
                print(f"  Producto escalar: {data['dot']:.4f}")
                print(
                    f"  ||v{i+1}|| = {data['norm1']:.4f}, ||v{j+1}|| = {data['norm2']:.4f}"
                )
                print(f"  Ángulo: {data['rad']:.4f} rad = {data['deg']:.4f}°")

    plot_vectors(vectors, dim)


if __name__ == "__main__":
    main()
