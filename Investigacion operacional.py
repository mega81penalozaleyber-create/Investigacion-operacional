"""
Calculadora de programación lineal con interfaz gráfica Tkinter
- Método Simplex (Big-M) con historial de tableau paso a paso
- Método Gráfico (solo 2 variables, sombreado de región factible, valores de Z en cada punto)

Requisitos:
    pip install numpy matplotlib scipy

Uso:
    Ejecuta el script; en la ventana principal elige "Método Simplex" o "Método Gráfico".
    Para Simplex: tras ingresar n y m, se piden los coeficientes mediante cuadros de diálogo.
    Al resolver, se abrirá un visor paso a paso con cada tableau (Anterior / Siguiente).
"""

import sys
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from tkinter.scrolledtext import ScrolledText
from scipy.spatial import ConvexHull

EPS = 1e-9
M_BIG = 1e6

# ------------------ UTILIDADES ------------------
def es_casi_cero(x):
    return abs(x) < EPS

# ------------------ SIMPLEX (Big-M) con historial ------------------
class SimplexBigM:
    def __init__(self, c, A, b, signs, maximize=True):
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.signs = list(signs)
        self.maximize = maximize
        self.history = []
        self._build_tableau()

    def _build_tableau(self):
        m, n = self.A.shape
        var_names = [f"x{j+1}" for j in range(n)]
        slack_count = surplus_count = art_count = 0

        # Asegurar b >= 0
        for i in range(m):
            if self.b[i] < 0:
                self.A[i, :] *= -1
                self.b[i] *= -1
                if self.signs[i] == 'LE': self.signs[i] = 'GE'
                elif self.signs[i] == 'GE': self.signs[i] = 'LE'

        # construir nombres de variables adicionales
        for s in self.signs:
            if s == 'LE':
                var_names.append(f's{slack_count+1}'); slack_count += 1
            elif s == 'GE':
                var_names.append(f'rs{surplus_count+1}'); surplus_count += 1
                var_names.append(f'a{art_count+1}'); art_count += 1
            elif s == 'EQ':
                var_names.append(f'a{art_count+1}'); art_count += 1

        total_vars = n + slack_count + surplus_count + art_count
        A_ext = np.zeros((m, total_vars))
        A_ext[:, :n] = self.A
        slack_i = surplus_i = art_i = 0

        for i, s in enumerate(self.signs):
            if s == 'LE':
                A_ext[i, n + slack_i] = 1
                slack_i += 1
            elif s == 'GE':
                A_ext[i, n + slack_count + surplus_i] = -1
                A_ext[i, n + slack_count + surplus_i + art_i] = 1
                surplus_i += 1
                art_i += 1
            elif s == 'EQ':
                A_ext[i, n + slack_count + surplus_count + art_i] = 1
                art_i += 1

        c = np.array(self.c, dtype=float)
        if not self.maximize:
            c = -c
        c_ext = np.zeros(total_vars)
        c_ext[:n] = c

        art_positions = []
        for j, name in enumerate(var_names):
            if name.startswith('a'):
                art_positions.append(j)
        for j in art_positions:
            c_ext[j] = -M_BIG

        tableau = np.zeros((m + 1, total_vars + 1))
        tableau[:m, :total_vars] = A_ext
        tableau[:m, -1] = self.b
        tableau[-1, :total_vars] = -c_ext

        basic = [-1] * m
        for i in range(m):
            for j in range(total_vars):
                col = tableau[:m, j]
                if es_casi_cero(col.sum() - 1) and es_casi_cero(col[i] - 1) and all(es_casi_cero(col[k]) for k in range(m) if k != i):
                    basic[i] = j
                    break

        self.tableau = tableau
        self.basic = basic
        self.total_vars = total_vars
        self.m = m
        self.n = n
        self.art_positions = art_positions
        self.var_names = var_names
        self.history.append({'step': 'Inicial', 'tableau': self.tableau.copy(), 'basic': self.basic.copy()})

    def _pivot(self, row, col):
        t = self.tableau
        pivot = t[row, col]
        if es_casi_cero(pivot):
            raise ValueError('Pivot casi cero')
        t[row, :] = t[row, :] / pivot
        for i in range(t.shape[0]):
            if i != row:
                t[i, :] = t[i, :] - t[i, col] * t[row, :]
        self.basic[row] = col
        entering_name = self.var_names[col] if col < len(self.var_names) else f'v{col+1}'
        desc = f'Pivot: entra {entering_name} (col {col+1}), fila {row+1}'
        self.history.append({'step': desc, 'tableau': self.tableau.copy(), 'basic': self.basic.copy()})

    def _find_pivot(self):
        obj = self.tableau[-1, :-1]
        entering = None
        for j in range(len(obj)):
            if obj[j] < -EPS:
                entering = j; break
        if entering is None:
            return None, None
        ratios = []
        for i in range(self.m):
            a = self.tableau[i, entering]
            if a > EPS:
                ratios.append((self.tableau[i, -1] / a, i))
        if not ratios:
            return entering, None
        _, row = min(ratios, key=lambda x: (x[0], self.basic[x[1]]))
        return entering, row

    def solve(self, max_iters=1000):
        it = 0
        while it < max_iters:
            entering, row = self._find_pivot()
            if entering is None:
                break
            if row is None:
                return {'status': 'unbounded', 'history': self.history}
            self._pivot(row, entering)
            it += 1

        x = np.zeros(self.total_vars)
        for i in range(self.m):
            if self.basic[i] != -1:
                x[self.basic[i]] = self.tableau[i, -1]
        x_orig = x[:self.n]
        z = self.tableau[-1, -1]
        if not self.maximize:
            z = -z
        for j in self.art_positions:
            if x[j] > EPS:
                return {'status': 'infeasible', 'history': self.history}
        return {'status': 'optimal', 'x': x_orig, 'objective': z, 'tableau': self.tableau.copy(), 'history': self.history}

# ------------------ MÉTODO GRÁFICO ------------------
def metodo_grafico(obj_type, c, constraints):
    a = np.array([con[:2] for con in constraints], dtype=float)
    b = np.array([con[3] for con in constraints], dtype=float)

    candidates = [(0.0, 0.0)]
    for (i, (a1, a2, s1, b1)), (j, (c1, c2, s2, b2)) in itertools.combinations(enumerate(constraints), 2):
        A = np.array([[a1, a2], [c1, c2]])
        if abs(np.linalg.det(A)) < EPS:
            continue
        sol = np.linalg.solve(A, np.array([b1, b2]))
        candidates.append((float(sol[0]), float(sol[1])))

    for (a1, a2, sign, bv) in constraints:
        if abs(a1) > EPS: candidates.append((bv / a1, 0.0))
        if abs(a2) > EPS: candidates.append((0.0, bv / a2))

    feasible = []
    for x1, x2 in candidates:
        if x1 < -1e-6 or x2 < -1e-6: continue
        ok = True
        for (a1, a2, sign, bv) in constraints:
            lhs = a1 * x1 + a2 * x2
            if sign == 'LE' and lhs - bv > 1e-6: ok = False; break
            if sign == 'GE' and bv - lhs > 1e-6: ok = False; break
        if ok: feasible.append((x1, x2))

    if not feasible:
        return None

    vals = [(c[0] * p[0] + c[1] * p[1], p) for p in feasible]
    best = max(vals, key=lambda x: x[0]) if obj_type.startswith('max') else min(vals, key=lambda x: x[0])

    # ---------------- Graficación ----------------
    xs = [p[0] for p in feasible]; ys = [p[1] for p in feasible]

    plt.figure(figsize=(7, 7))
    x_vals = np.linspace(0, max(xs + [10]), 200)
    y_vals = np.linspace(0, max(ys + [10]), 200)

    for (a1, a2, sign, bv) in constraints:
        if abs(a2) > EPS:
            y_line = (bv - a1 * x_vals) / a2
            plt.plot(x_vals, y_line, label=f"{a1}x1+{a2}x2 {sign} {bv}")
        else:
            x_line = np.full_like(y_vals, bv / a1)
            plt.plot(x_line, y_vals, label=f"{a1}x1+{a2}x2 {sign} {bv}")

    feasible_poly = np.array(feasible)
    try:
        hull = ConvexHull(feasible_poly)
        plt.fill(feasible_poly[hull.vertices, 0], feasible_poly[hull.vertices, 1],
                 color='lightgreen', alpha=0.5, label="Región factible")
    except Exception:
        pass

    plt.scatter(xs, ys, c='blue', label="Puntos factibles")
    for val, (x, y) in vals:
        plt.text(x, y, f"Z={val:.1f}", fontsize=9, color="black", ha='right')

    plt.scatter([best[1][0]], [best[1][1]], marker='*', s=250, c='red', label="Óptimo")

    plt.xlabel('x1'); plt.ylabel('x2')
    plt.title(f"Método gráfico ({obj_type}) valor={best[0]:.2f} en {best[1]}")
    plt.legend()
    plt.grid(True)
    plt.show()

    return {'optimal_value': best[0], 'point': best[1]}

# ------------------ VISOR DE TABLEAUX (GUI) ------------------
class TableauViewer(tk.Toplevel):
    def __init__(self, parent, history, var_names):
        super().__init__(parent)
        self.title('Visor de tableaux - Simplex paso a paso')
        self.history = history
        self.var_names = var_names
        self.idx = 0
        self.text = ScrolledText(self, width=120, height=30, font=('Courier', 10))
        self.text.pack(padx=8, pady=8)
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=4)
        tk.Button(btn_frame, text='Anterior', command=self.prev).pack(side='left', padx=4)
        tk.Button(btn_frame, text='Siguiente', command=self.next).pack(side='left', padx=4)
        tk.Button(btn_frame, text='Cerrar', command=self.destroy).pack(side='left', padx=4)
        self.show_step()

    def format_tableau(self, tableau, basic):
        m, totalc = tableau.shape
        totalc -= 1
        header = ['Basic'] + self.var_names + ['RHS']
        lines = []
        lines.append(' | '.join(h.center(12) for h in header))
        lines.append('-' * (14 * len(header)))
        for i in range(m - 1):
            bidx = basic[i]
            bname = self.var_names[bidx] if (bidx != -1 and bidx < len(self.var_names)) else str(bidx)
            row = [f"{tableau[i, j]:.6f}" for j in range(totalc)] + [f"{tableau[i, -1]:.6f}"]
            lines.append(bname.ljust(12) + ' | ' + ' | '.join(x.rjust(12) for x in row))
        lines.append('\nFila objetivo:')
        objrow = [f"{tableau[-1, j]:.6f}" for j in range(totalc)] + [f"{tableau[-1, -1]:.6f}"]
        lines.append('Objective   | ' + ' | '.join(x.rjust(12) for x in objrow))
        return '\n'.join(lines)

    def show_step(self):
        item = self.history[self.idx]
        desc = item.get('step', '')
        tab = item['tableau']
        basic = item['basic']
        s = f"Paso {self.idx+1}/{len(self.history)} - {desc}\n\n"
        s += self.format_tableau(tab, basic)
        self.text.delete('1.0', 'end')
        self.text.insert('1.0', s)
        self.text.see('1.0')

    def prev(self):
        if self.idx > 0:
            self.idx -= 1
            self.show_step()

    def next(self):
        if self.idx < len(self.history) - 1:
            self.idx += 1
            self.show_step()

# ------------------ INTERFAZ TKINTER ------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Calculadora Simplex y Método Gráfico")
        frm = ttk.Frame(root, padding=12)
        frm.pack()
        ttk.Button(frm, text="Método Simplex", command=self.open_simplex, width=30).grid(row=0, column=0, pady=6)
        ttk.Button(frm, text="Método Gráfico", command=self.open_grafico, width=30).grid(row=1, column=0, pady=6)

    def open_simplex(self):
        win = tk.Toplevel(self.root); win.title("Simplex")
        tk.Label(win, text="N variables:").grid(row=0, column=0, sticky='w')
        tk.Label(win, text="M restricciones:").grid(row=1, column=0, sticky='w')
        n_var = tk.Entry(win); m_con = tk.Entry(win)
        n_var.grid(row=0, column=1); m_con.grid(row=1, column=1)

        def next_step():
            try:
                n = int(n_var.get()); m = int(m_con.get())
                if n < 1 or m < 1:
                    raise ValueError()
            except Exception:
                messagebox.showerror("Error", "Ingresa números válidos para n y m")
                return

            coeffs = []
            for j in range(n):
                val = simpledialog.askfloat("Objetivo", f"Coeficiente c{j+1}:", parent=win)
                if val is None: return
                coeffs.append(float(val))

            maximize = messagebox.askyesno("Objetivo", "¿Deseas maximizar (Sí) o minimizar (No)?")
            A = []
            b = []
            signs = []
            for i in range(m):
                row = []
                for j in range(n):
                    v = simpledialog.askfloat("Restricción", f"a{i+1}{j+1}:", parent=win)
                    if v is None: return
                    row.append(float(v))
                s = simpledialog.askstring("Restricción", "Signo (LE/GE/EQ):", parent=win)
                if s is None: return
                s = s.strip().upper()
                if s not in ('LE', 'GE', 'EQ'):
                    messagebox.showerror("Error", "Signo inválido. Usa LE, GE o EQ")
                    return
                rhs = simpledialog.askfloat("Restricción", f"b{i+1}:", parent=win)
                if rhs is None: return
                A.append(row); signs.append(s); b.append(float(rhs))

            try:
                solver = SimplexBigM(coeffs, A, b, signs, maximize)
                res = solver.solve()
            except Exception as e:
                messagebox.showerror("Error", f"Ocurrió un error al construir el problema: {e}")
                return

            hist = res.get('history', [])
            if hist:
                TableauViewer(self.root, hist, solver.var_names)

            status = res.get('status')
            if status == 'optimal':
                xs = res['x']
                sol_text = '\n'.join([f"{solver.var_names[i]} = {xs[i]:.6f}" for i in range(len(xs))])
                sol_text += f"\nZ = {res['objective']:.6f}"
                messagebox.showinfo("Resultado óptimo", sol_text)
            else:
                messagebox.showwarning("Resultado", f"Estado: {status}")

        ttk.Button(win, text="Continuar", command=next_step).grid(row=2, columnspan=2, pady=8)

    def open_grafico(self):
        win = tk.Toplevel(self.root); win.title("Gráfico")
        tk.Label(win, text="Coeficiente c1:").grid(row=0, column=0, sticky='w')
        tk.Label(win, text="Coeficiente c2:").grid(row=1, column=0, sticky='w')
        c1 = tk.Entry(win); c2 = tk.Entry(win)
        c1.grid(row=0, column=1); c2.grid(row=1, column=1)

        def next_step():
            try:
                c = [float(c1.get()), float(c2.get())]
            except:
                messagebox.showerror("Error", "Coeficientes inválidos")
                return

            obj_type = 'max' if messagebox.askyesno("Objetivo", "¿Maximizar (Sí) o Minimizar (No)?") else 'min'

            m = simpledialog.askinteger("Restricciones", "¿Cuántas restricciones?", parent=win)
            if not m or m < 1: return

            cons = []
            for i in range(m):
                a1 = simpledialog.askfloat("Restricción", f"a{i+1}1:", parent=win)
                a2 = simpledialog.askfloat("Restricción", f"a{i+1}2:", parent=win)
                s = simpledialog.askstring("Restricción", "Signo (LE/GE):", parent=win)
                b = simpledialog.askfloat("Restricción", f"b{i+1}:", parent=win)
                if None in (a1, a2, s, b): return
                cons.append((a1, a2, s.strip().upper(), b))

            res = metodo_grafico(obj_type, c, cons)
            if not res:
                messagebox.showwarning("Resultado", "No hay solución factible")
            else:
                messagebox.showinfo("Resultado", f"Valor óptimo = {res['optimal_value']:.3f} en {res['point']}")

        ttk.Button(win, text="Continuar", command=next_step).grid(row=2, columnspan=2, pady=8)

# ------------------ PUNTO DE ENTRADA ------------------
if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()