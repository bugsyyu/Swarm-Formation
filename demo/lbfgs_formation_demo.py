import math
import time
from typing import List
import tkinter as tk

class LBFGS:
    def __init__(self, func, grad, m=5, max_iter=100, tol=1e-5):
        self.func = func
        self.grad = grad
        self.m = m
        self.max_iter = max_iter
        self.tol = tol

    def dot(self, a, b):
        return sum(x*y for x, y in zip(a, b))

    def add(self, a, b, scale=1.0):
        return [x + scale*y for x, y in zip(a, b)]

    def optimize(self, x0):
        x = x0[:]
        f = self.func(x)
        g = self.grad(x)
        s_list, y_list, rho_list = [], [], []
        for _ in range(self.max_iter):
            g_norm = math.sqrt(self.dot(g, g))
            if g_norm < self.tol:
                break
            q = g[:]
            alpha = []
            for s, y, rho in reversed(list(zip(s_list, y_list, rho_list))):
                a = rho * self.dot(s, q)
                alpha.append(a)
                q = self.add(q, y, -a)
            if s_list:
                ys = self.dot(y_list[-1], s_list[-1])
                yy = self.dot(y_list[-1], y_list[-1])
                gamma = ys/yy
            else:
                gamma = 1.0
            r = [gamma*qi for qi in q]
            for i, (s, y, rho) in enumerate(zip(s_list, y_list, rho_list)):
                beta = rho * self.dot(y, r)
                r = self.add(r, s, alpha[-(i+1)] - beta)
            p = [-ri for ri in r]
            step = 1.0
            f_val = f
            while True:
                x_new = self.add(x, p, step)
                f_new = self.func(x_new)
                if f_new <= f + 1e-4 * step * self.dot(g, p):
                    break
                step *= 0.5
                if step < 1e-20:
                    break
            g_new = self.grad(x_new)
            s = self.add(x_new, x, -1.0)
            yvec = self.add(g_new, g, -1.0)
            ys = self.dot(yvec, s)
            if ys > 1e-10:
                if len(s_list) == self.m:
                    s_list.pop(0); y_list.pop(0); rho_list.pop(0)
                s_list.append(s)
                y_list.append(yvec)
                rho_list.append(1.0/ys)
            x, f, g = x_new, f_new, g_new
        return x, f


def v_add(a: List[float], b: List[float]) -> List[float]:
    return [a[0] + b[0], a[1] + b[1]]

def v_sub(a: List[float], b: List[float]) -> List[float]:
    return [a[0] - b[0], a[1] - b[1]]

def v_mul(a: List[float], s: float) -> List[float]:
    return [a[0] * s, a[1] * s]

def v_dot(a: List[float], b: List[float]) -> float:
    return a[0] * b[0] + a[1] * b[1]

def v_norm(a: List[float]) -> float:
    return math.sqrt(v_dot(a, a))


class SwarmGraph:
    def __init__(self, desired_pos: List[List[float]]):
        self.set_desired_form(desired_pos)

    def calc_matrices(self, swarm: List[List[float]]):
        N = len(swarm)
        Adj = [[0.0 for _ in range(N)] for _ in range(N)]
        Deg = [0.0 for _ in range(N)]
        for i in range(N):
            for j in range(N):
                dx = swarm[i][0] - swarm[j][0]
                dy = swarm[i][1] - swarm[j][1]
                dist2 = dx * dx + dy * dy
                Adj[i][j] = dist2
                Deg[i] += dist2
        SNL = [[0.0 for _ in range(N)] for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i == j:
                    SNL[i][j] = 1.0
                else:
                    if Deg[i] < 1e-8 or Deg[j] < 1e-8:
                        SNL[i][j] = 0.0
                    else:
                        SNL[i][j] = -Adj[i][j] / math.sqrt(Deg[i]) / math.sqrt(Deg[j])
        return Adj, Deg, SNL

    def set_desired_form(self, swarm_des: List[List[float]]):
        self.nodes_des = [list(p) for p in swarm_des]
        self.Adj_des, self.Deg_des, self.Lhat_des = self.calc_matrices(self.nodes_des)

    def update(self, swarm: List[List[float]]):
        self.nodes = [list(p) for p in swarm]
        self.Adj, self.Deg, self.Lhat = self.calc_matrices(self.nodes)
        N = len(self.nodes)
        self.DLhat = [[self.Lhat[i][j] - self.Lhat_des[i][j] for j in range(N)] for i in range(N)]
        self.grad = [self.calc_grad(i) for i in range(N)]

    def calc_fnorm2(self) -> float:
        return sum(self.DLhat[i][j] * self.DLhat[i][j] for i in range(len(self.DLhat)) for j in range(len(self.DLhat)))

    def calc_grad(self, idx: int):
        N = len(self.nodes)
        Adj = self.Adj
        D = self.Deg
        DLhat = self.DLhat
        dfde = [0.0 for _ in range(N - 1)]
        dedp = [[0.0, 0.0] for _ in range(N - 1)]

        b0 = 0.0
        for k in range(N):
            if D[k] > 1e-8:
                b0 += Adj[idx][k] * DLhat[idx][k] / math.sqrt(D[k])
        if D[idx] > 1e-8:
            b0 = 2 * (D[idx] ** -1.5) * b0
        else:
            b0 = 0.0

        it = 0
        for i in range(N):
            if i == idx:
                continue
            tmp = 0.0
            for k in range(N):
                if D[k] > 1e-8:
                    tmp += Adj[i][k] * DLhat[i][k] / math.sqrt(D[k])
            if D[i] > 1e-8:
                tmp = 2 * (D[i] ** -1.5) * tmp
            else:
                tmp = 0.0
            if D[i] > 1e-8 and D[idx] > 1e-8:
                tmp += b0 + 4 * (-1 / (math.sqrt(D[i]) * math.sqrt(D[idx]))) * DLhat[i][idx]
            dfde[it] = tmp
            dedp[it] = [self.nodes[idx][0] - self.nodes[i][0], self.nodes[idx][1] - self.nodes[i][1]]
            it += 1

        grad = [0.0, 0.0]
        for k in range(N - 1):
            grad[0] += dfde[k] * dedp[k][0]
            grad[1] += dfde[k] * dedp[k][1]
        n = v_norm(grad)
        if n > 1e-7:
            grad = [grad[0] / n, grad[1] / n]
        else:
            grad = [0.0, 0.0]
        return grad

    def get_grad(self, idx: int):
        return self.grad[idx]

class FormationDemo:
    def __init__(self, n_agents: int = 3, n_steps: int = 20):
        self.n_agents = n_agents
        self.n_steps = n_steps
        self.spacing = 2.0
        self.rel_pos = []
        for i in range(n_agents):
            angle = 2 * math.pi * i / n_agents
            self.rel_pos.append([math.cos(angle) * self.spacing, math.sin(angle) * self.spacing])
        self.centroid_path = [[i * 0.4, 0.0] for i in range(n_steps)]
        self.obstacle = {'center': [4.0, 0.0], 'radius': 1.0}
        self.weights = {
            'formation': 1.0,
            'centroid': 5.0,
            'smooth': 1.0,
            'obstacle': 10.0,
            'swarm': 10.0
        }
        self.swarm_graph = SwarmGraph(self.rel_pos)

    def pack(self, X):
        flat = []
        for t in range(self.n_steps):
            for i in range(self.n_agents):
                flat += list(X[t][i])
        return flat

    def unpack(self, x):
        X = []
        it = iter(x)
        for t in range(self.n_steps):
            agents = []
            for _ in range(self.n_agents):
                agents.append([next(it), next(it)])
            X.append(agents)
        return X

    def cost_grad(self,x):
        X = self.unpack(x)
        f = 0.0
        g = [[[0.0, 0.0] for _ in range(self.n_agents)] for _ in range(self.n_steps)]

        desired_graph = self.swarm_graph
        # centroid following and formation
        for t in range(self.n_steps):
            X_t = [p[:] for p in X[t]]
            centroid = [sum(p[d] for p in X_t) / self.n_agents for d in (0, 1)]
            dc = v_sub(centroid, self.centroid_path[t])
            f += 0.5 * self.weights['centroid'] * v_dot(dc, dc)
            for i in range(self.n_agents):
                g[t][i] = v_add(g[t][i], v_mul(dc, self.weights['centroid'] / self.n_agents))

            desired_graph.update(X_t)
            sim_err = desired_graph.calc_fnorm2()
            f += self.weights['formation'] * sim_err
            for i in range(self.n_agents):
                grad = v_mul(desired_graph.get_grad(i), self.weights['formation'])
                g[t][i] = v_add(g[t][i], grad)

        # smoothness
        for t in range(1, self.n_steps):
            for i in range(self.n_agents):
                diff = v_sub(X[t][i], X[t-1][i])
                f += 0.5 * self.weights['smooth'] * v_dot(diff, diff)
                g[t][i] = v_add(g[t][i], v_mul(diff, self.weights['smooth']))
                g[t-1][i] = v_add(g[t-1][i], v_mul(diff, -self.weights['smooth']))

        # obstacle avoidance
        center = self.obstacle['center']
        radius = self.obstacle['radius']
        clearance = 0.5
        thresh = radius + clearance
        for t in range(self.n_steps):
            for i in range(self.n_agents):
                diff = v_sub(X[t][i], center)
                dist = v_norm(diff)
                dist_err = thresh - dist
                if dist_err > 0:
                    f += self.weights['obstacle'] * dist_err ** 3
                    if dist > 1e-6:
                        grad = v_mul(diff, -self.weights['obstacle'] * 3 * dist_err ** 2 / dist)
                        g[t][i] = v_add(g[t][i], grad)

        # inter-agent clearance
        swarm_clearance = 0.5
        for t in range(self.n_steps):
            for i in range(self.n_agents):
                for j in range(i + 1, self.n_agents):
                    diff = v_sub(X[t][i], X[t][j])
                    dist = v_norm(diff)
                    dist_err = swarm_clearance - dist
                    if dist_err > 0:
                        f += self.weights['swarm'] * dist_err ** 3
                        if dist > 1e-6:
                            grad = v_mul(diff, self.weights['swarm'] * 3 * dist_err ** 2 / dist)
                            g[t][i] = v_add(g[t][i], grad)
                            g[t][j] = v_add(g[t][j], v_mul(grad, -1))

        g_flat = self.pack(g)
        return f, g_flat

    def run(self):
        X0 = []
        for t in range(self.n_steps):
            agents = []
            for rel in self.rel_pos:
                agents.append(v_add(self.centroid_path[t], rel))
            X0.append(agents)

        x0 = self.pack(X0)
        f = lambda v: self.cost_grad(v)[0]
        grad = lambda v: self.cost_grad(v)[1]
        opt = LBFGS(f, grad, m=10, max_iter=200, tol=1e-5)
        x_opt, fv = opt.optimize(x0)
        self.X = self.unpack(x_opt)
        print('final cost', fv)

    def show(self):
        X = self.X
        scale = 40
        width = 600
        height = 400
        root = tk.Tk()
        canvas = tk.Canvas(root, width=width, height=height)
        canvas.pack()

        def draw_circle(cx, cy, r, color):
            canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline='')

        ox, oy = self.obstacle['center']
        rad = self.obstacle['radius']

        def draw_step(step=0):
            if step >= self.n_steps:
                root.after(1000, root.destroy)
                return
            canvas.delete("all")
            # draw obstacle
            draw_circle(width / 2 + ox * scale, height / 2 - oy * scale, rad * scale, 'gray')
            for idx, p in enumerate(X[step]):
                x = width / 2 + p[0] * scale
                y = height / 2 - p[1] * scale
                draw_circle(x, y, 5, 'blue')
                canvas.create_text(x + 8, y - 8, text=str(idx))
            canvas.create_text(10, 10, text=f'Time {step + 1}/{self.n_steps}', anchor='nw')
            root.update()
            root.after(100, lambda: draw_step(step + 1))

        draw_step(0)
        root.mainloop()

if __name__=='__main__':
    demo=FormationDemo(n_agents=3,n_steps=30)
    demo.run()
    demo.show()
