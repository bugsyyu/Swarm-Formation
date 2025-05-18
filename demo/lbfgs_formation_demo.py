import math

import time
try:
    import tkinter as tk
except Exception as e:  # pragma: no cover - tkinter may be unavailable
    tk = None


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


def norm(v):
    return math.sqrt(sum(x*x for x in v))

def sub(a,b):
    return [x-y for x,y in zip(a,b)]

def add(a,b):
    return [x+y for x,y in zip(a,b)]

def mul(a,scalar):
    return [x*scalar for x in a]

class FormationDemo:
    def __init__(self, n_agents=3, n_steps=20):
        self.n_agents = n_agents
        self.n_steps = n_steps
        self.spacing = 2.0
        self.rel_pos = []
        for i in range(n_agents):
            angle = 2*math.pi*i/n_agents
            self.rel_pos.append([math.cos(angle)*self.spacing, math.sin(angle)*self.spacing])
        self.centroid_path = [[i*0.4, 0.0] for i in range(n_steps)]
        self.obstacle = {'center':[4.0,0.0],'radius':1.0}
        self.weights = {'formation':1.0,'centroid':5.0,'smooth':1.0,'obstacle':10.0}

    def pack(self, X):
        flat=[]
        for t in range(self.n_steps):
            for i in range(self.n_agents):
                flat+=X[t][i]
        return flat

    def unpack(self, x):
        X=[];it=iter(x)
        for t in range(self.n_steps):
            agents=[]
            for i in range(self.n_agents):
                agents.append([next(it),next(it)])
            X.append(agents)
        return X

    def cost_grad(self,x):
        X=self.unpack(x)
        f=0.0
        g=[[ [0.0,0.0] for _ in range(self.n_agents)] for _ in range(self.n_steps)]
        desired_dists=[[norm(sub(self.rel_pos[i],self.rel_pos[j])) for j in range(self.n_agents)] for i in range(self.n_agents)]
        for t in range(self.n_steps):
            centroid=[sum(X[t][i][d] for i in range(self.n_agents))/self.n_agents for d in (0,1)]
            dc=sub(centroid,self.centroid_path[t])
            f+=0.5*self.weights['centroid']*norm(dc)**2
            for i in range(self.n_agents):
                g[t][i]=add(g[t][i],mul(dc,self.weights['centroid']/self.n_agents))
            for i in range(self.n_agents):
                for j in range(i+1,self.n_agents):
                    diff=sub(X[t][i],X[t][j])
                    dist=norm(diff)
                    desired=desired_dists[i][j]
                    if dist>1e-6:
                        dd=dist-desired
                        f+=0.5*self.weights['formation']*dd*dd
                        grad=mul(diff,self.weights['formation']*dd/dist)
                        g[t][i]=add(g[t][i],grad)
                        g[t][j]=add(g[t][j],mul(grad,-1))
        for t in range(1,self.n_steps):
            for i in range(self.n_agents):
                diff=sub(X[t][i],X[t-1][i])
                f+=0.5*self.weights['smooth']*norm(diff)**2
                g[t][i]=add(g[t][i],mul(diff,self.weights['smooth']))
                g[t-1][i]=add(g[t-1][i],mul(diff,-self.weights['smooth']))
        center=self.obstacle['center']; radius=self.obstacle['radius']; clearance=0.5
        for t in range(self.n_steps):
            for i in range(self.n_agents):
                diff=sub(X[t][i],center)
                dist=norm(diff)
                thresh=radius+clearance
                if dist<thresh:
                    dd=thresh-dist
                    f+=0.5*self.weights['obstacle']*dd*dd
                    if dist>1e-6:
                        grad=mul(diff,-self.weights['obstacle']*dd/dist)
                        g[t][i]=add(g[t][i],grad)
        g_flat=self.pack(g)
        return f,g_flat

    def run(self):
        X0=[]
        for t in range(self.n_steps):
            agents=[]
            for rel in self.rel_pos:
                agents.append(add(self.centroid_path[t],rel))
            X0.append(agents)
        x0=self.pack(X0)
        f=lambda v:self.cost_grad(v)[0]
        grad=lambda v:self.cost_grad(v)[1]
        opt=LBFGS(f,grad,m=7,max_iter=100,tol=1e-5)
        x_opt,fv=opt.optimize(x0)
        self.X=self.unpack(x_opt)
        print('final cost',fv)


    def show(self, dt=0.1):
        if tk is None:
            raise RuntimeError('Tkinter is not available in this environment')

        X = self.X
        cx, cy = self.obstacle['center']
        radius = self.obstacle['radius']

        scale = 50
        margin = 50
        width = int((max(p[0] for p in self.centroid_path)+3) * scale + 2 * margin)
        height = int((max(abs(p[1]) for p in self.centroid_path)+3) * scale + 2 * margin)

        root = tk.Tk()
        canvas = tk.Canvas(root, width=width, height=height, bg='white')
        canvas.pack()
        canvas.create_oval(
            cx * scale + margin - radius * scale,
            height - (cy * scale + margin) - radius * scale,
            cx * scale + margin + radius * scale,
            height - (cy * scale + margin) + radius * scale,
            fill='red', outline='')

        agents = [canvas.create_oval(0, 0, 0, 0, fill='blue') for _ in range(self.n_agents)]

        def draw(frame):
            if frame >= self.n_steps:
                root.destroy()
                return
            for idx, pos in enumerate(X[frame]):
                x = pos[0] * scale + margin
                y = height - (pos[1] * scale + margin)
                canvas.coords(agents[idx], x-5, y-5, x+5, y+5)
            root.after(int(dt * 1000), lambda: draw(frame + 1))

        draw(0)
        root.mainloop()


if __name__=='__main__':
    demo=FormationDemo(n_agents=3,n_steps=30)
    demo.run()
    demo.show()
