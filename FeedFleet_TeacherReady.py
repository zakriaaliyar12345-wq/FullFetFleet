import os, csv, queue, threading, heapq, itertools, traceback, datetime
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from torchvision import datasets

# ============ PALETTE — MODERN LIGHT ============
BG="#f4f5f7"; SURFACE="#ffffff"; SURFACE2="#eef0f3"; SURFACE3="#e2e5ea"
BORDER="#d8dce1"; TEXT="#1a1d21"; TEXT_DIM="#4a5560"; MUTED="#7a828c"
ACCENT="#4f46e5"; ACCENT_H="#6366f1"
CYAN="#0891b2"; GREEN="#16a34a"; YELLOW="#ca8a04"; RED="#dc2626"; GOLD="#eab308"
FONT = "Segoe UI"

matplotlib.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": "#fafbfc",
    "axes.edgecolor": BORDER, "axes.labelcolor": TEXT_DIM,
    "axes.titlecolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "grid.color": BORDER,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": FONT, "font.size": 9,
    "legend.facecolor": SURFACE, "legend.edgecolor": BORDER,
})

dev = "cuda" if torch.cuda.is_available() else "cpu"
ALGOS = ["FedFleet", "CFL-like", "HierFAVG", "FedAvg", "FedAsync"]
ICONS = {"FedFleet":"🚀","CFL-like":"🌀","HierFAVG":"🏛","FedAvg":"⚖","FedAsync":"⚡"}
HERE = os.path.dirname(os.path.abspath(__file__))
X = Y = Xt = Yt = WM = EM = None
CM20 = matplotlib.colormaps["tab20"]
CM10 = matplotlib.colormaps["tab10"]

def _hex(c):
    return "#{:02x}{:02x}{:02x}".format(int(c[0]*255), int(c[1]*255), int(c[2]*255))
COL = {n: _hex(CM10(i)) for i, n in enumerate(ALGOS)}

# ============================================================
# FL CORE
# ============================================================
def make_model():
    return nn.Sequential(nn.Linear(784,200), nn.ReLU(),
                         nn.Linear(200,200), nn.ReLU(),
                         nn.Linear(200,10)).to(dev)

def load_data(log):
    global X, Y, Xt, Yt, WM, EM
    if X is not None: return
    log("Loading MNIST...")
    rd = os.path.join(HERE, "data")
    tr = datasets.MNIST(rd, train=True, download=True)
    te = datasets.MNIST(rd, train=False, download=True)
    X = tr.data.float().div(255).view(-1,784).to(dev); Y = tr.targets.to(dev)
    Xt = te.data.float().div(255).view(-1,784).to(dev); Yt = te.targets.to(dev)
    WM = make_model(); EM = make_model()
    log("Dataset ready.")

def partition(C, rng):
    idx = np.argsort(Y.cpu().numpy(), kind="stable")
    n = 2*C; size = len(idx)//n
    sh = [idx[i*size:(i+1)*size] for i in range(n)]
    p = rng.permutation(n)
    return [torch.as_tensor(np.concatenate([sh[p[2*i]], sh[p[2*i+1]]]), device=dev) for i in range(C)]

def make_profile(C, H, rng):
    if H >= 0.24: z = rng.choice([-1.0,1.0], C)
    else:
        z = rng.standard_normal(C); z = (z-z.mean())/(z.std()+1e-9)
    n = np.clip(0.5 + np.sqrt(H)*z, 0, 1)
    return 5 + 20*n, float(n.var())

def make_groups(name, d, E, seed):
    C = len(d); E = min(E, C); o = np.argsort(d, kind="stable")
    if name == "FedFleet": return np.array_split(o, E)
    if name == "CFL-like": return [o[i::E] for i in range(E)]
    if name == "HierFAVG": return np.array_split(np.random.default_rng(seed+2).permutation(C), E)
    if name == "FedAvg": return [np.arange(C)]
    return [np.array([c]) for c in range(C)]

def avg_states(states):
    return {k: torch.stack([s[k] for s in states]).mean(0) for k in states[0]}

def mix(g, w, a): return {k: (1-a)*g[k] + a*w[k] for k in g}

def local_train(gs, ids, cfg):
    WM.load_state_dict(gs)
    opt = torch.optim.SGD(WM.parameters(), lr=cfg["lr"])
    for _ in range(cfg["L"]):
        p = ids[torch.randperm(len(ids), device=dev)]
        for i in range(0, len(p), 50):
            b = p[i:i+50]
            opt.zero_grad(); F.cross_entropy(WM(X[b]), Y[b]).backward(); opt.step()
    return {k: v.detach().clone() for k, v in WM.state_dict().items()}

@torch.no_grad()
def evaluate(st):
    EM.load_state_dict(st)
    return ((EM(Xt).argmax(1) == Yt).float().mean().item()*100)

def run_algo(name, cfg, q, stop):
    C = cfg["C"]; E = min(cfg["E"], C)
    rng = np.random.default_rng(cfg["seed"])
    d, _ = make_profile(C, cfg["H"], rng)
    cidx = partition(C, rng)
    groups = make_groups(name, d, E, cfg["seed"])
    drng = np.random.default_rng(cfg["seed"]+1)
    torch.manual_seed(cfg["seed"])
    drop = cfg["drop"]/100; cloud = cfg["cloud"]; base = cfg["alpha"]
    TMAX = cfg["tmax"]; target = cfg["target"]

    def st(lag):
        if lag < cfg["b"]: return 1.0
        return 1.0/(cfg["a"]*(lag - cfg["b"]) + 1)

    g = {k: v.clone() for k, v in make_model().state_dict().items()}
    t = 0; tt = [None]; last = [0.0]; counter = itertools.count()
    q.put(("start", name, d.tolist(), [list(map(int, gp)) for gp in groups], TMAX))

    def alive(ids): return [int(c) for c in ids if drng.random() >= drop]
    def emit_pt(clk):
        a = evaluate(g); q.put(("pt", name, clk, a, t)); last[0] = a
        if tt[0] is None and a >= target: tt[0] = clk
    def early(): return cfg["stop_at"] and tt[0] is not None

    if name in ("FedFleet", "CFL-like"):
        heap = []; G = len(groups)
        def launch(e, now):
            ids = alive(groups[e])
            dur = (d[ids].max() if ids else d[groups[e]].max()) + cloud
            w = avg_states([local_train(g, cidx[c], cfg) for c in ids]) if ids else None
            heapq.heappush(heap, (now+dur, next(counter), e, t, now, w))
        for e in range(G): launch(e, 0.0)
        while heap and not stop.is_set():
            clk, _, e, tau, start, w = heapq.heappop(heap)
            if clk > TMAX: break
            s = st(t - tau)
            if w is not None:
                g = mix(g, w, min(1.0, base*(1-e/G)*s)); t += 1; emit_pt(clk)
            q.put(("ev", name, e, start, clk, s if w is not None else -1))
            if early(): break
            launch(e, clk)

    elif name == "HierFAVG":
        clk = 0.0
        while clk < TMAX and not stop.is_set():
            start = clk; models = []; durs = []
            for e, grp in enumerate(groups):
                ids = alive(grp); durs.append(d[ids].max() if ids else d[grp].max())
                if ids: models.append(avg_states([local_train(g, cidx[c], cfg) for c in ids]))
                q.put(("ev", name, e, start, start+durs[-1]+cloud, 1.0))
            clk += max(durs) + cloud
            if clk > TMAX: break
            if models: g = avg_states(models); t += 1; emit_pt(clk)
            if early(): break

    elif name == "FedAvg":
        k = min(C, max(5, int(0.1*C))); clk = 0.0
        while clk < TMAX and not stop.is_set():
            ids = rng.choice(C, size=k, replace=False); part = alive(ids)
            dur = (d[part].max() if part else d[ids].max()) + cloud
            if part: g = avg_states([local_train(g, cidx[c], cfg) for c in part])
            q.put(("ev", name, 0, clk, clk+dur, 1.0 if part else -1))
            clk += dur
            if clk > TMAX: break
            if part: t += 1; emit_pt(clk)
            if early(): break

    else:
        bucket = np.zeros(C, int)
        for i, gp in enumerate(np.array_split(np.argsort(d, kind="stable"), E)): bucket[gp] = i
        heap = []
        def launch_c(c, now):
            w = local_train(g, cidx[c], cfg) if drng.random() >= drop else None
            heapq.heappush(heap, (now+dur if False else now+d[c]+cloud, next(counter), c, t, now, w))
        for c in range(C): launch_c(c, 0.0)
        while heap and not stop.is_set():
            clk, _, c, tau, start, w = heapq.heappop(heap)
            if clk > TMAX: break
            s = st(t - tau)
            if w is not None:
                g = mix(g, w, min(1.0, 0.5*base*s)); t += 1
                if t % 3 == 0: emit_pt(clk)
            q.put(("ev", name, int(bucket[c]), start, clk, s if w is not None else -1))
            if early(): break
            launch_c(c, clk)

    q.put(("done", name, tt[0], last[0], t))

def worker(cfg, algos, q, stop):
    try:
        load_data(lambda s: q.put(("log", s)))
        for name in algos:
            if stop.is_set(): break
            q.put(("log", f"Running {name}..."))
            run_algo(name, cfg, q, stop)
    except Exception:
        q.put(("log", traceback.format_exc()))
    q.put(("finished",))

# ============================================================
# STATE
# ============================================================
q = queue.Queue(); stop_ev = threading.Event(); running = [False]
results = {}; summary = {}; events = []; live = {}
cur = {"name":"FedFleet","d":None,"groups":None,"tmax":3000}
_prev_job = [None]

# ============================================================
# ROOT
# ============================================================
root = tk.Tk()
root.title("FedFleet Simulator")
root.geometry("1620x1000"); root.minsize(1300, 780)
root.configure(bg=BG)

# Forward declarations
def schedule_preview(*_): pass
def preview(): pass

style = ttk.Style()
try: style.theme_use("clam")
except Exception: pass
style.configure("Horizontal.TScale", background=SURFACE2, troughcolor=SURFACE2,
                bordercolor=BORDER, lightcolor=ACCENT, darkcolor=ACCENT)
style.configure("TCombobox", fieldbackground=SURFACE2, background=SURFACE2,
                foreground=TEXT, arrowcolor=ACCENT, bordercolor=BORDER)
style.map("TCombobox", fieldbackground=[("readonly", SURFACE2)],
          foreground=[("readonly", TEXT)])
style.configure("Modern.Treeview", background=SURFACE, fieldbackground=SURFACE,
                foreground=TEXT, rowheight=30, font=(FONT, 9), borderwidth=0)
style.configure("Modern.Treeview.Heading", background=SURFACE2,
                foreground=TEXT_DIM, font=(FONT, 9, "bold"), borderwidth=0)
style.map("Modern.Treeview", background=[("selected", ACCENT)],
          foreground=[("selected", "#ffffff")])
style.configure("TNotebook", background=BG, borderwidth=0)
style.configure("TNotebook.Tab", background=SURFACE2, foreground=TEXT_DIM,
                padding=[14, 6], font=(FONT, 9))
style.map("TNotebook.Tab", background=[("selected", SURFACE)],
          foreground=[("selected", ACCENT)])

# ============================================================
# WIDGETS
# ============================================================
class Btn(tk.Frame):
    def __init__(self, parent, text, command=None, primary=False,
                 danger=False, icon="", padx=12, pady=7):
        if danger: bgn, bgh, fgn = SURFACE2, "#fee2e2", RED
        elif primary: bgn, bgh, fgn = ACCENT, ACCENT_H, "#ffffff"
        else: bgn, bgh, fgn = SURFACE2, SURFACE3, TEXT
        super().__init__(parent, bg=bgn, cursor="hand2",
                         highlightthickness=1, highlightbackground=BORDER)
        self.cmd = command; self.bg0 = bgn; self.bg1 = bgh
        label = f"{icon}   {text}".strip() if icon else text
        self.lbl = tk.Label(self, text=label, bg=bgn, fg=fgn,
                            font=(FONT, 9, "bold" if primary else "normal"),
                            anchor="w", padx=padx, pady=pady)
        self.lbl.pack(fill="x")
        for w in (self, self.lbl):
            w.bind("<Button-1>", lambda e: self.cmd and self.cmd())
            w.bind("<Enter>", lambda e: (self.config(bg=self.bg1),
                                          self.lbl.config(bg=self.bg1)))
            w.bind("<Leave>", lambda e: (self.config(bg=self.bg0),
                                          self.lbl.config(bg=self.bg0)))


def card(parent, title=None, pad=12):
    outer = tk.Frame(parent, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER)
    inner = tk.Frame(outer, bg=SURFACE)
    inner.pack(fill="both", expand=True, padx=pad, pady=pad)
    if title:
        tk.Label(inner, text=title, bg=SURFACE, fg=TEXT,
                 font=(FONT, 10, "bold")).pack(anchor="w", pady=(0, 8))
    return outer, inner


class Slider:
    def __init__(self, parent, text, lo, hi, init, step,
                 fmt="{:.0f}", prev=False, unit=""):
        self.var = tk.DoubleVar(value=init); self.step = step
        self.fmt = fmt; self.prev = prev; self.unit = unit
        f = tk.Frame(parent, bg=SURFACE); f.pack(fill="x", pady=3)
        top = tk.Frame(f, bg=SURFACE); top.pack(fill="x")
        tk.Label(top, text=text, bg=SURFACE, fg=TEXT_DIM,
                 font=(FONT, 9)).pack(side="left")
        self.val = tk.Label(top, text=fmt.format(init)+unit, bg=SURFACE,
                            fg=ACCENT, font=(FONT, 9, "bold"))
        self.val.pack(side="right")
        self.scale = ttk.Scale(f, from_=lo, to=hi, variable=self.var,
                               command=self._cb)
        self.scale.pack(fill="x")
    def _cb(self, v):
        value = round(float(v)/self.step)*self.step
        self.var.set(value); self.val.config(text=self.fmt.format(value)+self.unit)
        if self.prev: schedule_preview()
    def get(self): return self.var.get()
    def show(self, v):
        self.var.set(v); self.val.config(text=self.fmt.format(v)+self.unit)

# ============================================================
# LAYOUT
# ============================================================
hdr = tk.Frame(root, bg=BG); hdr.pack(fill="x", padx=18, pady=(14, 10))
logo_c = tk.Canvas(hdr, width=38, height=38, bg=BG, highlightthickness=0)
logo_c.pack(side="left", padx=(0, 10))
logo_c.create_oval(2, 2, 36, 36, fill=ACCENT, outline="")
logo_c.create_text(19, 20, text="F", fill="#ffffff", font=(FONT, 18, "bold"))
ttl = tk.Frame(hdr, bg=BG); ttl.pack(side="left")
tk.Label(ttl, text="FedFleet", bg=BG, fg=TEXT, font=(FONT, 17, "bold")).pack(anchor="w")
tk.Label(ttl, text="Federated Learning · Client Speed Simulator",
         bg=BG, fg=MUTED, font=(FONT, 9)).pack(anchor="w")

status_pill = tk.Frame(hdr, bg=SURFACE2, highlightthickness=1,
                       highlightbackground=BORDER)
status_pill.pack(side="right")
status_dot = tk.Canvas(status_pill, width=10, height=10, bg=SURFACE2,
                       highlightthickness=0)
status_dot.pack(side="left", padx=(10, 6), pady=7)
dot = status_dot.create_oval(2, 2, 8, 8, fill=GREEN, outline="")
status = tk.Label(status_pill, text="Ready", bg=SURFACE2, fg=TEXT, font=(FONT, 9))
status.pack(side="left", padx=(0, 12), pady=7)

body = tk.Frame(root, bg=BG); body.pack(fill="both", expand=True, padx=18, pady=(0, 16))

# --- LEFT SIDEBAR ---
lo = tk.Frame(body, bg=BG, width=340)
lo.pack(side="left", fill="y", padx=(0, 14)); lo.pack_propagate(False)
lc = tk.Canvas(lo, bg=BG, borderwidth=0, highlightthickness=0, width=326)
lsb = ttk.Scrollbar(lo, orient="vertical", command=lc.yview)
lc.configure(yscrollcommand=lsb.set)
lsb.pack(side="right", fill="y"); lc.pack(side="left", fill="both", expand=True)
left = tk.Frame(lc, bg=BG)
lid = lc.create_window((0, 0), window=left, anchor="nw")
left.bind("<Configure>", lambda e: lc.configure(scrollregion=lc.bbox("all")))
lc.bind("<Configure>", lambda e: lc.itemconfigure(lid, width=e.width))
def _mw(e): lc.yview_scroll(int(-1*(e.delta/120)), "units")
lc.bind("<Enter>", lambda e: lc.bind_all("<MouseWheel>", _mw))
lc.bind("<Leave>", lambda e: lc.unbind_all("<MouseWheel>"))

# Settings card
c1o, c1 = card(left, "Settings"); c1o.pack(fill="x", pady=(0, 10))
alg_f = tk.Frame(c1, bg=SURFACE); alg_f.pack(fill="x", pady=(0, 6))
tk.Label(alg_f, text="Algorithm", bg=SURFACE, fg=TEXT_DIM,
         font=(FONT, 9)).pack(anchor="w", pady=(0, 3))
algo_var = tk.StringVar(value="FedFleet")
cb_algo = ttk.Combobox(alg_f, textvariable=algo_var, values=ALGOS,
                       state="readonly", font=(FONT, 9))
cb_algo.pack(fill="x")

vC = Slider(c1, "Phones / clients", 10, 200, 100, 10, prev=True)
vE = Slider(c1, "Edge servers", 2, 20, 10, 1, prev=True)
vH = Slider(c1, "Speed differences", 0, 0.25, 0.125, 0.005, "{:.3f}", prev=True)
vDrop = Slider(c1, "Dropout %", 0, 90, 0, 5)
vL = Slider(c1, "Local epochs", 1, 10, 3, 1)
vLR = Slider(c1, "Learning rate", 0.001, 0.1, 0.01, 0.001, "{:.3f}")
vCloud = Slider(c1, "Cloud delay (s)", 0, 100, 20, 1)
vAlpha = Slider(c1, "Base alpha", 0.1, 1.0, 0.8, 0.05, "{:.2f}")
vA = Slider(c1, "Staleness a", 0, 3, 1.0, 0.1, "{:.1f}")
vB = Slider(c1, "Staleness b", 0, 20, 5, 1)
vT = Slider(c1, "Max time (s)", 200, 30000, 3000, 100)
vTarget = Slider(c1, "Target accuracy %", 50, 99, 90, 1)

seed_f = tk.Frame(c1, bg=SURFACE); seed_f.pack(fill="x", pady=(6, 2))
tk.Label(seed_f, text="Random seed", bg=SURFACE, fg=TEXT_DIM,
         font=(FONT, 9)).pack(anchor="w", pady=(0, 3))
vSeed = tk.IntVar(value=0)
tk.Entry(seed_f, textvariable=vSeed, bg=SURFACE2, fg=TEXT,
         insertbackground=ACCENT, relief="flat", font=(FONT, 9),
         highlightthickness=1, highlightbackground=BORDER,
         highlightcolor=ACCENT).pack(fill="x", ipady=4, ipadx=6)

vStop = tk.BooleanVar(value=False)
sf = tk.Frame(c1, bg=SURFACE); sf.pack(fill="x", pady=(6, 0))
stop_chk = tk.Canvas(sf, width=16, height=16, bg=SURFACE,
                     highlightthickness=0, cursor="hand2")
stop_chk.pack(side="left", pady=3)
def draw_chk():
    stop_chk.delete("all")
    if vStop.get():
        stop_chk.create_rectangle(1, 1, 15, 15, fill=ACCENT, outline=ACCENT, width=0)
        stop_chk.create_line(4, 8, 7, 12, fill="#fff", width=2)
        stop_chk.create_line(7, 12, 12, 4, fill="#fff", width=2)
    else:
        stop_chk.create_rectangle(1, 1, 15, 15, outline=BORDER, width=1)
def toggle_stop():
    vStop.set(not vStop.get()); draw_chk()
stop_chk.bind("<Button-1>", lambda e: toggle_stop())
tk.Label(sf, text="  Stop when target reached", bg=SURFACE, fg=TEXT,
         font=(FONT, 9), cursor="hand2").pack(side="left")
draw_chk()

# Buttons card
c2o, c2 = card(left, "Actions"); c2o.pack(fill="x", pady=(0, 10))
Btn(c2, "Quick demo settings", command=lambda: quick_demo(),
    icon="⚡").pack(fill="x", pady=2)
Btn(c2, "Start selected algorithm",
    command=lambda: start([algo_var.get()]),
    primary=True, icon="▶").pack(fill="x", pady=2)
Btn(c2, "Run ALL 5 (compare)",
    command=lambda: start(ALGOS),
    primary=True, icon="▶▶").pack(fill="x", pady=2)
Btn(c2, "Stop", command=lambda: stop_ev.set(),
    danger=True, icon="■").pack(fill="x", pady=2)
Btn(c2, "Preview groups", command=lambda: preview(),
    icon="🔍").pack(fill="x", pady=2)
tk.Frame(c2, bg=BORDER, height=1).pack(fill="x", pady=6)
Btn(c2, "Open explanation window",
    command=lambda: open_explanation_window(), icon="📖").pack(fill="x", pady=2)
Btn(c2, "Open graph guide window",
    command=lambda: open_guide_window(), icon="📚").pack(fill="x", pady=2)
Btn(c2, "Clear results", command=lambda: clear_all(),
    icon="🧹").pack(fill="x", pady=2)
Btn(c2, "Save results + report",
    command=lambda: save_results(), icon="💾").pack(fill="x", pady=2)

# Progress card
c3o, c3 = card(left, "Progress"); c3o.pack(fill="x")
prog_lbl = tk.Label(c3, text="Idle", bg=SURFACE, fg=MUTED,
                    font=(FONT, 8), anchor="w")
prog_lbl.pack(fill="x", pady=(0, 4))
prog_pct = tk.Label(c3, text="0%", bg=SURFACE, fg=ACCENT,
                    font=(FONT, 12, "bold"), anchor="e")
prog_pct.pack(fill="x")
prog_c = tk.Canvas(c3, height=6, bg=SURFACE2, highlightthickness=0)
prog_c.pack(fill="x", pady=(3, 0))
prog_rect = prog_c.create_rectangle(0, 0, 0, 6, fill=ACCENT, outline="")
prog_val = [0.0]
def set_prog(pct):
    prog_val[0] = max(0.0, min(100.0, pct))
    prog_c.update_idletasks()
    w = prog_c.winfo_width() or 300
    prog_c.coords(prog_rect, 0, 0, w * prog_val[0]/100.0, 6)
    prog_pct.config(text=f"{prog_val[0]:.0f}%")
prog_c.bind("<Configure>", lambda e: set_prog(prog_val[0]))

# ============================================================
# RIGHT — 2x2 GRAPH GRID
# ============================================================
right = tk.Frame(body, bg=BG); right.pack(side="right", fill="both", expand=True)

grid = tk.Frame(right, bg=BG); grid.pack(fill="both", expand=True)
for r in range(2):
    grid.rowconfigure(r, weight=1)
for c in range(2):
    grid.columnconfigure(c, weight=1)

def make_panel(parent, r, c, title):
    fr = tk.Frame(parent, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER)
    fr.grid(row=r, column=c, sticky="nsew", padx=3, pady=3)
    tk.Label(fr, text=title, bg=SURFACE, fg=TEXT,
             font=(FONT, 10, "bold"), anchor="w").pack(fill="x", padx=10, pady=(6, 2))
    fig = Figure(figsize=(6, 4), dpi=90)
    ax = fig.add_subplot(111)
    cv = FigureCanvasTkAgg(fig, master=fr)
    cv.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
    return fig, ax, cv

fig1, ax1, cv1 = make_panel(grid, 0, 0,
    "1) Accuracy vs time — line that hits RED first wins")
fig2, ax2, cv2 = make_panel(grid, 0, 1,
    "2) Phones — high dot = slow, colour = its group")
fig3, ax3, cv3 = make_panel(grid, 1, 0,
    "3) Timeline — each bar = one group finishing")
fig4, ax4, cv4 = make_panel(grid, 1, 1,
    "4) Waiting time — SHORT bars = good")

status_bar = tk.Label(right, text="Preview of groups. Press Start to train.",
                      bg=BG, fg=TEXT_DIM, font=(FONT, 9), anchor="w")
status_bar.pack(fill="x", pady=(8, 0))

# ============================================================
# BOTTOM NOTEBOOK — Explanation / Guide / Results
# ============================================================
bottom = tk.Frame(root, bg=BG); bottom.pack(fill="x", padx=18, pady=(0, 14))

nb = ttk.Notebook(bottom, style="TNotebook"); nb.pack(fill="x")

# Tab: Explanation
exp_f = tk.Frame(nb, bg=SURFACE); nb.add(exp_f, text="  Simple explanation  ")
exp_txt = tk.Text(exp_f, wrap="word", font=(FONT, 10), bg=SURFACE, fg=TEXT,
                  relief="flat", padx=14, pady=10, height=9,
                  highlightthickness=0)
exp_txt.pack(fill="both", expand=True)
exp_txt.tag_configure("h", font=(FONT, 11, "bold"), foreground=ACCENT, spacing1=6)
exp_txt.tag_configure("b", foreground=TEXT)

# Tab: Guide
gd_f = tk.Frame(nb, bg=SURFACE); nb.add(gd_f, text="  How to read the graphs  ")
gd_txt = tk.Text(gd_f, wrap="word", font=(FONT, 10), bg=SURFACE, fg=TEXT,
                 relief="flat", padx=14, pady=10, height=9,
                 highlightthickness=0)
gd_txt.pack(fill="both", expand=True)
gd_txt.tag_configure("h", font=(FONT, 11, "bold"), foreground=CYAN, spacing1=6)
gd_txt.tag_configure("b", foreground=TEXT)

# Tab: Results
res_f = tk.Frame(nb, bg=SURFACE); nb.add(res_f, text="  Results  ")
cols = ("Algorithm", "Time to target (s)", "Final accuracy %", "Updates")
tree = ttk.Treeview(res_f, columns=cols, show="headings", height=5,
                    style="Modern.Treeview")
for c_, w_ in zip(cols, (180, 180, 180, 120)):
    tree.heading(c_, text=c_); tree.column(c_, width=w_, anchor="center")
tree.pack(fill="both", expand=True, padx=6, pady=6)

# Tab: Log
lg_f = tk.Frame(nb, bg=SURFACE); nb.add(lg_f, text="  Log  ")
log_txt = tk.Text(lg_f, wrap="word", font=(FONT, 9), bg=SURFACE, fg=TEXT_DIM,
                  relief="flat", padx=14, pady=10, height=9,
                  highlightthickness=0)
log_txt.pack(fill="both", expand=True)

# ============================================================
# CONFIG
# ============================================================
def get_cfg():
    return {"C": int(round(vC.get())), "E": int(round(vE.get())),
            "H": float(vH.get()), "drop": float(vDrop.get()),
            "L": int(round(vL.get())), "lr": float(vLR.get()),
            "cloud": float(vCloud.get()), "alpha": float(vAlpha.get()),
            "a": float(vA.get()), "b": int(round(vB.get())),
            "tmax": float(vT.get()), "target": float(vTarget.get()),
            "seed": int(vSeed.get()), "stop_at": bool(vStop.get())}

# ============================================================
# DRAWING
# ============================================================
def draw_graph1():
    ax = ax1; ax.clear(); cfg = get_cfg()
    for nm, (xs, ys) in results.items():
        if len(xs) == 0: continue
        ax.plot(xs, ys, label=f"{ICONS.get(nm,'')} {nm}", color=COL[nm],
                linewidth=2, marker="o", markersize=2)
    t = cfg["target"]
    ax.axhline(t, color=RED, linestyle="--", linewidth=1.8, label=f"Target {t:.0f}%")
    ax.axhspan(0, t, alpha=0.06, color=RED)
    ax.axhspan(t, 100, alpha=0.08, color=GREEN)
    ax.set_xlim(0, max(cfg["tmax"], 1)); ax.set_ylim(0, 100)
    ax.set_xlabel("Simulated time (s)"); ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=0.35)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    cv1.draw_idle()

def draw_graph2():
    ax = ax2; ax.clear(); d = cur["d"]; groups = cur["groups"]; name = cur["name"]
    if d is None:
        ax.text(0.5, 0.5, "Press Start or Preview", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color=MUTED)
        cv2.draw_idle(); return
    if name in ("FedAvg", "FedAsync"):
        colors = [MUTED]*len(d)
    else:
        lab = np.zeros(len(d), dtype=int)
        for i, gp in enumerate(groups): lab[gp] = i
        colors = [CM20(i % 20) for i in lab]
    ax.scatter(range(len(d)), d, c=colors, s=35, alpha=0.9,
               edgecolors=TEXT_DIM, linewidths=0.4)
    ax.set_xlabel("Client number"); ax.set_ylabel("Time needed (s)")
    ax.grid(alpha=0.35)
    cv2.draw_idle()

def draw_graph3():
    ax = ax3; ax.clear(); ev = events[-120:]
    ax.set_xlabel("Simulated time (s)"); ax.set_ylabel("Group")
    if not ev:
        ax.text(0.5, 0.5, "No events yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color=MUTED)
        ax.grid(alpha=0.35); cv3.draw_idle(); return
    for (e, s0, s1, s) in ev:
        dur = max(s1 - s0, 0.01)
        if s < 0:
            ax.barh(e, dur, left=s0, height=0.65, color=SURFACE3,
                    edgecolor=BORDER, linewidth=0.4)
        else:
            ax.barh(e, dur, left=s0, height=0.65, color=CM20(e % 20),
                    alpha=(0.4 + 0.6*min(1.0, s)),
                    edgecolor=BORDER, linewidth=0.4)
    ax.grid(alpha=0.35, axis="x")
    cv3.draw_idle()

def idle_values(name, d, groups):
    if name in ("FedFleet", "CFL-like"):
        return [float(np.mean(d[gp].max() - d[gp])) for gp in groups]
    if name == "HierFAVG":
        return [float(np.mean(d.max() - d[gp])) for gp in groups]
    if name == "FedAvg":
        return [float(np.mean(d.max() - d))]
    return [0.0]

def draw_graph4():
    ax = ax4; ax.clear(); d = cur["d"]; groups = cur["groups"]; name = cur["name"]
    if d is None:
        ax.text(0.5, 0.5, "Press Start or Preview", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color=MUTED)
        cv4.draw_idle(); return
    vals = idle_values(name, d, groups)
    bars = ax.bar(range(len(vals)), vals,
                  color=[CM20(i % 20) for i in range(len(vals))],
                  edgecolor=BORDER, linewidth=0.4)
    ax.set_xlabel("Group"); ax.set_ylabel("Avg waiting (s)")
    ax.grid(alpha=0.35, axis="y")
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8, color=TEXT)
    cv4.draw_idle()

def redraw_all():
    draw_graph1(); draw_graph2(); draw_graph3(); draw_graph4()

# ============================================================
# EXPLANATION TEXT
# ============================================================
def refresh_explanation():
    exp_txt.config(state="normal"); exp_txt.delete("1.0", "end")
    cfg = get_cfg()
    name = cur["name"]; d = cur["d"]
    exp_txt.insert("end", "1. What is happening right now?\n", "h")
    exp_txt.insert("end",
        f"Selected algorithm: {ICONS.get(name,'')} {name}\n", "b")
    if d is not None:
        exp_txt.insert("end",
            f"Clients: {len(d)}  ·  Edge servers: {cfg['E']}\n"
            f"Client speed range: {d.min():.1f}s to {d.max():.1f}s "
            f"(spread {d.max()-d.min():.1f}s)\n\n", "b")
    exp_txt.insert("end", "2. How the 4 graphs connect\n", "h")
    exp_txt.insert("end",
        "Graph 2 shows client speed differences → they create waiting.\n"
        "Graph 4 shows how much waiting each group creates.\n"
        "Graph 3 shows the actual training timeline.\n"
        "Graph 1 shows the final outcome: which algorithm reaches the "
        "target accuracy first.\n\n", "b")
    exp_txt.insert("end", "3. What to look for\n", "h")
    exp_txt.insert("end",
        f"• The red dashed line in graph 1 is the target ({cfg['target']:.0f}%).\n"
        "• In graph 2, FedFleet should group similar-height dots together.\n"
        "• In graph 4, shorter bars mean less waiting (better).\n", "b")
    exp_txt.config(state="disabled")

def refresh_guide():
    gd_txt.config(state="normal"); gd_txt.delete("1.0", "end")
    gd_txt.insert("end", "Graph 1 — Accuracy vs Time\n", "h")
    gd_txt.insert("end",
        "X = simulated time. Y = accuracy. Red dashed line = target.\n"
        "The curve that reaches the red line first wins.\n\n", "b")
    gd_txt.insert("end", "Graph 2 — Client Speed\n", "h")
    gd_txt.insert("end",
        "Each dot = one phone. Low dot = fast phone. High dot = slow phone.\n"
        "Same colour = same group. FedFleet groups similar speeds together.\n\n", "b")
    gd_txt.insert("end", "Graph 3 — Timeline\n", "h")
    gd_txt.insert("end",
        "Each horizontal bar = one group's training activity.\n"
        "Long bar = slower group. Faint bar = weak or stale update.\n\n", "b")
    gd_txt.insert("end", "Graph 4 — Waiting Time\n", "h")
    gd_txt.insert("end",
        "Y-axis = average time phones spend waiting for slower phones.\n"
        "Shorter bars = less waiting = more efficient training.\n", "b")
    gd_txt.config(state="disabled")

refresh_explanation(); refresh_guide()

# ============================================================
# INFO WINDOWS
# ============================================================
explanation_window = [None]; guide_window = [None]

def open_explanation_window():
    if explanation_window[0] is not None:
        try:
            if explanation_window[0].winfo_exists():
                explanation_window[0].deiconify()
                explanation_window[0].lift(); return
        except Exception: pass
    win = tk.Toplevel(root); explanation_window[0] = win
    win.title("Explanation"); win.geometry("800x600"); win.configure(bg=BG)
    wrap = tk.Frame(win, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER)
    wrap.pack(fill="both", expand=True, padx=14, pady=14)
    tx = tk.Text(wrap, wrap="word", font=(FONT, 11), bg=SURFACE, fg=TEXT,
                 relief="flat", padx=16, pady=16, highlightthickness=0)
    tx.pack(fill="both", expand=True)
    tx.insert("end", "FedFleet — Simple Explanation\n\n")
    tx.insert("end", "The problem\n")
    tx.insert("end", "Many phones each have private data. Each phone trains a "
                     "local model and sends updates to the server.\n\n")
    tx.insert("end", "Why speed matters\n")
    tx.insert("end", "Fast phones finish quickly but must wait for slow phones.\n\n")
    tx.insert("end", "What FedFleet does\n")
    tx.insert("end", "Groups phones by similar speed to reduce waiting.\n\n")
    tx.insert("end", "Main idea\n")
    tx.insert("end", "Different speeds → grouping → less waiting → faster training.\n")
    tx.config(state="disabled")

def open_guide_window():
    if guide_window[0] is not None:
        try:
            if guide_window[0].winfo_exists():
                guide_window[0].deiconify()
                guide_window[0].lift(); return
        except Exception: pass
    win = tk.Toplevel(root); guide_window[0] = win
    win.title("Reading Guide"); win.geometry("800x600"); win.configure(bg=BG)
    wrap = tk.Frame(win, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER)
    wrap.pack(fill="both", expand=True, padx=14, pady=14)
    tx = tk.Text(wrap, wrap="word", font=(FONT, 11), bg=SURFACE, fg=TEXT,
                 relief="flat", padx=16, pady=16, highlightthickness=0)
    tx.pack(fill="both", expand=True)
    tx.insert("end", "How to read every graph\n\n")
    tx.insert("end", "Graph 1 — Accuracy vs Time\n")
    tx.insert("end", "X = time. Y = accuracy. Red line = target.\n\n")
    tx.insert("end", "Graph 2 — Client Speed\n")
    tx.insert("end", "Each dot = one phone. Low = fast, High = slow.\n\n")
    tx.insert("end", "Graph 3 — Timeline\n")
    tx.insert("end", "Each bar = one group's work. Long bar = slow group.\n\n")
    tx.insert("end", "Graph 4 — Waiting Time\n")
    tx.insert("end", "Shorter bars = less waiting = better.\n")
    tx.config(state="disabled")

# ============================================================
# ACTIONS
# ============================================================
def preview():
    _prev_job[0] = None
    if running[0]: return
    cfg = get_cfg(); rng = np.random.default_rng(cfg["seed"])
    d, hact = make_profile(cfg["C"], cfg["H"], rng)
    name = algo_var.get()
    groups = make_groups(name, d, cfg["E"], cfg["seed"])
    cur.update(name=name, d=d, groups=groups, tmax=cfg["tmax"])
    status.config(text=f"Preview · {name}")
    status_bar.config(text=f"Preview of groups ({name}). "
                            f"Actual speed variation H = {hact:.3f}. "
                            f"Press Start to train.")
    refresh_explanation()
    redraw_all()

def schedule_preview(*_):
    if running[0]: return
    if _prev_job[0]:
        try: root.after_cancel(_prev_job[0])
        except Exception: pass
    _prev_job[0] = root.after(300, preview)

def start(algos):
    if running[0]: return
    cfg = get_cfg(); cur["tmax"] = cfg["tmax"]
    stop_ev.clear(); running[0] = True
    results.clear(); events.clear(); summary.clear(); live.clear()
    for item in tree.get_children(): tree.delete(item)
    log_txt.delete("1.0", "end")
    set_prog(0); status.config(text="Running...")
    status_dot.itemconfig(dot, fill=YELLOW)
    prog_lbl.config(text="Starting...")
    status_bar.config(text="Training...")
    threading.Thread(target=worker, args=(cfg, algos, q, stop_ev),
                     daemon=True).start()

def clear_all():
    results.clear(); events.clear(); summary.clear(); live.clear()
    for item in tree.get_children(): tree.delete(item)
    set_prog(0); status.config(text="Cleared.")
    prog_lbl.config(text="Idle")
    status_dot.itemconfig(dot, fill=GREEN)
    status_bar.config(text="Results cleared.")
    refresh_explanation(); redraw_all()

def quick_demo():
    for var, val in [(vC,50),(vE,5),(vH,0.125),(vDrop,0),
                     (vL,2),(vT,1500),(vTarget,85)]:
        var.show(val)
    if not vStop.get(): toggle_stop()
    schedule_preview()
    status.config(text="Quick demo loaded")

def save_results():
    if not results and not summary:
        messagebox.showinfo("Nothing to save", "Run the simulation first.")
        return
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    png = os.path.join(HERE, f"result_{stamp}.png")
    csvp = os.path.join(HERE, f"table_{stamp}.csv")
    txtp = os.path.join(HERE, f"report_{stamp}.txt")
    fig = Figure(figsize=(12, 7), dpi=150, facecolor=SURFACE)
    ax = fig.add_subplot(111)
    cfg = get_cfg()
    for nm, (xs, ys) in results.items():
        if len(xs) == 0: continue
        ax.plot(xs, ys, label=nm, color=COL[nm], linewidth=2)
    ax.axhline(cfg["target"], color=RED, linestyle="--", linewidth=2,
               label=f"Target {cfg['target']:.0f}%")
    ax.set_xlim(0, max(cfg["tmax"], 1)); ax.set_ylim(0, 100)
    ax.set_xlabel("Simulated time (s)"); ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in tree.get_children(): w.writerow(tree.item(r)["values"])
    with open(txtp, "w", encoding="utf-8") as f:
        f.write("FedFleet Report\n==========\n\nSettings:\n")
        for k, v in get_cfg().items(): f.write(f"{k} = {v}\n")
        f.write("\nResults:\n")
        for name, (tm, acc, upd) in summary.items():
            f.write(f"{name}: time={tm}, accuracy={acc:.2f}, updates={upd}\n")
    status.config(text=f"Saved {os.path.basename(png)}")
    messagebox.showinfo("Saved", "Files saved in the program folder.")

# ============================================================
# POLL
# ============================================================
def poll():
    got = False
    try:
        while True:
            m = q.get_nowait(); kind = m[0]
            if kind == "log":
                log_txt.insert("end", m[1] + "\n")
                log_txt.see("end")
                status.config(text=m[1][:80])
            elif kind == "start":
                _, name, d, groups, tmax = m
                results[name] = ([], [])
                events.clear()
                cur.update(name=name, d=np.array(d),
                           groups=[np.array(gp) for gp in groups], tmax=tmax)
                live[name] = {"hist": []}
                status.config(text=f"Training {name}")
                refresh_explanation()
                got = True
            elif kind == "pt":
                _, name, clk, acc, t = m
                if name not in results: results[name] = ([], [])
                results[name][0].append(clk); results[name][1].append(acc)
                if name not in live: live[name] = {"hist": []}
                live[name]["hist"].append((clk, acc))
                set_prog(100*clk/max(cur["tmax"], 1))
                status.config(text=f"{name} · {clk:.0f}s · acc {acc:.1f}%")
                prog_lbl.config(text=f"{name} · update #{t}")
                status_bar.config(
                    text=f"{name}: simulated time {clk:.0f}s of {cur['tmax']:.0f}s "
                         f"| updates {t} | accuracy {acc:.1f}%")
                got = True
            elif kind == "ev":
                events.append(m[2:])
                if len(events) > 500: del events[:250]
                got = True
            elif kind == "done":
                _, name, tt, acc, t = m
                summary[name] = (tt, acc, t)
                tree.insert("", "end", values=(
                    name,
                    f"{tt:.0f}" if tt is not None else "Not reached",
                    f"{acc:.2f}", t))
                got = True
            elif kind == "finished":
                running[0] = False
                status.config(text="Finished ✓")
                prog_lbl.config(text="Complete"); set_prog(100)
                status_dot.itemconfig(dot, fill=GREEN)
                status_bar.config(text="Training finished. See Results tab.")
                got = True
    except queue.Empty:
        pass
    if got: redraw_all()
    root.after(400, poll)

def on_close():
    stop_ev.set()
    for w in (explanation_window[0], guide_window[0]):
        try:
            if w: w.destroy()
        except Exception: pass
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.after(400, poll)
root.after(600, preview)
root.mainloop()