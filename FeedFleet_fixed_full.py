import os, csv, queue, threading, heapq, itertools, traceback, datetime
import tkinter as tk
from tkinter import ttk
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from torchvision import datasets

dev = "cuda" if torch.cuda.is_available() else "cpu"
ALGOS = ["FedFleet", "CFL-like", "HierFAVG", "FedAvg", "FedAsync"]
HERE = os.path.dirname(os.path.abspath(__file__))
X = Y = Xt = Yt = WM = EM = None
CM20 = matplotlib.colormaps["tab20"]
CM10 = matplotlib.colormaps["tab10"]
COL = {n: CM10(i) for i, n in enumerate(ALGOS)}


# =====================  FL core (simulation)  =====================
def make_model():
    return nn.Sequential(nn.Linear(784, 200), nn.ReLU(),
                         nn.Linear(200, 200), nn.ReLU(),
                         nn.Linear(200, 10)).to(dev)


def load_data(log):
    global X, Y, Xt, Yt, WM, EM
    if X is not None:
        return
    log("Loading MNIST (first time it downloads ~10 MB)...")
    rd = os.path.join(HERE, "data")
    tr = datasets.MNIST(rd, train=True, download=True)
    te = datasets.MNIST(rd, train=False, download=True)
    X = tr.data.float().div(255).view(-1, 784).to(dev); Y = tr.targets.to(dev)
    Xt = te.data.float().div(255).view(-1, 784).to(dev); Yt = te.targets.to(dev)
    WM = make_model(); EM = make_model()
    log("MNIST ready.")


def partition(C, rng):
    idx = np.argsort(Y.cpu().numpy(), kind="stable")
    n_sh = 2 * C
    size = len(idx) // n_sh
    shards = [idx[i * size:(i + 1) * size] for i in range(n_sh)]
    perm = rng.permutation(n_sh)
    return [torch.as_tensor(np.concatenate([shards[perm[2 * i]], shards[perm[2 * i + 1]]]),
                            device=dev) for i in range(C)]


def make_profile(C, H, rng):
    if H >= 0.24:
        z = rng.choice([-1.0, 1.0], C)
    else:
        z = rng.standard_normal(C)
        z = (z - z.mean()) / (z.std() + 1e-9)
    n = np.clip(0.5 + np.sqrt(H) * z, 0, 1)
    d = 5 + 20 * n
    return d, float(n.var())


def make_groups(name, d, E, seed):
    C = len(d)
    E = min(E, C)
    order = np.argsort(d, kind="stable")
    if name == "FedFleet":
        return np.array_split(order, E)
    if name == "CFL-like":
        return [order[i::E] for i in range(E)]
    if name == "HierFAVG":
        return np.array_split(np.random.default_rng(seed + 2).permutation(C), E)
    if name == "FedAvg":
        return [np.arange(C)]
    return [np.array([c]) for c in range(C)]


def avg_states(states):
    return {k: torch.stack([s[k] for s in states]).mean(0) for k in states[0]}


def mix(g, w, a):
    return {k: (1 - a) * g[k] + a * w[k] for k in g}


def local_train(gstate, ids, cfg):
    WM.load_state_dict(gstate)
    opt = torch.optim.SGD(WM.parameters(), lr=cfg["lr"])
    for _ in range(cfg["L"]):
        p = ids[torch.randperm(len(ids), device=dev)]
        for i in range(0, len(p), 50):
            b = p[i:i + 50]
            opt.zero_grad()
            F.cross_entropy(WM(X[b]), Y[b]).backward()
            opt.step()
    return {k: v.detach().clone() for k, v in WM.state_dict().items()}


@torch.no_grad()
def evaluate(state):
    EM.load_state_dict(state)
    return (EM(Xt).argmax(1) == Yt).float().mean().item() * 100


def run_algo(name, cfg, q, stop):
    C = cfg["C"]; E = min(cfg["E"], C)
    rng = np.random.default_rng(cfg["seed"])
    d, _ = make_profile(C, cfg["H"], rng)
    cidx = partition(C, rng)
    groups = make_groups(name, d, E, cfg["seed"])
    drng = np.random.default_rng(cfg["seed"] + 1)
    torch.manual_seed(cfg["seed"])
    drop, cloud, base = cfg["drop"] / 100, cfg["cloud"], cfg["alpha"]
    TMAX, target = cfg["tmax"], cfg["target"]
    st = lambda lag: 1.0 if lag < cfg["b"] else 1.0 / (cfg["a"] * (lag - cfg["b"]) + 1)
    g = {k: v.clone() for k, v in make_model().state_dict().items()}
    t = 0
    tt = [None]; last = [0.0]; counter = itertools.count()
    q.put(("start", name, d.tolist(), [list(map(int, gp)) for gp in groups], TMAX))

    def alive(ids):
        return [int(c) for c in ids if drng.random() >= drop]

    def emit_pt(clock):
        a = evaluate(g)
        q.put(("pt", name, clock, a, t))
        last[0] = a
        if tt[0] is None and a >= target:
            tt[0] = clock

    def early():
        return cfg["stop_at"] and tt[0] is not None

    if name in ("FedFleet", "CFL-like"):
        heap = []
        G = len(groups)

        def launch(e, now):
            ids = alive(groups[e])
            dur = (d[ids].max() if ids else d[groups[e]].max()) + cloud
            w = avg_states([local_train(g, cidx[c], cfg) for c in ids]) if ids else None
            heapq.heappush(heap, (now + dur, next(counter), e, t, now, w))

        for e in range(G):
            launch(e, 0.0)
        while heap and not stop.is_set():
            clock, _, e, tau, start, w = heapq.heappop(heap)
            if clock > TMAX:
                break
            s = st(t - tau)
            if w is not None:
                a_t = min(1.0, base * (1 - e / G) * s)
                g = mix(g, w, a_t); t += 1
                emit_pt(clock)
            q.put(("ev", name, e, start, clock, s if w is not None else -1))
            if early():
                break
            launch(e, clock)

    elif name == "HierFAVG":
        clock = 0.0
        while clock < TMAX and not stop.is_set():
            start = clock; models = []; durs = []
            for e, grp in enumerate(groups):
                ids = alive(grp)
                durs.append(d[ids].max() if ids else d[grp].max())
                if ids:
                    models.append(avg_states([local_train(g, cidx[c], cfg) for c in ids]))
                q.put(("ev", name, e, start, start + durs[-1] + cloud, 1.0))
            clock += max(durs) + cloud
            if clock > TMAX:
                break
            if models:
                g = avg_states(models); t += 1
                emit_pt(clock)
            if early():
                break

    elif name == "FedAvg":
        k = min(C, max(5, int(0.1 * C)))
        clock = 0.0
        while clock < TMAX and not stop.is_set():
            ids = rng.choice(C, size=k, replace=False)
            part = alive(ids)
            dur = (d[part].max() if part else d[ids].max()) + cloud
            if part:
                g = avg_states([local_train(g, cidx[c], cfg) for c in part])
            q.put(("ev", name, 0, clock, clock + dur, 1.0 if part else -1))
            clock += dur
            if clock > TMAX:
                break
            if part:
                t += 1
                emit_pt(clock)
            if early():
                break

    else:  # FedAsync
        bucket = np.zeros(C, int)
        for i, gp in enumerate(np.array_split(np.argsort(d, kind="stable"), E)):
            bucket[gp] = i
        heap = []

        def launch_c(c, now):
            w = local_train(g, cidx[c], cfg) if drng.random() >= drop else None
            heapq.heappush(heap, (now + d[c] + cloud, next(counter), c, t, now, w))

        for c in range(C):
            launch_c(c, 0.0)
        while heap and not stop.is_set():
            clock, _, c, tau, start, w = heapq.heappop(heap)
            if clock > TMAX:
                break
            s = st(t - tau)
            if w is not None:
                g = mix(g, w, min(1.0, 0.5 * base * s)); t += 1
                if t % 3 == 0:
                    emit_pt(clock)
            q.put(("ev", name, int(bucket[c]), start, clock, s if w is not None else -1))
            if early():
                break
            launch_c(c, clock)

    q.put(("done", name, tt[0], last[0], t))


def worker(cfg, algos, q, stop):
    try:
        load_data(lambda s: q.put(("log", s)))
        for name in algos:
            if stop.is_set():
                break
            q.put(("log", f"--- running {name} ---"))
            run_algo(name, cfg, q, stop)
    except Exception:
        q.put(("log", traceback.format_exc()))
    q.put(("finished",))


# =====================  GUI  =====================
root = tk.Tk()
root.title("FedFleet Simulator (with simple explanations)")
root.geometry("1550x930")
q = queue.Queue()
stop_ev = threading.Event()
running = [False]
results = {}      # name -> (times, accs)
summary = {}      # name -> (time_to_target or None, final_acc, updates)
events = []
cur = {"name": "FedFleet", "d": None, "groups": None, "tmax": 3000}

left = ttk.Frame(root, padding=8); left.pack(side="left", fill="y")
right = ttk.Frame(root); right.pack(side="right", fill="both", expand=True)

ctrl = ttk.LabelFrame(left, text="Settings (move the sliders)", padding=6); ctrl.pack(fill="x")
row = [0]
_prev_job = [None]


def schedule_preview(*_):
    if running[0]:
        return
    if _prev_job[0]:
        root.after_cancel(_prev_job[0])
    _prev_job[0] = root.after(300, preview)


def slider(text, lo, hi, init, step, fmt="{:.0f}", prev=False):
    var = tk.DoubleVar(value=init)
    ttk.Label(ctrl, text=text).grid(row=row[0], column=0, sticky="w", pady=2)
    val = ttk.Label(ctrl, text=fmt.format(init), width=7)
    val.grid(row=row[0], column=2)

    def show(x):
        var.set(x); val.config(text=fmt.format(x))

    def cb(v):
        show(round(float(v) / step) * step)
        if prev:
            schedule_preview()

    ttk.Scale(ctrl, from_=lo, to=hi, variable=var, command=cb, length=150).grid(row=row[0], column=1, padx=4)
    var.show = show
    row[0] += 1
    return var


ttk.Label(ctrl, text="Algorithm").grid(row=row[0], column=0, sticky="w")
algo_var = tk.StringVar(value="FedFleet")
cb_algo = ttk.Combobox(ctrl, textvariable=algo_var, values=ALGOS, state="readonly", width=16)
cb_algo.grid(row=row[0], column=1, columnspan=2, sticky="w", pady=3)
cb_algo.bind("<<ComboboxSelected>>", schedule_preview)
row[0] += 1

vC = slider("Phones / clients (C)", 10, 200, 100, 10, prev=True)
vE = slider("Edge servers (E)", 2, 20, 10, 1, prev=True)
vH = slider("Speed differences (H)", 0, 0.25, 0.125, 0.005, "{:.3f}", prev=True)
vDrop = slider("Phones dropping out %", 0, 90, 0, 5)
vL = slider("Local epochs (L)", 1, 10, 3, 1)
vLR = slider("Learning rate", 0.001, 0.1, 0.01, 0.001, "{:.3f}")
vCloud = slider("Cloud delay (s)", 0, 100, 20, 1)
vAlpha = slider("Base alpha", 0.1, 1.0, 0.8, 0.05, "{:.2f}")
vA = slider("Staleness a", 0, 3, 1.0, 0.1, "{:.1f}")
vB = slider("Staleness b", 0, 20, 5, 1)
vT = slider("Max simulated time (s)", 200, 30000, 3000, 100)
vTarget = slider("Target accuracy %", 50, 99, 90, 1)

ttk.Label(ctrl, text="Seed (random)").grid(row=row[0], column=0, sticky="w")
vSeed = tk.IntVar(value=0)
sp = ttk.Spinbox(ctrl, from_=0, to=9999, textvariable=vSeed, width=8, command=schedule_preview)
sp.grid(row=row[0], column=1, sticky="w"); row[0] += 1
vStop = tk.BooleanVar(value=False)
ttk.Checkbutton(ctrl, text="Stop when target reached", variable=vStop).grid(row=row[0], column=0, columnspan=3, sticky="w")


def get_cfg():
    return dict(C=int(round(vC.get())), E=int(round(vE.get())), H=float(vH.get()),
                drop=float(vDrop.get()), L=int(round(vL.get())), lr=float(vLR.get()),
                cloud=float(vCloud.get()), alpha=float(vAlpha.get()), a=float(vA.get()),
                b=int(round(vB.get())), tmax=float(vT.get()), target=float(vTarget.get()),
                seed=int(vSeed.get()), stop_at=bool(vStop.get()))


# ---- plots
fig = Figure(figsize=(10, 6.0), dpi=90, constrained_layout=True)
ax_acc = fig.add_subplot(221); ax_dev = fig.add_subplot(222)
ax_gan = fig.add_subplot(223); ax_idle = fig.add_subplot(224)
canvas = FigureCanvasTkAgg(fig, master=right)
canvas.get_tk_widget().pack(fill="both", expand=True)
TS = 9


def draw_acc():
    ax = ax_acc; ax.clear()
    for nm, (xs, ys) in results.items():
        ax.plot(xs, ys, label=nm, color=COL[nm], lw=1.6)
    ax.axhline(get_cfg()["target"], color="red", ls="--", lw=1)
    ax.set_xlim(0, cur["tmax"]); ax.set_ylim(0, 100)
    ax.set_xlabel("Simulated time (s)"); ax.set_ylabel("AI accuracy (%)")
    ax.set_title("1) Accuracy vs time: the line that hits RED first wins", fontsize=TS)
    ax.grid(alpha=.3)
    if results:
        ax.legend(fontsize=8, loc="lower right")


def draw_dev():
    ax = ax_dev; ax.clear()
    d, groups, name = cur["d"], cur["groups"], cur["name"]
    if d is None:
        return
    if name in ("FedAvg", "FedAsync"):
        colors = "gray"
    else:
        lab = np.zeros(len(d), int)
        for i, gp in enumerate(groups):
            lab[gp] = i
        colors = [CM20(i % 20) for i in lab]
    ax.scatter(range(len(d)), d, c=colors, s=18)
    ax.set_xlabel("Phone number"); ax.set_ylabel("Seconds a phone needs")
    ax.set_title(f"2) Phones: high dot = slow phone, colour = its group ({name})", fontsize=TS)
    ax.grid(alpha=.3)


def idle_values(name, d, groups):
    if name in ("FedFleet", "CFL-like"):
        return [float(np.mean(d[gp].max() - d[gp])) for gp in groups]
    if name == "HierFAVG":
        return [float(np.mean(d.max() - d[gp])) for gp in groups]
    if name == "FedAvg":
        return [float(np.mean(d.max() - d))]
    return [0.0]


def draw_idle():
    ax = ax_idle; ax.clear()
    d, groups, name = cur["d"], cur["groups"], cur["name"]
    if d is None:
        return
    vals = idle_values(name, d, groups)
    ax.bar(range(len(vals)), vals, color=[CM20(i % 20) for i in range(len(vals))])
    ax.set_xlabel("Group (fleet / edge server)"); ax.set_ylabel("Waiting time (s)")
    ax.set_title(f"4) Waiting per phone: SHORT bars = good (avg {np.mean(vals):.1f}s)", fontsize=TS)
    ax.grid(alpha=.3)


def draw_gantt():
    ax = ax_gan; ax.clear()
    ev = events[-80:]
    ax.set_title("3) Timeline: each bar = one group finishing (faint = old update)", fontsize=TS)
    ax.set_xlabel("Simulated time (s)"); ax.set_ylabel("Group")
    if not ev:
        return
    for (e, s0, s1, s) in ev:
        if s < 0:
            ax.barh(e, s1 - s0, left=s0, height=0.7, color="lightgray", edgecolor="k", lw=.3)
        else:
            ax.barh(e, s1 - s0, left=s0, height=0.7, color=CM20(e % 20),
                    alpha=0.35 + 0.65 * min(1.0, s), edgecolor="k", lw=.3)
    ax.set_xlim(min(x[1] for x in ev), max(x[2] for x in ev))
    ax.set_ylim(-0.7, max(x[0] for x in ev) + 0.7)


def redraw(all_panels=False):
    draw_acc(); draw_gantt()
    if all_panels:
        draw_dev(); draw_idle()
    canvas.draw_idle()


# ---- bottom area: status, table, tabs
status = ttk.Label(right, text="Ready.", font=("Segoe UI", 10, "bold"), wraplength=1000)
status.pack(fill="x", padx=6)
prog = ttk.Progressbar(right, maximum=100); prog.pack(fill="x", padx=6, pady=2)
bottom = ttk.Frame(right); bottom.pack(fill="x")
cols = ("Algorithm", "Time to target (s)", "Final acc %", "Updates")
tree = ttk.Treeview(bottom, columns=cols, show="headings", height=8)
for c_, w_ in zip(cols, (100, 120, 80, 65)):
    tree.heading(c_, text=c_); tree.column(c_, width=w_, anchor="center")
tree.pack(side="left", padx=4, pady=4)

nb = ttk.Notebook(bottom); nb.pack(side="left", fill="both", expand=True, padx=4, pady=4)


def make_text(parent, h=12):
    fr = ttk.Frame(parent)
    sb = ttk.Scrollbar(fr); sb.pack(side="right", fill="y")
    tx = tk.Text(fr, height=h, width=70, wrap="word", font=("Segoe UI", 10), yscrollcommand=sb.set)
    tx.pack(side="left", fill="both", expand=True); sb.config(command=tx.yview)
    tx.tag_configure("h", font=("Segoe UI", 11, "bold"), spacing1=6)
    tx.tag_configure("good", foreground="#0a7a2f")
    tx.tag_configure("warn", foreground="#b35a00")
    return fr, tx


f1, explbox = make_text(nb); nb.add(f1, text="Simple explanation (auto)")
f2, guidebox = make_text(nb); nb.add(f2, text="How to read the graphs")
f3, logbox = make_text(nb); nb.add(f3, text="Log")

GUIDE = [
    ("h", "The story\n"),
    ("", "Imagine 100 phones. Each phone has pictures of handwritten digits, but only of about 2 different digits. "
         "We want ONE smart AI that knows all digits. Phones may NOT send their pictures (privacy), so each phone trains "
         "a small AI and sends only the AI. The server mixes them together.\n\n"
         "Problem: some phones are fast, some are slow. If the server waits for the slowest phone, fast phones sit idle. "
         "FedFleet puts phones with SIMILAR speed in the same group (fleet), so nobody waits long.\n\n"),
    ("h", "Graph 1 - Accuracy vs time (most important)\n"),
    ("", "Up = smarter AI. Left = faster. The red dashed line is your target. The algorithm whose line touches red "
         "first is the fastest. This is the main result for your report.\n\n"),
    ("h", "Graph 2 - Phones\n"),
    ("", "Each dot is one phone. High dot = slow phone. Colour = which group it belongs to. For FedFleet you see clean "
         "colour bands (fast together, slow together). For HierFAVG the colours are mixed randomly.\n\n"),
    ("h", "Graph 3 - Timeline\n"),
    ("", "Each bar is one group finishing a round and sending its result to the server. Faint bars are OLD updates that "
         "count less (staleness). Gaps show waiting. Use it only to see when things happen.\n\n"),
    ("h", "Graph 4 - Waiting time\n"),
    ("", "How long a phone sits idle waiting for the slowest phone of its group. Short bars are good. This proves the "
         "paper's idea: grouping by speed reduces waiting.\n\n"),
    ("h", "Main settings in simple words\n"),
    ("", "- Speed differences (H): 0 = all phones same speed, 0.25 = half very fast and half very slow.\n"
         "- Phones dropping out %: phones that randomly disappear each round.\n"
         "- Cloud delay: extra time to talk to the far-away central server.\n"
         "- Target accuracy: the goal; we measure how fast each method gets there.\n"
         "- Seed: changes the random numbers. Try 3 different seeds and average, for a fair result.\n\n"),
    ("h", "Quick start\n"),
    ("", "Click 'Quick demo settings', then 'Run ALL 5'. Then read the 'Simple explanation' tab and click "
         "'Save results + report'.\n"),
]
for tag, s in GUIDE:
    guidebox.insert("end", s, tag)
guidebox.config(state="disabled")


def log(s):
    logbox.insert("end", s + "\n"); logbox.see("end")


def set_expl(parts):
    explbox.config(state="normal"); explbox.delete("1.0", "end")
    for tag, s in parts:
        explbox.insert("end", s, tag)
    explbox.config(state="disabled")


def h_level(H):
    if H < 0.02:
        return "almost none (all phones have nearly the same speed)"
    if H < 0.06:
        return "low (phones are a bit different)"
    if H < 0.18:
        return "medium (a realistic mix of fast and slow phones)"
    return "high (very fast and very slow phones)"


def explain():
    cfg = get_cfg(); d = cur["d"]; P = []
    add = lambda tag, s: P.append((tag, s))

    add("h", "1. What is happening right now?\n")
    add("", f"We pretend there are {cfg['C']} phones and {cfg['E']} edge servers (small helper servers). Each phone trains "
            f"a small AI on its own digit pictures and sends only the AI, never the pictures. The central server mixes "
            f"the AIs into one smart AI. Time is simulated, so your PC does not really wait.\n\n")

    add("h", "2. Your settings in plain words\n")
    add("", f"Speed differences: {h_level(cfg['H'])}.\n")
    if cfg["drop"] > 0:
        add("", f"About {cfg['drop']:.0f}% of phones disappear each round, so training is harder and the curve is shakier.\n")
    else:
        add("", "No phones drop out.\n")
    add("", f"Goal: reach {cfg['target']:.0f}% accuracy. The method that gets there in the least simulated time wins.\n\n")

    if d is not None and len(d) > 1:
        add("h", "3. Does grouping by speed reduce waiting? (no training needed)\n")
        E = min(cfg["E"], len(d))
        iF = float(np.mean(idle_values("FedFleet", d, make_groups("FedFleet", d, E, cfg["seed"]))))
        iH = float(np.mean(idle_values("HierFAVG", d, make_groups("HierFAVG", d, E, cfg["seed"]))))
        if iH < 0.5:
            add("", "With these settings all phones have almost the same speed, so nobody waits much in any method. "
                    "FedFleet cannot help here. Raise 'Speed differences (H)' to see its advantage.\n\n")
        else:
            cut = 100 * (iH - iF) / iH
            add("", f"If phones are put in groups randomly (HierFAVG), a phone waits about {iH:.1f} s per round. "
                    f"If they are grouped by speed (FedFleet), it waits only about {iF:.1f} s.\n")
            add("good", f"That is about {cut:.0f}% less waiting. This is the main idea of the paper (graph 4).\n\n")

    add("h", "4. Results of your runs\n")
    if not summary:
        add("", "No training results yet. Click 'Quick demo settings' and then 'Run ALL 5'.\n")
    else:
        tg = cfg["target"]
        reached = sorted([(tm, n) for n, (tm, a, u) in summary.items() if tm is not None])
        missed = [(n, a) for n, (tm, a, u) in summary.items() if tm is None]
        for i, (tm, n) in enumerate(reached):
            add("good" if i == 0 else "", f"{i + 1}. {n} reached {tg:.0f}% after {tm:.0f} simulated seconds.\n")
        for n, a in missed:
            add("warn", f"-  {n} did NOT reach {tg:.0f}% (it got to {a:.1f}%).\n")
        add("", "\n")
        add("h", "5. What does this mean?\n")
        if "FedFleet" in summary:
            tF = summary["FedFleet"][0]
            if tF is not None and reached and reached[0][1] == "FedFleet":
                add("good", "FedFleet was the FASTEST. This matches the paper: grouping similar-speed phones saves time.\n")
                if "HierFAVG" in summary and summary["HierFAVG"][0]:
                    tH = summary["HierFAVG"][0]
                    add("", f"It was about {100 * (tH - tF) / tH:.0f}% faster than HierFAVG "
                            f"({tF:.0f}s vs {tH:.0f}s), because HierFAVG makes fast phones wait for slow ones.\n")
            elif tF is not None:
                add("warn", f"FedFleet reached the target but {reached[0][1]} was faster. That can happen in a small, "
                            "simple simulation (for example when H is small). Try a higher H, another seed, or a lower "
                            "'Base alpha' (0.5). Report what you really see.\n")
            else:
                add("warn", "FedFleet did not reach the target in the allowed time. Increase 'Max simulated time', lower "
                            "the target, or lower 'Base alpha'.\n")
        if "FedAsync" in summary and summary["FedAsync"][0] is None:
            add("", "FedAsync updates the server one phone at a time, so old updates make it jumpy and it often "
                    "stays below the target.\n")
        if len(summary) == 1:
            add("", "Only one algorithm was run. Press 'Run ALL 5' to compare.\n")
        add("", "\nRemember: speeds are made up and time is simulated, so compare the TREND with the paper, not the exact numbers.\n")
    set_expl(P)


# ---- separate explanation and guide windows
def open_text_window(title, source_widget):
    """Open a separate, scrollable window containing an existing text panel."""
    window = tk.Toplevel(root)
    window.title(title)
    window.geometry("900x650")
    window.minsize(650, 450)

    header = ttk.Frame(window, padding=(16, 14, 16, 8))
    header.pack(fill="x")
    ttk.Label(header, text=title, font=("Segoe UI", 16, "bold")).pack(anchor="w")

    body = ttk.Frame(window, padding=(16, 4, 16, 12))
    body.pack(fill="both", expand=True)
    scrollbar = ttk.Scrollbar(body, orient="vertical")
    scrollbar.pack(side="right", fill="y")

    text = tk.Text(
        body, wrap="word", font=("Segoe UI", 11), padx=14, pady=12,
        yscrollcommand=scrollbar.set, undo=False
    )
    text.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=text.yview)

    try:
        content = source_widget.get("1.0", "end-1c")
    except tk.TclError:
        content = "No content is available yet."
    text.insert("1.0", content.strip() or "No content is available yet.")
    text.config(state="disabled")

    footer = ttk.Frame(window, padding=(16, 0, 16, 14))
    footer.pack(fill="x")
    ttk.Button(footer, text="Close", command=window.destroy).pack(side="right")
    return window


def open_explanation_window():
    """Show the current automatic explanation in its own window."""
    explain()
    return open_text_window("FedFleet — Simple Explanation", explbox)


def open_guide_window():
    """Show the graph-reading guide in its own window."""
    return open_text_window("FedFleet — How to Read the Graphs", guidebox)


# ---- actions
def preview():
    _prev_job[0] = None
    if running[0]:
        return
    cfg = get_cfg()
    rng = np.random.default_rng(cfg["seed"])
    d, hact = make_profile(cfg["C"], cfg["H"], rng)
    name = algo_var.get()
    cur.update(name=name, d=d, groups=make_groups(name, d, cfg["E"], cfg["seed"]), tmax=cfg["tmax"])
    status.config(text=f"Preview of groups ({name}). Actual speed difference H = {hact:.3f}. "
                       "Press Start to train.")
    redraw(True); explain()


def start(algos):
    if running[0]:
        return
    cfg = get_cfg()
    cur["tmax"] = cfg["tmax"]
    stop_ev.clear(); running[0] = True
    for b in (btn_start, btn_all):
        b.state(["disabled"])
    threading.Thread(target=worker, args=(cfg, algos, q, stop_ev), daemon=True).start()


def clear_all():
    results.clear(); events.clear(); summary.clear()
    for r in tree.get_children():
        tree.delete(r)
    redraw(True); explain()


def quick_demo():
    for var, x in ((vC, 50), (vE, 5), (vH, 0.125), (vDrop, 0), (vL, 2), (vT, 1500), (vTarget, 85)):
        var.show(x)
    vStop.set(True)
    schedule_preview()


def save_results():
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    png = os.path.join(HERE, f"result_plots_{stamp}.png")
    csvp = os.path.join(HERE, f"result_table_{stamp}.csv")
    txt = os.path.join(HERE, f"report_{stamp}.txt")
    fig.savefig(png, dpi=200)
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in tree.get_children():
            w.writerow(tree.item(r)["values"])
    cfg = get_cfg()
    with open(txt, "w", encoding="utf-8") as f:
        f.write("FedFleet simulation report\n==========================\n")
        f.write("Settings: " + ", ".join(f"{k}={v}" for k, v in cfg.items()) + "\n\n")
        f.write(explbox.get("1.0", "end"))
    log(f"Saved: {png}\n       {csvp}\n       {txt}")
    status.config(text="Saved 3 files (picture, table, text report) in the program folder.")


bf = ttk.Frame(left, padding=(0, 8)); bf.pack(fill="x")
ttk.Button(bf, text="Open explanation window", command=open_explanation_window).pack(fill="x", pady=2)
ttk.Button(bf, text="Open graph guide window", command=open_guide_window).pack(fill="x", pady=2)
ttk.Button(bf, text="Quick demo settings (fast)", command=quick_demo).pack(fill="x", pady=2)
btn_start = ttk.Button(bf, text="▶ Start selected algorithm", command=lambda: start([algo_var.get()]))
btn_start.pack(fill="x", pady=2)
btn_all = ttk.Button(bf, text="▶▶ Run ALL 5 (compare)", command=lambda: start(ALGOS))
btn_all.pack(fill="x", pady=2)
ttk.Button(bf, text="■ Stop", command=lambda: stop_ev.set()).pack(fill="x", pady=2)
ttk.Button(bf, text="Preview groups", command=preview).pack(fill="x", pady=2)
ttk.Button(bf, text="Clear results", command=clear_all).pack(fill="x", pady=2)
ttk.Button(bf, text="💾 Save results + report", command=save_results).pack(fill="x", pady=2)


def poll():
    got = False; panels = False; refresh = False
    try:
        while True:
            m = q.get_nowait()
            k = m[0]
            if k == "log":
                log(m[1])
            elif k == "start":
                _, name, d, groups, tmax = m
                results[name] = ([], []); events.clear()
                cur.update(name=name, d=np.array(d), groups=[np.array(gp) for gp in groups], tmax=tmax)
                status.config(text=f"Training {name}... (fast phones finish early, slow ones later)")
                panels = True; got = True; refresh = True
            elif k == "pt":
                _, name, clock, acc, t = m
                results[name][0].append(clock); results[name][1].append(acc)
                prog["value"] = 100 * clock / max(cur["tmax"], 1)
                status.config(text=f"{name}: simulated time {clock:.0f}s of {cur['tmax']:.0f}s | "
                                   f"server updated {t} times | AI now recognises {acc:.1f}% of test digits")
                got = True
            elif k == "ev":
                events.append(m[2:])
                if len(events) > 400:
                    del events[:200]
                got = True
            elif k == "done":
                _, name, tt, acc, t = m
                summary[name] = (tt, acc, t)
                tree.insert("", "end", values=(name, f"{tt:.0f}" if tt is not None else "not reached",
                                               f"{acc:.2f}", t))
                got = True; refresh = True
            elif k == "finished":
                running[0] = False
                for b in (btn_start, btn_all):
                    b.state(["!disabled"])
                status.config(text="Finished. Read the 'Simple explanation' tab, then click 'Save results + report'.")
                nb.select(f1)
                got = True; refresh = True
    except queue.Empty:
        pass
    if got:
        redraw(panels)
    if refresh:
        explain()
    root.after(300, poll)


def on_close():
    stop_ev.set(); root.destroy()


root.protocol("WM_DELETE_WINDOW", on_close)
root.after(300, poll)
root.after(300, preview)
root.mainloop()