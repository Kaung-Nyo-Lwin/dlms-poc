"""Browser picking widget: click things on an image, get pixels back.

One page, three modes, shared by every survey step in ``calibrate.py``:

* ``points``  — click N (or any number of) locations; optionally type a world
  coordinate for each, which is what a ground-control survey is, or a taped
  length for each pair, which is what a scale check is.
* ``box``     — drag a rectangle, then nudge it; used to cut a sticker template.
* ``shapes``  — draw named points / lines / polygons; used for ROIs.

Why a browser and not an OpenCV window: highgui needs a display, and on WSL a
missing one ends in ``abort()`` rather than an exception you can catch. A
loopback HTTP server works everywhere the operator can open a tab, which is the
same reason the studio is built this way.

Precision comes from three things, and they are the whole point of the widget:
the image is served at native resolution and magnified by the canvas rather than
resampled by the server, so a zoomed-in pixel is a real pixel; a loupe follows
the cursor; and every click can be snapped to the nearest sub-pixel corner,
which removes most of the operator's hand jitter on a painted cross.

Standalone: numpy + opencv + the standard library.
"""

from __future__ import annotations

import contextlib
import json
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

_TERM = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)


def refine_corner(image: np.ndarray, x: float, y: float, win: int = 9) -> tuple[float, float]:
    """Snap a click to the nearest sub-pixel corner.

    Ground control marks are painted crosses or plate corners, and snapping
    removes most of the operator's hand jitter. When no corner dominates the
    window ``cornerSubPix`` returns close to the input, so this is safe to apply
    always — except when it runs away, which means there was nothing to snap to.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if not (win < x < w - win and win < y < h - win):
        return x, y
    out = cv2.cornerSubPix(gray, np.array([[[x, y]]], np.float32), (win, win), (-1, -1), _TERM)
    rx, ry = float(out[0, 0, 0]), float(out[0, 0, 1])
    return (x, y) if np.hypot(rx - x, ry - y) > win else (rx, ry)


def _serve(image: np.ndarray, meta: dict, open_browser: bool = True) -> dict | None:
    """Run the page until the operator saves or cancels. None means cancelled."""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise SystemExit("could not encode the image for the picker")
    png = buf.tobytes()
    token = secrets.token_urlsafe(12)
    done = threading.Event()
    result: dict = {}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_a: object) -> None:
            pass

        def _send(self, body: bytes, ctype: str, last: bool = False) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if last:
                # Release the socket on the way out. The browser would otherwise
                # hold this connection open for reuse, and a half-open keep-alive
                # is the thing most likely to keep the server alive after the
                # operator has finished.
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            self.wfile.write(body)

        def _auth(self, q: dict) -> bool:
            # Any page in the operator's browser can reach a loopback port; the
            # token is what stops an unrelated tab from posting a survey.
            return q.get("t", [""])[0] == token

        def do_GET(self) -> None:
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/" and self._auth(q):
                self._send(_PAGE.encode(), "text/html; charset=utf-8")
            elif u.path == "/image.png" and self._auth(q):
                self._send(png, "image/png")
            elif u.path == "/meta" and self._auth(q):
                h, w = image.shape[:2]
                self._send(json.dumps({**meta, "width": w, "height": h}).encode(),
                           "application/json")
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if not self._auth(q):
                self.send_error(403)
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if u.path == "/snap":
                x, y = refine_corner(image, float(body["x"]), float(body["y"]))
                self._send(json.dumps({"x": x, "y": y}).encode(), "application/json")
            elif u.path == "/done":
                result.update(body)
                self._send(b"{}", "application/json", last=True)
                done.set()
            elif u.path == "/cancel":
                self._send(b"{}", "application/json", last=True)
                done.set()
            else:
                self.send_error(404)

    # Threaded, and not by preference. A plain HTTPServer serves one connection
    # at a time, and a browser keeps its connection open for reuse — so the
    # accept loop stays parked inside that socket, `shutdown()` below never
    # returns, and the step hangs *after* the operator has pressed Save without
    # ever writing the file. A client that sends `Connection: close` (urllib,
    # curl) hides this completely, which is exactly why it survived testing.
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    url = f"http://127.0.0.1:{server.server_port}/?t={token}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"  picker: {url}", flush=True)
    if open_browser:
        with contextlib.suppress(Exception):
            webbrowser.open(url)
    try:
        done.wait()
    except KeyboardInterrupt:
        result.clear()
    finally:
        server.shutdown()
        server.server_close()
    return result or None


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------


def pick_points(image, title, n=None, world=False, snap=True, hint="", open_browser=True,
                focus=None):
    """Click locations. ``world=True`` also asks for each point's X,Y in mm.

    ``focus`` is ``(x, y, span)`` in image pixels: open centred there, zoomed so
    that ``span`` fills the window, instead of fitted to the whole image. `r`
    still resets to the fit. Worth passing whenever the caller already knows
    roughly where the subject is — on a full station frame it is the difference
    between clicking and hunting.

    Returns ``[{"px": [x, y], "world_mm": [X, Y]}, ...]`` (``world_mm`` absent
    when ``world`` is False), or None if cancelled.
    """
    got = _serve(image, {"mode": "points", "title": title, "n": n, "world": world,
                         "snap": snap, "hint": hint,
                         "focus": None if focus is None else [float(v) for v in focus]},
                 open_browser)
    return None if got is None else got.get("points")


def pick_bars(image, title, hint="", open_browser=True, focus=None):
    """Click the two ends of each taped length. Returns a list of bars, or None.

    The same page as ``pick_points``, with the toolbar field holding one number
    instead of two: type the millimetres the tape read, then click both ends.
    Unlike a ground control point, neither end needs a world coordinate — the
    length is the whole observation, which is what makes these cheap to lay.

    Returns ``[{"a_px": [x, y], "b_px": [x, y], "length_mm": d}, ...]``.
    """
    got = _serve(image, {"mode": "points", "title": title, "n": None, "world": False,
                         "bars": True, "snap": True, "hint": hint,
                         "focus": None if focus is None else [float(v) for v in focus]},
                 open_browser)
    if got is None or not got.get("points"):
        return None
    pts = got["points"]
    return [{"a_px": pts[i]["px"], "b_px": pts[i + 1]["px"],
             "length_mm": float(pts[i]["len_mm"])} for i in range(0, len(pts) - 1, 2)]


def pick_box(image, title, mm_per_px=None, hint="", open_browser=True):
    """Drag a rectangle. Returns ``(x, y, w, h)`` in image pixels, or None.

    ``mm_per_px`` makes the readout metric, which is what lets the operator
    check the box against the marker's tape-measured size before committing.
    """
    got = _serve(image, {"mode": "box", "title": title, "mmPerPx": mm_per_px,
                         "snap": True, "hint": hint}, open_browser)
    if got is None or not got.get("box"):
        return None
    x0, y0, x1, y1 = got["box"]
    return min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)


def review_layers(image, title, layers, marks=(), hint="", open_browser=True):
    """Show worked-out overlays for approval. True to accept, None to redo.

    ``layers`` is a list of ``{"name", "points", "closed", "colour", "on"}`` in
    image pixels, each with its own toolbar toggle. ``marks`` are drawn always,
    for the raw clicks an adjustment moved.

    The page draws and nothing else: every coordinate is computed in Python and
    handed over finished. A preview that recomputed the geometry in JavaScript
    would be a second implementation free to drift from the first, and the one
    thing a preview must never do is agree with itself while disagreeing with
    the file.
    """
    got = _serve(image, {"mode": "review", "title": title, "hint": hint,
                         "layers": list(layers), "marks": [list(m) for m in marks],
                         "snap": False}, open_browser)
    return None if got is None else bool(got.get("ok"))


def pick_shapes(image, title, hint="", open_browser=True):
    """Draw named point / line / polygon shapes. Returns a list, or None."""
    got = _serve(image, {"mode": "shapes", "title": title, "snap": True, "hint": hint},
                 open_browser)
    return None if got is None else got.get("shapes")


_PAGE = r"""<!doctype html><meta charset=utf-8><title>survey</title>
<style>
 html,body{margin:0;height:100%;background:#0d1117;color:#c9d1d9;
   font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;overflow:hidden}
 #cv{display:block;cursor:crosshair}
 #bar{position:fixed;top:0;left:0;right:0;padding:8px 12px;background:#161b22ee;
   border-bottom:1px solid #30363d;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 #bar b{color:#e6edf3}
 kbd{background:#21262d;border:1px solid #30363d;border-radius:3px;padding:0 4px}
 .muted{color:#8b949e}
 #loupe{position:fixed;right:12px;bottom:12px;border:1px solid #30363d;background:#000}
 #list{position:fixed;right:12px;top:56px;width:280px;max-height:55vh;overflow:auto;
   background:#161b22ee;border:1px solid #30363d;border-radius:6px;padding:8px}
 #list div{padding:2px 4px;border-bottom:1px solid #21262d;white-space:nowrap;
   overflow:hidden;text-overflow:ellipsis}
 button{background:#238636;color:#fff;border:0;border-radius:5px;padding:5px 12px;
   font:inherit;cursor:pointer}
 button.g{background:#21262d;border:1px solid #30363d;color:#c9d1d9}
</style>
<div id=bar>
  <b id=title>…</b><span class=muted id=hint></span>
  <span class=muted id=stat></span><span id=msg></span>
  <input id=entry autocomplete=off>
  <span id=layerbtns style=display:none></span>
  <span id=shapebtns style=display:none>
    <button class="g s" data-t=point>point (p)</button>
    <button class="g s" data-t=line>line (l)</button>
    <button class="g s" data-t=polygon>polygon (g)</button>
  </span>
  <span class=muted><kbd>wheel</kbd> zoom · <kbd>drag</kbd> pan · <kbd>s</kbd> snap ·
    <kbd>u</kbd> undo · <kbd>r</kbd> reset · <kbd>←↑↓→</kbd> nudge</span>
  <button id=ok>Save (Enter)</button><button class=g id=no>Cancel (Esc)</button>
</div>
<canvas id=cv></canvas><canvas id=loupe width=180 height=180></canvas>
<div id=list></div>
<script>
const Q = location.search, cv = document.getElementById("cv"), g = cv.getContext("2d");
const lo = document.getElementById("loupe"), lg = lo.getContext("2d");
let meta = null, img = new Image(), zoom = 1, cx = 0, cy = 0;
let pts = [], shapes = [], cur = [], box = null, drag = null, snap = true, mouse = [0, 0];

const W = () => cv.width, H = () => cv.height;
const toImg = (sx, sy) => [cx + (sx - W() / 2) / zoom, cy + (sy - H() / 2) / zoom];
const toScr = (ix, iy) => [(ix - cx) * zoom + W() / 2, (iy - cy) * zoom + H() / 2];
const bye = (p, msg) => { if (p) post(p);
  document.body.innerHTML = `<p style=padding:2em>${msg} — you can close this tab</p>`; };
// A plain toolbar field, never a modal and never a blocking prompt. Committing a
// shape must not depend on anything the browser might refuse: a dialog can be
// suppressed outright in an embedded view, and even when it renders it may never
// receive keyboard focus — which used to leave a promise pending forever and
// every subsequent keystroke swallowed. So the field is optional, read only at
// the moment a shape is finished, and finishing is always immediate.
const entry = () => document.getElementById("entry");
function takeEntry() {
  const v = entry().value.trim();
  entry().value = "";
  return v;
}
let msgTimer = 0;
function say(text, bad) {
  const el = document.getElementById("msg");
  el.textContent = text; el.className = bad ? "bad" : "";
  clearTimeout(msgTimer);
  msgTimer = setTimeout(() => { el.textContent = ""; }, 4000);
}
const post = (p, b) => fetch(p + Q, {method: "POST", headers: {"Content-Type": "application/json"},
                                     body: JSON.stringify(b || {})}).then(r => r.json());

function resize() { cv.width = innerWidth; cv.height = innerHeight; draw(); }
// Open on the thing being clicked, not on the whole frame. A station image is
// 3840 px wide and the car is a fifth of it, so fitting the lot means every pass
// starts by scrolling and scrubbing to find the subject. `focus` is [x, y, span]
// in image pixels: centre there, and zoom so that span fills most of the window.
function focusView() {
  if (!meta.focus) return false;
  const [fx, fy, span] = meta.focus;
  cx = fx; cy = fy;
  zoom = Math.min(60, Math.max(0.02, Math.min(W(), H()) * 0.8 / Math.max(span, 1)));
  return true;
}
function reset() { zoom = Math.min(W() / meta.width, H() / meta.height);
                   cx = meta.width / 2; cy = meta.height / 2; draw(); }

function draw() {
  if (!meta) return;
  g.setTransform(1, 0, 0, 1, 0, 0); g.fillStyle = "#0d1117"; g.fillRect(0, 0, W(), H());
  g.imageSmoothingEnabled = zoom < 1.5;
  const [ox, oy] = toScr(0, 0);
  g.drawImage(img, ox, oy, meta.width * zoom, meta.height * zoom);

  const dot = (p, col, label) => {
    const [x, y] = toScr(p[0], p[1]);
    g.strokeStyle = col; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(x - 8, y); g.lineTo(x + 8, y);
    g.moveTo(x, y - 8); g.lineTo(x, y + 8); g.stroke();
    g.beginPath(); g.arc(x, y, 3, 0, 7); g.stroke();
    if (label) { g.fillStyle = col; g.fillText(label, x + 10, y - 8); }
  };
  const chain = (p, col, closed) => {
    if (p.length < 2) return;
    g.strokeStyle = col; g.lineWidth = 2; g.beginPath();
    p.forEach((q, i) => { const [x, y] = toScr(q[0], q[1]); i ? g.lineTo(x, y) : g.moveTo(x, y); });
    if (closed) g.closePath();
    g.stroke();
  };
  g.font = "12px ui-monospace, monospace";

  if (meta.mode === "points") {
    if (meta.bars)
      for (let i = 0; i + 1 < pts.length; i += 2)
        chain([pts[i].px, pts[i + 1].px], "#f778ba", false);
    pts.forEach((p, i) => dot(p.px, "#f778ba", labelOf(p, i)));
  }
  if (meta.mode === "shapes") {
    shapes.forEach(s => { chain(s.points_px, "#3fb950", s.type === "polygon");
                          s.points_px.forEach(q => dot(q, "#3fb950"));
                          dot(s.points_px[0], "#3fb950", s.name); });
    chain(cur, "#ffa657", false); cur.forEach(q => dot(q, "#ffa657"));
  }
  if (meta.mode === "review") {
    // Nothing is computed here. Every coordinate arrived from Python already
    // worked out, so what is drawn cannot disagree with what gets written.
    (meta.marks || []).forEach(q => dot(q, "#8b949e"));
    (meta.layers || []).forEach(L => {
      if (!L.on) return;
      chain(L.points, L.colour, L.closed);
      L.points.forEach(q => dot(q, L.colour));
    });
  }
  if (meta.mode === "box" && box) {
    const [x0, y0] = toScr(box[0], box[1]), [x1, y1] = toScr(box[2], box[3]);
    g.strokeStyle = "#58a6ff"; g.lineWidth = 2; g.strokeRect(x0, y0, x1 - x0, y1 - y0);
    dot([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], "#ff7b72");
  }
  loupe(); status();
}

function labelOf(p, i) {
  if (meta.bars) return i % 2 ? "" : `bar ${i / 2 + 1}: ${p.len_mm} mm`;
  return p.world_mm ? `${i + 1}: ${p.world_mm[0]}, ${p.world_mm[1]} mm` : String(i + 1);
}

function loupe() {
  const [ix, iy] = toImg(mouse[0], mouse[1]), Z = 10, half = lo.width / (2 * Z);
  lg.imageSmoothingEnabled = false;
  lg.fillStyle = "#000"; lg.fillRect(0, 0, lo.width, lo.height);
  lg.drawImage(img, ix - half, iy - half, 2 * half, 2 * half, 0, 0, lo.width, lo.height);
  lg.strokeStyle = "#ffd33d"; lg.lineWidth = 1;
  lg.beginPath(); lg.moveTo(lo.width / 2, 0); lg.lineTo(lo.width / 2, lo.height);
  lg.moveTo(0, lo.height / 2); lg.lineTo(lo.width, lo.height / 2); lg.stroke();
}

function status() {
  const [ix, iy] = toImg(mouse[0], mouse[1]);
  let s = `${ix.toFixed(1)}, ${iy.toFixed(1)} px · ${zoom.toFixed(2)}x · snap ${snap ? "ON" : "off"}`;
  if (meta.mode === "box" && box) {
    const w = Math.abs(box[2] - box[0]), h = Math.abs(box[3] - box[1]);
    s += ` · box ${w.toFixed(0)}x${h.toFixed(0)} px`;
    if (meta.mmPerPx) s += ` = ${(w * meta.mmPerPx).toFixed(0)}x${(h * meta.mmPerPx).toFixed(0)} mm`;
  }
  if (meta.mode === "shapes")
    s += cur.length ? ` · ${cur.length} pt open — p / l / g to finish` : " · no shape open";
  if (meta.mode === "points" && meta.bars)
    s += pts.length % 2 ? " · one end down — click the other"
                        : ` · ${pts.length / 2} bar(s)`;
  document.getElementById("stat").textContent = s;
  const L = document.getElementById("list");
  const rows = meta.mode === "shapes"
    ? shapes.map(s => `${s.name} · ${s.type} · ${s.points_px.length} pt`)
    : (meta.mode === "points" ? pts.map((p, i) => labelOf(p, i)) : []);
  L.style.display = rows.length ? "block" : "none";
  L.innerHTML = rows.map(r => `<div>${r}</div>`).join("");
}

async function place(sx, sy) {
  let [ix, iy] = toImg(sx, sy);
  if (snap) { const r = await post("/snap", {x: ix, y: iy}); ix = r.x; iy = r.y; }
  if (meta.mode === "points") {
    if (meta.n && pts.length >= meta.n) return;
    const p = {px: [ix, iy]};
    if (meta.world) {
      const t = entry().value.trim();
      if (!t) return say("type this mark's world X,Y in the box first, then click it", 1);
      const v = t.split(",").map(Number);
      if (v.length !== 2 || v.some(isNaN)) return say("need two numbers, e.g.  1800, 0", 1);
      p.world_mm = v;
      takeEntry();
    } else if (meta.bars && pts.length % 2 === 0) {
      // The length belongs to the pair, and is read once, at the end that opens
      // it — so `u` takes the length back out with the click that carried it.
      const t = entry().value.trim();
      if (!t) return say("type the taped length in mm in the box first, then click both ends", 1);
      const v = Number(t);
      if (!isFinite(v) || v <= 0) return say("need one positive length in mm, e.g.  10000", 1);
      p.len_mm = v;
      takeEntry();
    }
    pts.push(p);
    if (meta.world) say(`point ${pts.length} at ${p.world_mm[0]}, ${p.world_mm[1]} mm`);
    if (meta.bars) say(pts.length % 2 ? `bar ${(pts.length + 1) / 2} open — click the other end`
                                      : `bar ${pts.length / 2} closed`);
  } else if (meta.mode === "shapes") {
    cur.push([ix, iy]);
  }
  draw();
}

function finishShape(type) {
  if (!cur.length) return say("click at least one point on the image first", 1);
  if (type !== "point" && cur.length < 2) return say(`a ${type} needs at least two points`, 1);
  if (type === "polygon" && cur.length < 3) return say("a polygon needs at least three points", 1);
  const name = takeEntry() || `roi${shapes.length + 1}`;
  shapes.push({name, type, points_px: cur});
  cur = []; draw();
  say(`"${name}" saved — ${shapes.length} shape(s) so far`);
}

cv.addEventListener("mousedown", e => {
  mouse = [e.offsetX, e.offsetY];
  if (e.button === 1 || e.shiftKey) { drag = [e.offsetX, e.offsetY]; return; }
  if (e.button === 2) return;
  if (meta.mode === "review") return;             // look, do not edit
  if (meta.mode === "box") { const p = toImg(e.offsetX, e.offsetY); box = [p[0], p[1], p[0], p[1]];
                             drag = "box"; return; }
  place(e.offsetX, e.offsetY);
});
cv.addEventListener("mousemove", e => {
  mouse = [e.offsetX, e.offsetY];
  if (drag === "box") { const p = toImg(e.offsetX, e.offsetY); box[2] = p[0]; box[3] = p[1]; }
  else if (drag) { cx -= (e.offsetX - drag[0]) / zoom; cy -= (e.offsetY - drag[1]) / zoom;
                   drag = [e.offsetX, e.offsetY]; }
  draw();
});
addEventListener("mouseup", () => { drag = null; });
cv.addEventListener("contextmenu", e => {
  e.preventDefault();
  if (meta.mode === "shapes" && cur.length) cur.pop();
  else if (meta.mode === "points") pts.pop();
  draw();
});
cv.addEventListener("wheel", e => {
  e.preventDefault();
  const before = toImg(e.offsetX, e.offsetY);
  zoom = Math.min(60, Math.max(0.02, zoom * (e.deltaY < 0 ? 1.25 : 0.8)));
  const after = toImg(e.offsetX, e.offsetY);
  cx += before[0] - after[0]; cy += before[1] - after[1];
  draw();
}, {passive: false});

addEventListener("keydown", e => {
  if (e.target && e.target.id === "entry") {   // typing a name, not driving the tool
    if (e.key === "Escape") entry().blur();
    return;
  }
  const step = e.shiftKey ? 1 : 0.25;
  const last = meta.mode === "shapes" ? (cur.length ? cur : null)
             : (meta.mode === "points" && pts.length ? pts[pts.length - 1].px : null);
  const tgt = Array.isArray(last) && Array.isArray(last[0]) ? last[last.length - 1] : last;
  if (e.key === "Enter") { save(); }
  else if (e.key === "Escape") { bye("/cancel", "cancelled"); }
  else if (e.key === "s") { snap = !snap; draw(); }
  else if (e.key === "r") { reset(); }
  else if (e.key === "u") { if (meta.mode === "shapes") { cur.length ? cur.pop() : shapes.pop(); }
                            else if (meta.mode === "points") pts.pop(); else box = null; draw(); }
  else if (meta.mode === "shapes" && e.key === "p") finishShape("point");
  else if (meta.mode === "shapes" && e.key === "l") finishShape("line");
  else if (meta.mode === "shapes" && e.key === "g") finishShape("polygon");
  else if (tgt && e.key.startsWith("Arrow")) {
    e.preventDefault();
    if (e.key === "ArrowLeft") tgt[0] -= step; if (e.key === "ArrowRight") tgt[0] += step;
    if (e.key === "ArrowUp") tgt[1] -= step;   if (e.key === "ArrowDown") tgt[1] += step;
    draw();
  }
});

async function save() {
  if (meta.mode === "review") { await post("/done", {ok: true}); return bye(null, "accepted"); }
  if (meta.mode === "points") {
    if (meta.n && pts.length !== meta.n) return say(`need exactly ${meta.n} points`, 1);
    if (!pts.length) return say("nothing picked", 1);
    if (meta.bars && pts.length % 2)
      return say("the last bar has only one end — click the other, or press u", 1);
    await post("/done", {points: pts});
  } else if (meta.mode === "box") {
    if (!box) return say("drag a box first", 1);
    await post("/done", {box});
  } else {
    if (cur.length) return say(`${cur.length} point(s) still open — press p, l or g`
                               + " (or use the buttons) to finish that shape first", 1);
    if (!shapes.length) return say("nothing drawn", 1);
    await post("/done", {shapes});
  }
  bye(null, "saved");
}
for (const b of document.querySelectorAll("#shapebtns button"))
  b.onclick = () => finishShape(b.dataset.t);
document.getElementById("ok").onclick = save;
document.getElementById("no").onclick = () => bye("/cancel", "cancelled");

fetch("/meta" + Q).then(r => r.json()).then(m => {
  meta = m;
  document.getElementById("title").textContent = m.title;
  document.getElementById("hint").textContent = m.hint || "";
  if (m.mode === "shapes" || m.world || m.bars) {
    entry().style.display = "";
    entry().placeholder = m.bars ? "taped length in mm — then click both ends"
                        : m.world ? "world X, Y in mm — then click the mark"
                                  : "name for the next shape (optional)";
  }
  if (m.mode === "review") {
    const host = document.getElementById("layerbtns");
    host.style.display = "";
    m.layers.forEach(L => {
      const b = document.createElement("button");
      b.textContent = L.name;
      b.className = L.on ? "" : "g";
      b.onclick = () => { L.on = !L.on; b.className = L.on ? "" : "g"; draw(); };
      host.appendChild(b);
    });
    document.getElementById("ok").textContent = "Looks right — save (Enter)";
    document.getElementById("no").textContent = "Redo (Esc)";
  }
  if (m.mode === "shapes") document.getElementById("shapebtns").style.display = "";
  if (m.mode === "shapes")
    document.getElementById("hint").textContent =
      (m.hint ? m.hint + " · " : "") + "finish a shape: p point · l line · g polygon";
  snap = m.snap !== false;
  img.onload = () => { resize(); reset(); if (focusView()) draw(); };
  img.src = "/image.png" + Q;
});
addEventListener("resize", resize);
</script>
"""
