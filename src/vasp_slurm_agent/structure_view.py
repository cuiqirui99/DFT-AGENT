"""Periodic display copies; calculation structures are never modified."""

from __future__ import annotations

from html import escape
from itertools import product
import json

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

from vasp_slurm_agent.numerical_defaults import vacuum_axes


COLORS = ("#507d9c", "#dc995b", "#6c9c76", "#ad79a7", "#cd6868", "#83983d", "#648fbc", "#a88664")


def display_repeat(structure: Structure) -> tuple[int, int, int]:
    """Repeat small cells without multiplying long vacuum directions."""
    if len(structure) > 24:
        return (1, 1, 1)
    vacuum = {item["axis"] for item in vacuum_axes(structure)}
    return tuple(1 if axis in vacuum else 2 for axis in range(3))


def display_structure(structure: Structure, repeat=None) -> Structure:
    shown = structure.copy()
    shown.make_supercell(display_repeat(structure) if repeat is None else repeat)
    return shown


def cell_edges(structure: Structure) -> np.ndarray:
    corners = np.asarray(list(product((0, 1), repeat=3)))
    coords = structure.lattice.get_cartesian_coords(corners)
    return np.asarray([coords[[i, j]] for i in range(8) for j in range(i + 1, 8)
                       if np.abs(corners[i] - corners[j]).sum() == 1])


def element_colors(structure: Structure) -> dict[str, str]:
    return {name: COLORS[index % len(COLORS)] for index, name in
            enumerate(sorted({str(site.specie.symbol) for site in structure}))}


def structure_figure(structure: Structure):
    """Static view with undistorted lengths and a separate orientation inset."""
    import matplotlib.pyplot as plt

    shown = display_structure(structure)
    colors = element_colors(shown)
    fig = plt.figure(figsize=(7.2, 4.8), facecolor="white")
    axis = fig.add_axes([0.01, 0.01, 0.81, 0.98], projection="3d")
    for edge in cell_edges(shown):
        axis.plot(*edge.T, color="#a8b4be", linewidth=0.8, alpha=0.85)
    for name, color in colors.items():
        positions = np.asarray([site.coords for site in shown if site.specie.symbol == name])
        axis.scatter(*positions.T, s=100, color=color, edgecolor="white", linewidth=0.5,
                     label=name, depthshade=True)
    bounds = np.concatenate([cell_edges(shown).reshape(-1, 3), shown.cart_coords])
    center = (bounds.min(axis=0) + bounds.max(axis=0)) / 2
    widths = np.maximum(np.ptp(bounds, axis=0) * 1.08, 0.1)
    for set_limit, midpoint, width in zip((axis.set_xlim, axis.set_ylim, axis.set_zlim), center, widths):
        set_limit(midpoint - width / 2, midpoint + width / 2)
    axis.set_box_aspect(widths, zoom=1.12)
    axis.set_proj_type("ortho")
    axis.view_init(elev=18, azim=-65)
    axis.set_axis_off()
    axis.legend(loc="upper left", bbox_to_anchor=(1.0, 0.95), frameon=False)
    orientation = fig.add_axes([0.82, 0.05, 0.17, 0.23], projection="3d")
    for name, vector, color in zip("abc", structure.lattice.matrix, ("#c45b58", "#4f936c", "#4c7da9")):
        vector = vector / np.linalg.norm(vector)
        orientation.quiver(0, 0, 0, *vector, color=color, arrow_length_ratio=0.15)
        orientation.text(*(vector * 1.18), name, color=color, fontsize=10)
    orientation.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), zlim=(-1.3, 1.3))
    orientation.set_box_aspect((1, 1, 1))
    orientation.set_proj_type("ortho")
    orientation.view_init(elev=18, azim=-65)
    orientation.set_axis_off()
    return fig


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")


def viewer_html(structure: Structure, script: str, fallback_script: str,
                message: str, height: int = 430) -> str:
    shown = display_structure(structure)
    repeat = display_repeat(structure)
    repeats = " × ".join(str(value) for value in repeat)
    colors = element_colors(structure)
    legend = "".join(f'<span class="element"><i style="background:{color}"></i>{escape(name)}</span>'
                     for name, color in colors.items())
    models = [{"poscar": Poscar(item).get_str(), "edges": cell_edges(item).tolist(), "count": len(item)}
              for item in (shown, structure)]
    vectors = structure.lattice.matrix / np.linalg.norm(structure.lattice.matrix, axis=1)[:, None]
    return f'''<style>
html,body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#425466;}}
.crystal{{border:1px solid #e3e8ed;border-radius:12px;overflow:hidden;background:#fff;}}
.toolbar{{padding:10px 14px;display:flex;align-items:center;gap:8px;border-bottom:1px solid #eef1f4;font-size:12px;}}
button{{background:#fff;border:1px solid #d4dde5;border-radius:6px;padding:5px 10px;color:#425466;cursor:pointer;}}
button.active{{background:#eaf1f7;border-color:#7c9bb4;color:#254c6c;}}
.note{{margin-left:auto;color:#687b8e;}}
.stage{{display:flex;height:{height - 48}px;}}
#dft-viewer{{position:relative;flex:1;min-width:0;}}
.side{{width:88px;display:flex;flex-direction:column;justify-content:space-between;padding:14px 4px 10px;}}
.element{{display:flex;align-items:center;gap:7px;font-size:12px;margin-bottom:9px;}}
.element i{{display:inline-block;width:11px;height:11px;border-radius:50%;}}
#orientation{{width:84px;height:94px;}}
@media(max-width:400px){{.side{{width:64px;}} #orientation{{width:64px;}} .toolbar{{padding:8px;}}}}
</style>
<div class="crystal">
<div class="toolbar"><button id="crystal" class="active">Crystal</button><button id="cell">Input cell</button><span id="view-note" class="note"></span></div>
<div class="stage"><div id="dft-viewer"></div><div class="side"><div>{legend}</div><svg id="orientation" viewBox="0 0 84 94" aria-label="Lattice directions"></svg></div></div>
</div>
<script>
(function() {{
  var models = {_json(models)}, colors = {_json(colors)}, vectors = {_json(vectors.tolist())};
  var box = document.getElementById("dft-viewer");
  var note = document.getElementById("view-note");
  var viewer, initialized = false;
  function orientation() {{
    var q = viewer.getView().slice(4,8);
    var svg = document.getElementById("orientation");
    var palette = ["#c45b58", "#4f936c", "#4c7da9"];
    var content = '<circle cx="40" cy="49" r="2" fill="#7d8994"/>';
    vectors.forEach(function(v, i) {{
      var u = q.slice(0,3), w = q[3];
      var dot = u[0]*v[0]+u[1]*v[1]+u[2]*v[2];
      var cross = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]];
      var norm = u[0]*u[0]+u[1]*u[1]+u[2]*u[2];
      var r = v.map(function(value, j) {{ return 2*dot*u[j]+(w*w-norm)*value+2*w*cross[j]; }});
      var x = 40+27*r[0], y = 49-27*r[1];
      content += '<line x1="40" y1="49" x2="'+x+'" y2="'+y+'" stroke="'+palette[i]+'" stroke-width="2"/>';
      content += '<text x="'+(40+35*r[0])+'" y="'+(53-35*r[1])+'" fill="'+palette[i]+'" font-size="12" text-anchor="middle">'+"abc"[i]+'</text>';
    }});
    svg.innerHTML = content;
  }}
  function draw(index) {{
    viewer.removeAllModels(); viewer.removeAllShapes();
    viewer.addModel(models[index].poscar, "vasp");
    Object.keys(colors).forEach(function(element) {{
      viewer.setStyle({{elem:element}}, {{sphere:{{scale:0.25,color:colors[element]}}}});
    }});
    models[index].edges.forEach(function(edge) {{
      viewer.addLine({{start:{{x:edge[0][0],y:edge[0][1],z:edge[0][2]}},end:{{x:edge[1][0],y:edge[1][1],z:edge[1][2]}},color:"#a8b4be",linewidth:1}});
    }});
    viewer.zoomTo(); viewer.zoom(window.innerWidth < 400 ? 1.75 : 1.35);
    if (!initialized) {{ viewer.rotate(25,"x"); viewer.rotate(32,"y"); initialized = true; }}
    viewer.render(); orientation();
    note.textContent = index === 0 ? {_json(repeats + " view · " + str(len(structure)) + " atoms in input")} : {_json(str(len(structure)) + " atoms in input")};
    document.getElementById("crystal").className = index === 0 ? "active" : "";
    document.getElementById("cell").className = index === 1 ? "active" : "";
  }}
  function start() {{
    try {{
      viewer = $3Dmol.createViewer(box, {{backgroundColor:"white",orthographic:true}});
      viewer.setViewChangeCallback(orientation);
      draw(0);
      document.getElementById("crystal").onclick = function() {{ draw(0); }};
      document.getElementById("cell").onclick = function() {{ draw(1); }};
      window.addEventListener("resize",function() {{ viewer.resize(); viewer.render(); }});
    }} catch(error) {{ box.innerText = {_json(message)}; }}
  }}
  function load(src, next) {{
    var script = document.createElement("script"); script.src = src;
    script.onload = start; script.onerror = next; document.head.appendChild(script);
  }}
  if (window.$3Dmol) {{ start(); }}
  else {{ load({_json(script)}, function() {{ load({_json(fallback_script)}, function() {{box.innerText = {_json(message)};}}); }}); }}
}})();
</script>'''
