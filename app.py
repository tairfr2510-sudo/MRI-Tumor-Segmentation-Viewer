"""
MRAI Engine — Web Demo
Browser port of the MRAI Tumor Segmentation & 3D-Export Engine (MriFINAL.py).
Reuses ai_engine.py / DSP_Engine.py unchanged. The imaging pipeline (slicing,
rotation, crosshairs, range mask, sharpening, STL export, PDF report, DSP)
mirrors MriFINAL.py's logic and visual theme exactly; only the Tkinter/matplotlib
widget layer is replaced with Gradio so it can run in a browser for multiple
concurrent visitors instead of one desktop session.
"""
import datetime
import tempfile

import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
from skimage import measure
from skimage.filters import unsharp_mask
from fpdf import FPDF
import stl.mesh as stl_mesh

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import gradio as gr

import ai_engine
import DSP_Engine

try:
    import spaces  # provided by the HF Spaces ZeroGPU runtime; absent locally
except ImportError:
    class _NoOpGPU:
        def __call__(self, duration=None):
            def decorator(fn):
                return fn
            return decorator

    spaces = type("spaces", (), {"GPU": _NoOpGPU()})()

# ======================
# THEME — identical palette to MriFINAL.py
# ======================
THEME = {
    "bg": "#121212",
    "panel_bg": "#1E1E1E",
    "fg": "#E0E0E0",
    "accent": "#00E5FF",
    "button_bg": "#333333",
    "button_hover": "#4D4D4D",
    "slider_track": "#404040",
    "success": "#00C853",
}
CROSS_COLORS = ["#00E5FF", "#FF4081", "#76FF03"]  # sagittal / coronal / axial reference lines
AI_COLOR = "#FF4081"
MASK_COLOR = "#FF5722"
AI_BTN_COLOR = "#6200EA"
DSP_BTN_COLOR = "#FF9800"

SAMPLE_FILE = "BraTS20_Validation_089_t1ce.nii"
BLANK = np.zeros((420, 420, 3), dtype=np.uint8)
BLANK[:] = (18, 18, 18)


# ======================
# Geometry helpers — ported verbatim from MriFINAL.py
# ======================
def transform_coords(r, c, h, w, k):
    k = k % 4
    if k == 0:
        return r, c
    elif k == 1:
        return w - 1 - c, r
    elif k == 2:
        return h - 1 - r, w - 1 - c
    else:
        return c, h - 1 - r


def get_slice_dims(axis, sx, sy, sz):
    if axis == "axial":
        return sx, sy
    if axis == "coronal":
        return sx, sz
    return sy, sz


def new_state():
    return {
        "data": None,
        "voxel_size": 1.0,
        "shape": (0, 0, 0),
        "rot": {"axial": 3, "coronal": 3, "sagittal": 3},
        "zoom": 1.0,
        "mask_on": False,
        "sharpen_on": False,
        "show_ai_mask": True,
        "ai_mask": None,
        "ai_confidence": 0.0,
        "ai_range": None,
        "ai_peak_z": -1,
        "clinical_volume_cm3": 0.0,
    }


def status_html(text, color, size="1.05em"):
    return f"<div style='text-align:center;color:{color};font-size:{size};font-weight:bold;line-height:1.4'>{text}</div>"


READY_HTML = status_html("AI STATUS: READY — load a scan and run analysis", THEME["accent"])


# ======================
# Data loading
# ======================
def load_volume(path):
    nii = nib.load(path)
    raw = nii.get_fdata()
    vmin, vmax = raw.min(), raw.max()
    data = (raw - vmin) / (vmax - vmin) if vmax > vmin else raw
    try:
        voxel_size = float(nii.header.get_zooms()[0])
    except Exception:
        voxel_size = 1.0
    return data.astype(np.float32), voxel_size


def _load_from_path(path):
    if not path:
        raise gr.Error("No file selected.")
    data, voxel_size = load_volume(path)
    sx, sy, sz = data.shape
    state = new_state()
    state.update({"data": data, "voxel_size": voxel_size, "shape": (sx, sy, sz)})
    load_status = f"Loaded volume {sx}x{sy}x{sz} | voxel size ~{voxel_size:.2f} mm"
    return (
        state,
        load_status,
        gr.update(minimum=0, maximum=sz - 1, value=sz // 2),
        gr.update(minimum=0, maximum=sy - 1, value=sy // 2),
        gr.update(minimum=0, maximum=sx - 1, value=sx // 2),
        READY_HTML,
        "Show Mask",
        "Sharpen: OFF",
        "Hide AI",
    )


def on_load(file_obj):
    return _load_from_path(file_obj.name if file_obj is not None else None)


def on_load_sample():
    return _load_from_path(SAMPLE_FILE)


# ======================
# Rendering — dark theme, origin='lower', crosshairs, mask & AI contours
# (mirrors refresh()/update_masks()/update_ai_mask()/update_all_crosshairs() in MriFINAL.py)
# ======================
def render_plane(img, title, crosshair, mask_range, ai_overlay, zoom, cmap_name):
    rows, cols = img.shape
    fig, ax = plt.subplots(figsize=(4.3, 4.3), dpi=100)
    fig.patch.set_facecolor(THEME["bg"])
    ax.set_facecolor(THEME["bg"])
    ax.imshow(img, cmap=cmap_name, origin="lower", aspect="equal", vmin=0, vmax=1,
              extent=[-0.5, cols - 0.5, -0.5, rows - 0.5])

    if crosshair is not None:
        r, c, col_h, col_v = crosshair
        if zoom > 1.01:
            half_w, half_h = (cols / zoom) / 2, (rows / zoom) / 2
            ax.set_xlim(c - half_w, c + half_w)
            ax.set_ylim(r - half_h, r + half_h)
        else:
            ax.set_xlim(-0.5, cols - 0.5)
            ax.set_ylim(-0.5, rows - 0.5)
        ax.axhline(r, color=col_h, lw=1, alpha=0.85)
        ax.axvline(c, color=col_v, lw=1, alpha=0.85)
    else:
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(-0.5, rows - 0.5)

    if mask_range is not None:
        mn, mx = mask_range
        m = ((img >= mn) & (img <= mx)).astype(float)
        if m.max() > 0:
            ax.contour(m, levels=[0.5], colors=MASK_COLOR, linewidths=1.4,
                       extent=[-0.5, cols - 0.5, -0.5, rows - 0.5])

    if ai_overlay is not None and ai_overlay.max() > 0:
        ax.contour(ai_overlay, levels=[0.5], colors=AI_COLOR, linewidths=1.8, linestyles="dashed",
                   extent=[-0.5, cols - 0.5, -0.5, rows - 0.5])

    ax.set_title(title, color=THEME["accent"], fontsize=13, fontweight="bold", pad=8)
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def _get_slice(data, axis, idx, sharpen_on, sharpen_radius, sharpen_amount):
    if axis == "axial":
        sl = data[:, :, idx]
    elif axis == "coronal":
        sl = data[:, idx, :]
    else:
        sl = data[idx, :, :]
    if sharpen_on:
        sl = np.clip(unsharp_mask(sl, radius=sharpen_radius, amount=sharpen_amount), 0, 1)
    return sl


def master_render(state, z, y, x, cmap_name, mask_min, mask_max, sharpen_radius, sharpen_amount):
    if not state or state.get("data") is None:
        return BLANK, BLANK, BLANK, ""

    data = state["data"]
    sx, sy, sz = state["shape"]
    z, y, x = int(np.clip(z, 0, sz - 1)), int(np.clip(y, 0, sy - 1)), int(np.clip(x, 0, sx - 1))
    rot = state["rot"]

    img_a = np.rot90(_get_slice(data, "axial", z, state["sharpen_on"], sharpen_radius, sharpen_amount), k=rot["axial"])
    img_c = np.rot90(_get_slice(data, "coronal", y, state["sharpen_on"], sharpen_radius, sharpen_amount), k=rot["coronal"])
    img_s = np.rot90(_get_slice(data, "sagittal", x, state["sharpen_on"], sharpen_radius, sharpen_amount), k=rot["sagittal"])

    h, w = get_slice_dims("axial", sx, sy, sz)
    r_a, c_a = transform_coords(x, y, h, w, rot["axial"])
    h, w = get_slice_dims("coronal", sx, sy, sz)
    r_c, c_c = transform_coords(x, z, h, w, rot["coronal"])
    h, w = get_slice_dims("sagittal", sx, sy, sz)
    r_s, c_s = transform_coords(y, z, h, w, rot["sagittal"])

    mask_range = (mask_min, mask_max) if state["mask_on"] else None

    ai_overlay = None
    if state["show_ai_mask"] and state.get("ai_mask") is not None:
        ai_overlay = np.rot90(state["ai_mask"][:, :, z], k=rot["axial"])

    img_a_out = render_plane(img_a, "AXIAL", (r_a, c_a, CROSS_COLORS[0], CROSS_COLORS[1]),
                              mask_range, ai_overlay, state["zoom"], cmap_name)
    img_c_out = render_plane(img_c, "CORONAL", (r_c, c_c, CROSS_COLORS[0], CROSS_COLORS[2]),
                              mask_range, None, state["zoom"], cmap_name)
    img_s_out = render_plane(img_s, "SAGITTAL", (r_s, c_s, CROSS_COLORS[1], CROSS_COLORS[2]),
                              mask_range, None, state["zoom"], cmap_name)

    voxel_val = data[x, y, z]
    mask_txt = f"[{mask_min:.2f}-{mask_max:.2f}]" if state["mask_on"] else "OFF"
    info = (f"Voxel: [{x},{y},{z}] &nbsp;|&nbsp; Val: {voxel_val:.3f} &nbsp;|&nbsp; "
            f"Mask: {mask_txt} &nbsp;|&nbsp; Sharpen: {'ON' if state['sharpen_on'] else 'OFF'} "
            f"&nbsp;|&nbsp; Zoom: {state['zoom']:.1f}x")
    info_html = status_html(info, THEME["accent"], size="0.9em")

    return img_a_out, img_c_out, img_s_out, info_html


# ======================
# View control state mutators
# ======================
def bump_rot(state, axis):
    if state and state.get("data") is not None:
        state["rot"][axis] = (state["rot"][axis] + 1) % 4
    return state


def zoom_view(state, factor):
    if state and state.get("data") is not None:
        state["zoom"] = float(np.clip(state["zoom"] * factor, 1.0, 20.0))
    return state


def reset_view(state):
    if state and state.get("data") is not None:
        state["zoom"] = 1.0
    return state


def toggle_mask(state):
    if state and state.get("data") is not None:
        state["mask_on"] = not state["mask_on"]
    label = "Hide Mask" if (state and state.get("mask_on")) else "Show Mask"
    return state, gr.update(value=label)


def toggle_sharpen(state):
    if state and state.get("data") is not None:
        state["sharpen_on"] = not state["sharpen_on"]
    label = f"Sharpen: {'ON' if (state and state.get('sharpen_on')) else 'OFF'}"
    return state, gr.update(value=label)


def toggle_ai_overlay(state):
    if state and state.get("data") is not None:
        state["show_ai_mask"] = not state["show_ai_mask"]
    label = "Hide AI" if (state and state.get("show_ai_mask")) else "Show AI"
    return state, gr.update(value=label)


# ======================
# AI tumor detection
# ======================
@spaces.GPU(duration=60)
def _scan_volume(data, rot_k):
    """Runs on a ZeroGPU worker when available; plain-array args/return only."""
    import torch
    if torch.cuda.is_available() and ai_engine.tumor_ai is not None:
        ai_engine.device = torch.device("cuda")
        ai_engine.tumor_ai.to(ai_engine.device)
    return ai_engine.scan_full_volume(data, rot_k, progress_callback=None)


def run_ai_analysis(state, progress=gr.Progress()):
    if not state or state.get("data") is None:
        raise gr.Error("Load a scan first.")

    progress(0.2, desc="Scanning volume with 3D FCN...")
    has_tumor, confidence, t_range, peak_slice, mask, anchor = _scan_volume(state["data"], 0)
    progress(1.0, desc="Done")

    if has_tumor:
        state["ai_mask"] = mask
        state["ai_confidence"] = confidence
        state["ai_range"] = t_range
        state["ai_peak_z"] = peak_slice
        html = status_html(
            f"AI DETECTION: TUMOR FOUND | Confidence: {confidence:.1f}%<br>"
            f"Z-Slices: {t_range[0]}-{t_range[1]} | Peak at Z={peak_slice}", AI_COLOR)
        z_update = gr.update(value=peak_slice)
    else:
        state["ai_mask"] = None
        state["ai_confidence"] = 0.0
        state["ai_range"] = None
        state["ai_peak_z"] = -1
        html = status_html("AI ANALYSIS: SCAN COMPLETE<br>NO TUMOR DETECTED", THEME["success"])
        z_update = gr.update()

    return state, html, z_update


# ======================
# 3D STL export
# ======================
def generate_stl(state, mask_min, mask_max, source):
    if not state or state.get("data") is None:
        raise gr.Error("Load a scan first.")

    data = state["data"]
    if source == "AI Tumor Mask":
        if state.get("ai_mask") is None:
            raise gr.Error("Run AI analysis first — no tumor mask available.")
        binary_vol = state["ai_mask"].astype(bool)
        colorscale = [[0, "#3a0018"], [0.5, "#FF4081"], [1, "#ffd9e8"]]
    else:
        binary_vol = (data >= mask_min) & (data <= mask_max)
        colorscale = [[0, "#063a42"], [0.5, "#00E5FF"], [1, "#eafcff"]]

    if binary_vol.sum() == 0:
        raise gr.Error("Empty selection — adjust the intensity range sliders.")

    smooth_vol = gaussian_filter(binary_vol.astype(float), sigma=1)
    voxel = state["voxel_size"]
    verts, faces, _, _ = measure.marching_cubes(
        smooth_vol, level=0.5, step_size=1, spacing=(voxel, voxel, voxel)
    )

    mesh = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        mesh.vectors[i] = verts[f]

    tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
    mesh.save(tmp.name)

    volume_mm3, _, _ = mesh.get_mass_properties()
    volume_cm3 = abs(volume_mm3) / 1000.0
    state["clinical_volume_cm3"] = volume_cm3

    fig = go.Figure(data=[go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=verts[:, 2],
        colorscale=colorscale,
        showscale=False,
        opacity=1.0,
        flatshading=False,
        lighting=dict(ambient=0.42, diffuse=0.85, specular=0.45, roughness=0.45, fresnel=0.2),
        lightposition=dict(x=150, y=200, z=250),
    )])
    fig.update_layout(
        scene=dict(aspectmode="data",
                   xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                   bgcolor=THEME["bg"],
                   camera=dict(eye=dict(x=1.6, y=1.6, z=1.1))),
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["bg"],
        margin=dict(l=0, r=0, t=30, b=0),
        template="plotly_dark",
        title=dict(text=f"Volume: {volume_cm3:.2f} cm³", font=dict(color=THEME["accent"])),
    )

    status = status_html(f"3D GENERATED | VOLUME: {volume_cm3:.2f} cm³", THEME["success"])
    return state, fig, tmp.name, status


# ======================
# PDF report
# ======================
def generate_pdf(state, axial_img, coronal_img, sagittal_img):
    if not state or state.get("data") is None:
        raise gr.Error("Load a scan first.")

    from PIL import Image
    tmp_axial = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    tmp_coronal = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    tmp_sagittal = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    Image.fromarray(axial_img).save(tmp_axial)
    Image.fromarray(coronal_img).save(tmp_coronal)
    Image.fromarray(sagittal_img).save(tmp_sagittal)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, txt="MRI Clinical Analysis Report", ln=True, align="C")

    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, txt=f"Scan Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, txt="AI & Volumetric Analysis:", ln=True, align="L")
    pdf.set_font("Arial", "", 11)

    if state.get("ai_confidence"):
        pdf.cell(0, 10, txt=f"AI DETECTION: TUMOR FOUND | Confidence: {state['ai_confidence']:.1f}%", ln=True, align="L")
        pdf.cell(0, 10, txt=f"VOLUME: {abs(state.get('clinical_volume_cm3', 0.0)):.2f} cm3", ln=True, align="L")
    else:
        pdf.cell(0, 10, txt="No AI tumor detection recorded for this session.", ln=True, align="L")

    pdf.image(tmp_axial, x=10, y=80, w=60)
    pdf.image(tmp_coronal, x=75, y=80, w=60)
    pdf.image(tmp_sagittal, x=140, y=80, w=60)

    out_path = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name
    pdf.output(out_path)
    return out_path


# ======================
# DSP demo (FFT Gaussian high-pass, ported from open_dsp_window in MriFINAL.py)
# ======================
def dsp_render(state, z, cutoff):
    if not state or state.get("data") is None:
        return BLANK, BLANK
    data = state["data"]
    sx, sy, sz = state["shape"]
    z = int(np.clip(z, 0, sz - 1))
    current_slice = np.rot90(data[:, :, z], k=state["rot"]["axial"])
    edges = DSP_Engine.apply_fft_highpass(current_slice, cutoff)
    sharpened = np.clip(current_slice + 1.5 * edges, 0.0, 1.0)

    def to_rgb(img, title):
        fig, ax = plt.subplots(figsize=(4.3, 4.3), dpi=100)
        fig.patch.set_facecolor(THEME["bg"])
        ax.set_facecolor(THEME["bg"])
        ax.imshow(img, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax.set_title(title, color=THEME["accent"], fontsize=12, fontweight="bold")
        ax.axis("off")
        fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return buf

    return to_rgb(current_slice, "Original Slice"), to_rgb(sharpened, "FFT Sharpened MRI")


# ======================
# CSS — forces the exact MriFINAL.py dark palette across all Gradio components
# ======================
CSS = f"""
:root, .dark {{
    --body-background-fill: {THEME['bg']};
    --background-fill-primary: {THEME['bg']};
    --background-fill-secondary: {THEME['panel_bg']};
    --block-background-fill: {THEME['panel_bg']};
    --block-border-color: #333333;
    --panel-border-color: #333333;
    --body-text-color: {THEME['fg']};
    --body-text-color-subdued: #9A9A9A;
    --border-color-primary: #333333;
    --input-background-fill: #262626;
    --button-secondary-background-fill: {THEME['button_bg']};
    --button-secondary-background-fill-hover: {THEME['button_hover']};
    --button-secondary-text-color: {THEME['fg']};
    --color-accent: {THEME['accent']};
    --link-text-color: {THEME['accent']};
    --slider-color: {THEME['accent']};
}}
body, .gradio-container {{ background: {THEME['bg']} !important; font-family: 'Inter', 'Roboto', Arial, sans-serif !important; }}
h1, h2, h3, h4, p, span, label {{ color: {THEME['fg']}; font-family: 'Inter', 'Roboto', Arial, sans-serif !important; }}
#page-title h1 {{ color: {THEME['accent']} !important; letter-spacing: 1px; }}
#sidebar-col {{ background: {THEME['panel_bg']}; border-radius: 10px; padding: 14px; border: 1px solid #2a2a2a; }}
.block {{ border: 1px solid #262626 !important; }}
.section-header {{ color: {THEME['accent']} !important; font-weight: 700; margin: 6px 0 2px 0; letter-spacing: .5px; }}

/* action buttons: brand colors + soft drop shadow, per the exact spec */
button {{ box-shadow: 0 2px 5px rgba(0,0,0,0.45) !important; border: none !important; }}
#btn-ai button {{ background: {AI_BTN_COLOR} !important; color: white !important; font-weight: 700; }}
#btn-3d button {{ background: {THEME['success']} !important; color: white !important; font-weight: 700; }}
#btn-dsp button {{ background: {DSP_BTN_COLOR} !important; color: white !important; font-weight: 700; }}
#btn-pdf button {{ background: {THEME['accent']} !important; color: {THEME['bg']} !important; font-weight: 700; }}

/* sliders: grey track, white handle with cyan ring, cyan fill/value — per the exact spec */
input[type="range"] {{ accent-color: {THEME['accent']} !important; height: 6px !important; }}
input[type="range"]::-webkit-slider-runnable-track {{ background: #424242 !important; height: 6px !important; border-radius: 3px !important; }}
input[type="range"]::-webkit-slider-thumb {{
    -webkit-appearance: none; appearance: none;
    width: 16px !important; height: 16px !important; margin-top: -5px !important;
    border-radius: 50% !important; background: #FFFFFF !important;
    border: 2px solid {THEME['accent']} !important; box-shadow: 0 1px 3px rgba(0,0,0,0.6) !important;
}}
input[type="range"]::-moz-range-track {{ background: #424242 !important; height: 6px !important; border-radius: 3px !important; }}
input[type="range"]::-moz-range-thumb {{
    width: 16px !important; height: 16px !important; border-radius: 50% !important;
    background: #FFFFFF !important; border: 2px solid {THEME['accent']} !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.6) !important;
}}
input[type="number"] {{ color: {THEME['accent']} !important; font-weight: 700 !important; }}

/* fully hide scrollbars everywhere (content still scrolls, just no visible thumb/track) */
* {{ scrollbar-width: none !important; -ms-overflow-style: none !important; }}
*::-webkit-scrollbar {{ width: 0 !important; height: 0 !important; display: none !important; }}
.gradio-container {{ overflow-x: hidden !important; }}

/* tight, desktop-app-like spacing so everything fits one screen like MriFINAL.py's window */
.gradio-container {{ max-width: 100% !important; padding: 6px 10px !important; }}
#page-title {{ margin: 0 !important; padding: 0 !important; }}
#page-title h1 {{ font-size: 1em !important; margin: 0 !important; padding: 2px 0 !important; text-align: center; }}
.gr-row, .gr-column, .form {{ gap: 4px !important; }}
.block {{ padding: 4px !important; }}
label span {{ font-size: 0.8em !important; }}
button {{ min-height: 32px !important; padding: 4px 8px !important; }}
"""

# ======================
# UI
# ======================
with gr.Blocks(title="MRAI Tumor Segmentation Engine", theme=gr.themes.Base(), css=CSS) as demo:
    state = gr.State(new_state())

    with gr.Column(elem_id="page-title"):
        gr.Markdown("### BIOMED MRI VIEWER — MRAI Engine")

    info_html = gr.HTML(status_html("Load a scan to begin.", THEME["accent"], size="0.85em"))
    ai_status_out = gr.HTML(READY_HTML)

    with gr.Row():
        # ---------------- SIDEBAR — narrow left column, same order as MriFINAL.py's Figure window ----------------
        with gr.Column(scale=2, min_width=220, elem_id="sidebar-col"):
            sample_btn = gr.Button("Load Sample Case (BraTS20)")
            file_input = gr.File(label="Upload NIfTI (.nii / .nii.gz)", file_types=[".nii", ".gz"])
            load_status = gr.Textbox(label="Status", interactive=False)

            pdf_btn = gr.Button("Generate PDF", elem_id="btn-pdf")
            pdf_file = gr.File(label="Download PDF Report")

            gr.Markdown("Color Map", elem_classes="section-header")
            cmap_radio = gr.Radio(["gray", "hot", "viridis", "inferno"], value="gray", show_label=False)

            gr.Markdown("View Controls", elem_classes="section-header")
            with gr.Row():
                zin_btn = gr.Button("Z+")
                zout_btn = gr.Button("Z-")
            reset_btn = gr.Button("Reset View")

            stl_btn = gr.Button("GENERATE 3D STL", elem_id="btn-3d")
            stl_source = gr.Radio(
                ["Intensity Range", "AI Tumor Mask"], value="Intensity Range", show_label=False,
            )

            with gr.Row():
                ai_btn = gr.Button("RUN AI ANALYSIS", elem_id="btn-ai")
                ai_toggle_btn = gr.Button("Hide AI")

            dsp_toggle_btn = gr.Button("ADVANCED DSP (FFT)", elem_id="btn-dsp")

        # ---------------- MAIN VIEWER — images / rotate row / slider row / bottom toolbar, like the Figure window ----------------
        with gr.Column(scale=9):
            with gr.Tabs() as tabs:
                with gr.Tab("Viewer", id=0):
                    with gr.Row():
                        axial_out = gr.Image(label=None, show_label=False, interactive=False, height=380)
                        coronal_out = gr.Image(label=None, show_label=False, interactive=False, height=380)
                        sagittal_out = gr.Image(label=None, show_label=False, interactive=False, height=380)

                    with gr.Row():
                        rot_a_btn = gr.Button("Rot Ax", size="sm")
                        rot_c_btn = gr.Button("Rot Cor", size="sm")
                        rot_s_btn = gr.Button("Rot Sag", size="sm")

                    with gr.Row():
                        z_slider = gr.Slider(0, 10, value=5, step=1, label="Z-Slice")
                        y_slider = gr.Slider(0, 10, value=5, step=1, label="Y-Slice")
                        x_slider = gr.Slider(0, 10, value=5, step=1, label="X-Slice")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**RANGE SELECTOR**", elem_classes="section-header")
                            with gr.Row():
                                mask_toggle_btn = gr.Button("Show Mask")
                                mask_min = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Min")
                                mask_max = gr.Slider(0.0, 1.0, value=0.60, step=0.01, label="Max")
                        with gr.Column():
                            gr.Markdown("**ENHANCEMENT**", elem_classes="section-header")
                            with gr.Row():
                                sharpen_toggle_btn = gr.Button("Sharpen: OFF")
                                sharpen_radius = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="Radius")
                                sharpen_amount = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="Amount")

                with gr.Tab("3D Export", id=1):
                    stl_status = gr.HTML("")
                    with gr.Row():
                        stl_plot = gr.Plot(label="3D Preview")
                    stl_file = gr.File(label="Download STL")

                with gr.Tab("DSP (FFT)", id=2):
                    dsp_cutoff = gr.Slider(0.0, 0.5, value=0.2, step=0.01, label="Cutoff Radius")
                    with gr.Row():
                        dsp_before = gr.Image(label=None, show_label=False)
                        dsp_after = gr.Image(label=None, show_label=False)

    # ======================
    # Wiring
    # ======================
    view_inputs = [state, z_slider, y_slider, x_slider, cmap_radio, mask_min, mask_max,
                    sharpen_radius, sharpen_amount]
    view_outputs = [axial_out, coronal_out, sagittal_out, info_html]

    load_outputs = [state, load_status, z_slider, y_slider, x_slider, ai_status_out,
                     mask_toggle_btn, sharpen_toggle_btn, ai_toggle_btn]
    file_input.change(on_load, inputs=[file_input], outputs=load_outputs).then(
        master_render, inputs=view_inputs, outputs=view_outputs)
    sample_btn.click(on_load_sample, outputs=load_outputs).then(
        master_render, inputs=view_inputs, outputs=view_outputs)

    for ctrl in [z_slider, y_slider, x_slider, cmap_radio, mask_min, mask_max,
                 sharpen_radius, sharpen_amount]:
        ctrl.change(master_render, inputs=view_inputs, outputs=view_outputs)

    rot_a_btn.click(lambda s: bump_rot(s, "axial"), [state], [state]).then(
        master_render, view_inputs, view_outputs)
    rot_c_btn.click(lambda s: bump_rot(s, "coronal"), [state], [state]).then(
        master_render, view_inputs, view_outputs)
    rot_s_btn.click(lambda s: bump_rot(s, "sagittal"), [state], [state]).then(
        master_render, view_inputs, view_outputs)

    zin_btn.click(lambda s: zoom_view(s, 1.5), [state], [state]).then(
        master_render, view_inputs, view_outputs)
    zout_btn.click(lambda s: zoom_view(s, 1 / 1.5), [state], [state]).then(
        master_render, view_inputs, view_outputs)
    reset_btn.click(reset_view, [state], [state]).then(
        master_render, view_inputs, view_outputs)

    mask_toggle_btn.click(toggle_mask, [state], [state, mask_toggle_btn]).then(
        master_render, view_inputs, view_outputs)
    sharpen_toggle_btn.click(toggle_sharpen, [state], [state, sharpen_toggle_btn]).then(
        master_render, view_inputs, view_outputs)
    ai_toggle_btn.click(toggle_ai_overlay, [state], [state, ai_toggle_btn]).then(
        master_render, view_inputs, view_outputs)

    ai_btn.click(lambda: gr.Tabs(selected=0), outputs=[tabs]).then(
        run_ai_analysis, inputs=[state], outputs=[state, ai_status_out, z_slider]).then(
        master_render, view_inputs, view_outputs)

    stl_btn.click(lambda: gr.Tabs(selected=1), outputs=[tabs]).then(
        generate_stl, inputs=[state, mask_min, mask_max, stl_source],
        outputs=[state, stl_plot, stl_file, stl_status])

    pdf_btn.click(generate_pdf, inputs=[state, axial_out, coronal_out, sagittal_out], outputs=[pdf_file])

    dsp_toggle_btn.click(lambda: gr.Tabs(selected=2), outputs=[tabs]).then(
        dsp_render, inputs=[state, z_slider, dsp_cutoff], outputs=[dsp_before, dsp_after])
    dsp_cutoff.change(dsp_render, inputs=[state, z_slider, dsp_cutoff], outputs=[dsp_before, dsp_after])

if __name__ == "__main__":
    demo.queue().launch()
