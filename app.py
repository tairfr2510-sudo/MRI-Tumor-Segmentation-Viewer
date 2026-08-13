"""
MRAI Engine — Web Demo
Browser-based demo of the MRAI Tumor Segmentation & 3D-Export Engine.
Reuses the AI (ai_engine.py) and DSP (DSP_Engine.py) modules from the desktop app;
the imaging/viewer logic below mirrors MriFINAL.py but is rewritten as stateless
functions driven by Gradio instead of Tkinter/matplotlib widgets.
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

SAMPLE_FILE = "BraTS20_Validation_089_t1ce.nii"
ROT_K = 3  # fixed display rotation, matches the desktop app's default orientation
MASK_MIN_DEFAULT, MASK_MAX_DEFAULT = 0.25, 0.60
BLANK = np.zeros((10, 10, 3), dtype=np.uint8)


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


def on_load(file_obj):
    path = file_obj.name if file_obj is not None else None
    return _load_from_path(path)


def on_load_sample():
    return _load_from_path(SAMPLE_FILE)


def _load_from_path(path):
    if not path:
        raise gr.Error("No file selected.")
    data, voxel_size = load_volume(path)
    sx, sy, sz = data.shape
    state = {
        "data": data,
        "voxel_size": voxel_size,
        "shape": (sx, sy, sz),
        "ai_mask": None,
        "ai_confidence": 0.0,
        "ai_range": None,
        "clinical_volume_cm3": 0.0,
    }
    status = f"Loaded volume {sx}x{sy}x{sz} | voxel size ~{voxel_size:.2f} mm"
    return (
        state,
        status,
        gr.update(minimum=0, maximum=sz - 1, value=sz // 2),
        gr.update(minimum=0, maximum=sy - 1, value=sy // 2),
        gr.update(minimum=0, maximum=sx - 1, value=sx // 2),
    )


# ======================
# Slicing + rendering
# ======================
def axial_slice(data, z):
    return data[:, :, z]


def coronal_slice(data, y):
    return data[:, y, :]


def sagittal_slice(data, x):
    return data[x, :, :]


def render_slice(slice2d, cmap_name, sharpen, sharpen_radius, sharpen_amount,
                  mask_min, mask_max, ai_mask_2d=None):
    img = np.rot90(slice2d, k=ROT_K)
    if sharpen:
        img = np.clip(unsharp_mask(img, radius=sharpen_radius, amount=sharpen_amount), 0, 1)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.imshow(img, cmap=cmap_name, vmin=0, vmax=1)

    range_mask = ((img >= mask_min) & (img <= mask_max)).astype(float)
    if range_mask.max() > 0:
        ax.contour(range_mask, levels=[0.5], colors="#FF5722", linewidths=1.2)

    if ai_mask_2d is not None and ai_mask_2d.max() > 0:
        ax.contour(ai_mask_2d, levels=[0.5], colors="#FF4081", linewidths=1.8, linestyles="dashed")

    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def update_view(state, z, y, x, cmap_name, sharpen, sharpen_radius, sharpen_amount,
                 mask_min, mask_max, show_ai):
    if not state or state.get("data") is None:
        return BLANK, BLANK, BLANK

    data = state["data"]
    sx, sy, sz = state["shape"]
    z = int(min(max(z, 0), sz - 1))
    y = int(min(max(y, 0), sy - 1))
    x = int(min(max(x, 0), sx - 1))

    ai_mask = state.get("ai_mask") if show_ai else None
    ai_slice = ai_mask[:, :, z] if ai_mask is not None else None

    axial_img = render_slice(axial_slice(data, z), cmap_name, sharpen, sharpen_radius,
                              sharpen_amount, mask_min, mask_max, ai_slice)
    coronal_img = render_slice(coronal_slice(data, y), cmap_name, sharpen, sharpen_radius,
                                sharpen_amount, mask_min, mask_max, None)
    sagittal_img = render_slice(sagittal_slice(data, x), cmap_name, sharpen, sharpen_radius,
                                 sharpen_amount, mask_min, mask_max, None)
    return axial_img, coronal_img, sagittal_img


# ======================
# AI tumor detection
# ======================
def run_ai_analysis(state, progress=gr.Progress()):
    if not state or state.get("data") is None:
        raise gr.Error("Load a scan first.")

    def cb(current, total):
        progress(current / total, desc="Scanning volume with 3D FCN...")

    has_tumor, confidence, t_range, peak_slice, mask, anchor = ai_engine.scan_full_volume(
        state["data"], ROT_K, progress_callback=cb
    )

    if has_tumor:
        state["ai_mask"] = mask
        state["ai_confidence"] = confidence
        state["ai_range"] = t_range
        status = (f"TUMOR DETECTED | Confidence: {confidence:.1f}% | "
                  f"Z-range: {t_range[0]}-{t_range[1]} | Peak slice: {peak_slice}")
        z_update = gr.update(value=peak_slice)
    else:
        state["ai_mask"] = None
        state["ai_confidence"] = 0.0
        state["ai_range"] = None
        status = "SCAN COMPLETE — No tumor mass detected."
        z_update = gr.update()

    return state, status, z_update, gr.update(value=True)


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
    else:
        binary_vol = (data >= mask_min) & (data <= mask_max)

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
        color="#00E5FF" if source != "AI Tumor Mask" else "#FF4081",
        opacity=0.9,
    )])
    fig.update_layout(
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=30, b=0),
        template="plotly_dark",
        title=f"Volume: {volume_cm3:.2f} cm³",
    )

    status = f"3D model generated | Volume: {volume_cm3:.2f} cm³"
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
# DSP demo (FFT high-pass)
# ======================
def dsp_preview(state, z, cutoff):
    if not state or state.get("data") is None:
        return BLANK, BLANK
    data = state["data"]
    sx, sy, sz = state["shape"]
    z = int(min(max(z, 0), sz - 1))
    original = np.rot90(axial_slice(data, z), k=ROT_K)
    filtered = DSP_Engine.apply_fft_highpass(original, cutoff)

    def to_rgb(img):
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return buf

    return to_rgb(original), to_rgb(filtered)


# ======================
# UI
# ======================
with gr.Blocks(title="MRAI Tumor Segmentation Engine") as demo:
    state = gr.State(None)

    gr.Markdown(
        """
        # MRAI — Tumor Segmentation & 3D-Export Engine (Web Demo)
        A 3D U-Net (Fully Convolutional Network) trained on BraTS scans automatically detects and
        contours brain tumors, computes true-scale physical volume, and exports 3D-printable STL meshes.
        Click **Load Sample Case** below for a zero-setup demo, or upload your own NIfTI (`.nii`/`.nii.gz`) file.

        *Developer: Tair Fridman — B.Sc. Biomedical Engineering, Technion*
        """
    )

    with gr.Row():
        file_input = gr.File(label="Upload NIfTI file (.nii / .nii.gz)", file_types=[".nii", ".gz"])
        with gr.Column():
            sample_btn = gr.Button("Load Sample Case (BraTS20)", variant="primary")
            load_status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        cmap = gr.Dropdown(["gray", "hot", "viridis", "inferno"], value="gray", label="Colormap")
        sharpen = gr.Checkbox(label="Sharpen (Unsharp Mask)", value=False)
        sharpen_radius = gr.Slider(0.5, 3.0, value=1.0, step=0.1, label="Sharpen Radius")
        sharpen_amount = gr.Slider(0.5, 3.0, value=1.0, step=0.1, label="Sharpen Amount")

    with gr.Row():
        mask_min = gr.Slider(0.0, 1.0, value=MASK_MIN_DEFAULT, step=0.01, label="Intensity Range Min")
        mask_max = gr.Slider(0.0, 1.0, value=MASK_MAX_DEFAULT, step=0.01, label="Intensity Range Max")
        show_ai = gr.Checkbox(label="Show AI Tumor Overlay", value=True)

    with gr.Row():
        z_slider = gr.Slider(0, 10, value=5, step=1, label="Axial (Z)")
        y_slider = gr.Slider(0, 10, value=5, step=1, label="Coronal (Y)")
        x_slider = gr.Slider(0, 10, value=5, step=1, label="Sagittal (X)")

    with gr.Row():
        axial_out = gr.Image(label="Axial", interactive=False)
        coronal_out = gr.Image(label="Coronal", interactive=False)
        sagittal_out = gr.Image(label="Sagittal", interactive=False)

    with gr.Row():
        ai_btn = gr.Button("Run AI Analysis", variant="primary")
        ai_status = gr.Textbox(label="AI Status", interactive=False)

    gr.Markdown("---\n## 3D Export & Volumetry")
    with gr.Row():
        stl_source = gr.Radio(
            ["Intensity Range (Full Brain / Selection)", "AI Tumor Mask"],
            value="Intensity Range (Full Brain / Selection)",
            label="Mesh Source",
        )
        stl_btn = gr.Button("Generate 3D STL", variant="primary")
    with gr.Row():
        stl_plot = gr.Plot(label="3D Preview")
        stl_file = gr.File(label="Download STL")
    stl_status = gr.Textbox(label="3D Status", interactive=False)

    gr.Markdown("---\n## Clinical PDF Report")
    pdf_btn = gr.Button("Generate PDF Report")
    pdf_file = gr.File(label="Download PDF")

    gr.Markdown("---\n## Frequency-Domain DSP — Gaussian High-Pass Filter (FFT)")
    with gr.Row():
        dsp_z = gr.Slider(0, 10, value=5, step=1, label="Slice (Z)")
        dsp_cutoff = gr.Slider(0.0, 1.0, value=0.1, step=0.01, label="Cutoff Radius")
    with gr.Row():
        dsp_before = gr.Image(label="Original")
        dsp_after = gr.Image(label="High-Pass Filtered")

    # ---- wiring ----
    view_inputs = [state, z_slider, y_slider, x_slider, cmap, sharpen, sharpen_radius,
                    sharpen_amount, mask_min, mask_max, show_ai]
    view_outputs = [axial_out, coronal_out, sagittal_out]

    file_input.change(on_load, inputs=[file_input],
                       outputs=[state, load_status, z_slider, y_slider, x_slider]).then(
        update_view, inputs=view_inputs, outputs=view_outputs
    )
    sample_btn.click(on_load_sample, outputs=[state, load_status, z_slider, y_slider, x_slider]).then(
        update_view, inputs=view_inputs, outputs=view_outputs
    )

    for ctrl in [z_slider, y_slider, x_slider, cmap, sharpen, sharpen_radius, sharpen_amount,
                 mask_min, mask_max, show_ai]:
        ctrl.change(update_view, inputs=view_inputs, outputs=view_outputs)

    ai_btn.click(run_ai_analysis, inputs=[state], outputs=[state, ai_status, z_slider, show_ai]).then(
        update_view, inputs=view_inputs, outputs=view_outputs
    )

    stl_btn.click(generate_stl, inputs=[state, mask_min, mask_max, stl_source],
                   outputs=[state, stl_plot, stl_file, stl_status])

    pdf_btn.click(generate_pdf, inputs=[state, axial_out, coronal_out, sagittal_out], outputs=[pdf_file])

    dsp_z.change(dsp_preview, inputs=[state, dsp_z, dsp_cutoff], outputs=[dsp_before, dsp_after])
    dsp_cutoff.change(dsp_preview, inputs=[state, dsp_z, dsp_cutoff], outputs=[dsp_before, dsp_after])

if __name__ == "__main__":
    demo.queue().launch()
