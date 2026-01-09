import os
import numpy as np
import tifffile as tiff
import SimpleITK as sitk
import matplotlib.pyplot as plt
import napari  # pip install napari[all]

# ==========================================================
# 1) Paths
# ==========================================================
base_dir   = os.path.join(os.path.dirname(__file__), "..", "data")
atlas_path = os.path.join(base_dir, "atlas", "reference.tiff")
maldi_path = os.path.join(base_dir, "maldi", "5x_pc1_uint8.nii.gz")
out_dir    = os.path.join(os.path.dirname(__file__), "..", "result")
os.makedirs(out_dir, exist_ok=True)

print(f"\n📂 Atlas:  {atlas_path}\n📂 MALDI:  {maldi_path}\n📁 Output: {out_dir}")

# ==========================================================
# 2) Load
# ==========================================================
atlas_np = tiff.imread(atlas_path)                        # (Z,Y,X)
maldi_sitk_raw = sitk.ReadImage(maldi_path, sitk.sitkFloat32)
maldi_np_raw = sitk.GetArrayFromImage(maldi_sitk_raw)     # (Z,Y,X)

print("Atlas shape (Z,Y,X):", atlas_np.shape)
print("MALDI shape (Z,Y,X):", maldi_np_raw.shape)

# ==========================================================
# 3) Reorient to sagittal plane
# ==========================================================
atlas_sag = np.transpose(atlas_np, (2, 1, 0))      # (X,Y,Z)
maldi_sag = np.transpose(maldi_np_raw, (2, 1, 0))  # (X,Y,Z)
maldi_sag = np.flip(maldi_sag, axis=0)             # flip LR to match atlas

# ==========================================================
# 4) Crop hemisphere and normalize
# ==========================================================
def normalize(a):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, (1, 99))
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)

mid = atlas_sag.shape[0] // 2
atlas_half = atlas_sag[mid:, :, :]           # right hemisphere
atlas_half = np.flip(atlas_half, axis=0)     # flip to match MALDI

# --- offset from cross-correlation result (pre-computed) ---
offset = 20
nx_maldi = maldi_sag.shape[0]
nx_atlas = atlas_half.shape[0]

start = max(0, min(offset, nx_atlas - nx_maldi))
end   = start + nx_maldi
atlas_arr = atlas_half[start:end, :, :]
print(f"[Slice-map] Using atlas X-slices {start}:{end} -> {atlas_arr.shape} vs MALDI {maldi_sag.shape}")

atlas_arr = normalize(atlas_arr)
maldi_arr = normalize(maldi_sag)

# ==========================================================
# 5) Preview in Napari before registration
# ==========================================================
viewer = napari.Viewer()
viewer.add_image(maldi_arr, name="MALDI (Fixed)", colormap="gray",
                 blending="additive", opacity=0.7)
viewer.add_image(atlas_arr, name="Atlas (Cropped)", colormap="magenta",
                 blending="additive", opacity=0.5)
napari.run()

# ==========================================================
# 6) Pre-registration preview slice
# ==========================================================
x_m = maldi_arr.shape[0] // 2
x_a = atlas_arr.shape[0] // 2
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(atlas_arr[x_a, :, :], cmap='gray')
plt.title("Atlas Hemisphere (sagittal)")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(maldi_arr[x_m, :, :], cmap='gray')
plt.title("MALDI (sagittal)")
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "before_registration.png"),
            bbox_inches='tight')
plt.close()
print("🖼️ Saved sagittal before_registration.png")

# ==========================================================
# 7) Convert to SimpleITK
# ==========================================================
atlas_sitk = sitk.GetImageFromArray(atlas_arr)
atlas_sitk.SetSpacing((1,1,1))
atlas_sitk.SetOrigin((0,0,0))
atlas_sitk.SetDirection([1,0,0,0,1,0,0,0,1])

maldi_sitk = sitk.GetImageFromArray(maldi_arr)
maldi_sitk.SetSpacing((1,1,1))
maldi_sitk.SetOrigin((0,0,0))
maldi_sitk.SetDirection([1,0,0,0,1,0,0,0,1])

# ==========================================================
# 8) Histogram Matching
# ==========================================================
print("🧩 Histogram matching atlas to MALDI...")
hm = sitk.HistogramMatchingImageFilter()
hm.SetNumberOfHistogramLevels(256)
hm.SetNumberOfMatchPoints(15)
hm.ThresholdAtMeanIntensityOn()
atlas_sitk = hm.Execute(atlas_sitk, maldi_sitk)

# ==========================================================
# 9) Registration Pipeline
# ==========================================================
def extract_euler(tfm):
    if isinstance(tfm, sitk.CompositeTransform):
        if tfm.GetNumberOfTransforms() > 0:
            inner = tfm.GetNthTransform(0)
            if isinstance(inner, sitk.Euler3DTransform):
                return inner
            else:
                return sitk.Euler3DTransform(inner)
    return tfm

# --- Rigid ---
print("\n🚀 Stage 1: Rigid registration")
rig = sitk.ImageRegistrationMethod()
rig.SetMetricAsMattesMutualInformation(50)
rig.SetInterpolator(sitk.sitkLinear)
rig.SetOptimizerAsGradientDescent(1.0, 300, 1e-6, 10)
rig.SetOptimizerScalesFromPhysicalShift()
rig.SetShrinkFactorsPerLevel([4,2,1])
rig.SetSmoothingSigmasPerLevel([2,1,0])
rig.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
rig.SetInitialTransform(
    sitk.CenteredTransformInitializer(
        maldi_sitk, atlas_sitk, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS),
    inPlace=False)
rig_tfm = extract_euler(rig.Execute(fixed=maldi_sitk, moving=atlas_sitk))
print("✅ Rigid done")

# --- Affine ---
print("\n🚀 Stage 2: Affine registration")
aff_seed = sitk.AffineTransform(3)
aff_seed.SetMatrix(rig_tfm.GetMatrix())
aff_seed.SetTranslation(rig_tfm.GetTranslation())
aff_seed.SetCenter(rig_tfm.GetCenter())

aff = sitk.ImageRegistrationMethod()
aff.SetMetricAsMattesMutualInformation(50)
aff.SetInterpolator(sitk.sitkLinear)
aff.SetOptimizerAsGradientDescent(0.4, 200, 1e-6, 10)
aff.SetOptimizerScalesFromPhysicalShift()
aff.SetShrinkFactorsPerLevel([4,2,1])
aff.SetSmoothingSigmasPerLevel([2,1,0])
aff.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
aff.SetInitialTransform(aff_seed, inPlace=False)
aff_tfm = aff.Execute(fixed=maldi_sitk, moving=atlas_sitk)
print("✅ Affine done")

# ==========================================================
# 10) Global B-spline
# ==========================================================
print("\n🚀 Stage 3a: Global B-spline")
grid_mm = [60, 60, 60]
phys = [s*p for s,p in zip(maldi_sitk.GetSize(), maldi_sitk.GetSpacing())]
mesh = [max(1, int(phys[i]/grid_mm[i] + 0.5)) for i in range(3)]
bs_init = sitk.BSplineTransformInitializer(maldi_sitk, mesh, order=3)

nl = sitk.ImageRegistrationMethod()
nl.SetMetricAsMattesMutualInformation(50)
nl.SetInterpolator(sitk.sitkLinear)
nl.SetOptimizerAsGradientDescent(learningRate=0.5,
                                 numberOfIterations=120,
                                 convergenceMinimumValue=1e-6,
                                 convergenceWindowSize=10)
nl.SetOptimizerScalesFromPhysicalShift()
nl.SetShrinkFactorsPerLevel([4,2,1])
nl.SetSmoothingSigmasPerLevel([2,1,0])
nl.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
nl.SetMovingInitialTransform(aff_tfm)
nl.SetInitialTransform(bs_init, inPlace=False)
bs_global = nl.Execute(maldi_sitk, atlas_sitk)
print("✅ Global B-spline done")

# ==========================================================
# Helper: bake composite transform to displacement field
# ==========================================================
def bake_to_displacement_field(transform, reference_image,
                               pixel_type=sitk.sitkVectorFloat64):
    f = sitk.TransformToDisplacementFieldFilter()
    f.SetReferenceImage(reference_image)  # copies size/origin/spacing/direction
    f.SetOutputPixelType(pixel_type)
    return f.Execute(transform)

size = maldi_sitk.GetSize()
phys = [s*p for s,p in zip(size, maldi_sitk.GetSpacing())]

# ==========================================================
# 11) Stage 3b: Local B-spline (cerebellum / brainstem)
# ==========================================================
print("\n🎯 Stage 3b: Local B-spline refinement (cerebellum ROI)")

# chain so far: affine + global bspline
chain_cb = sitk.CompositeTransform([aff_tfm, bs_global])
df_cb = bake_to_displacement_field(chain_cb, maldi_sitk)
df_cb_tfm = sitk.DisplacementFieldTransform(df_cb)

# cerebellum + posterior region ROI
roi_cb_mask = sitk.Image(size, sitk.sitkUInt8)
roi_cb_mask.CopyInformation(maldi_sitk)
x0_cb = int(0.60 * size[0]); x1_cb = size[0]
y0_cb = int(0.25 * size[1]); y1_cb = int(0.95 * size[1])
z0_cb = int(0.05 * size[2]); z1_cb = int(0.95 * size[2])

roi_cb = sitk.RegionOfInterest(sitk.Image(size, sitk.sitkUInt8)+1,
                               size=[x1_cb-x0_cb, y1_cb-y0_cb, z1_cb-z0_cb],
                               index=[x0_cb, y0_cb, z0_cb])
roi_cb_mask = sitk.Paste(roi_cb_mask, roi_cb, roi_cb.GetSize(),
                         destinationIndex=[x0_cb, y0_cb, z0_cb])

grid_mm_cb = [35, 35, 35]
mesh_cb = [max(1, int(phys[i]/grid_mm_cb[i] + 0.5)) for i in range(3)]
bs_cb_init = sitk.BSplineTransformInitializer(maldi_sitk, mesh_cb, order=3)

nl_cb = sitk.ImageRegistrationMethod()
nl_cb.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
nl_cb.SetMetricFixedMask(roi_cb_mask)
nl_cb.SetInterpolator(sitk.sitkLinear)
nl_cb.SetOptimizerAsGradientDescent(learningRate=0.3,
                                    numberOfIterations=100,
                                    convergenceMinimumValue=1e-6,
                                    convergenceWindowSize=10)
nl_cb.SetOptimizerScalesFromPhysicalShift()
nl_cb.SetShrinkFactorsPerLevel([2,1])
nl_cb.SetSmoothingSigmasPerLevel([1,0])
nl_cb.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
nl_cb.SetMovingInitialTransform(df_cb_tfm)
nl_cb.SetInitialTransform(bs_cb_init, inPlace=False)

def _iter_cb():
    print(f"[CB] iter={nl_cb.GetOptimizerIteration():4d}  "
          f"metric={nl_cb.GetMetricValue():.6f}")
nl_cb.AddCommand(sitk.sitkIterationEvent, _iter_cb)

bs_cb = nl_cb.Execute(fixed=maldi_sitk, moving=atlas_sitk)
print("✅ Local B-spline (cerebellum) done")

# ==========================================================
# 12) Stage 3c: Local B-spline (hippocampus)
# ==========================================================
print("\n🎯 Stage 3c: Local B-spline refinement (hippocampus ROI)")

# chain so far: affine + global + cerebellum
chain_hip = sitk.CompositeTransform([aff_tfm, bs_global, bs_cb])
df_hip = bake_to_displacement_field(chain_hip, maldi_sitk)
df_hip_tfm = sitk.DisplacementFieldTransform(df_hip)

# hippocampus ROI (more anterior + mid-depth)
roi_hip_mask = sitk.Image(size, sitk.sitkUInt8)
roi_hip_mask.CopyInformation(maldi_sitk)
x0_hip = int(0.30 * size[0]); x1_hip = int(0.70 * size[0])
y0_hip = int(0.20 * size[1]); y1_hip = int(0.85 * size[1])
z0_hip = int(0.15 * size[2]); z1_hip = int(0.90 * size[2])

roi_hip = sitk.RegionOfInterest(sitk.Image(size, sitk.sitkUInt8)+1,
                                size=[x1_hip-x0_hip, y1_hip-y0_hip, z1_hip-z0_hip],
                                index=[x0_hip, y0_hip, z0_hip])
roi_hip_mask = sitk.Paste(roi_hip_mask, roi_hip, roi_hip.GetSize(),
                          destinationIndex=[x0_hip, y0_hip, z0_hip])

# ==========================================================
# 12) Stage 3d: Local B-spline Posterior Cerebellar Cortex ROI
# ==========================================================
print("\n🎯 Stage 3d: Local B-spline refinement (Posterior Cerebellar Cortex ROI)")

# --- Build ROI mask (same size as MALDI) ---
roi_post_cb_mask = sitk.Image(size, sitk.sitkUInt8)
roi_post_cb_mask.CopyInformation(maldi_sitk)

# Bounding box tailored to the circled region
x0_post = int(0.10 * size[0])   # posterior-left region
x1_post = int(0.38 * size[0])
y0_post = int(0.30 * size[1])   # upper cerebellar cortex
y1_post = int(0.75 * size[1])
z0_post = int(0.15 * size[2])   # medial depth
z1_post = int(0.85 * size[2])

roi_post = sitk.RegionOfInterest(sitk.Image(size, sitk.sitkUInt8) + 1,
                                 size=[x1_post - x0_post,
                                       y1_post - y0_post,
                                       z1_post - z0_post],
                                 index=[x0_post, y0_post, z0_post])

roi_post_cb_mask = sitk.Paste(roi_post_cb_mask,
                              roi_post,
                              roi_post.GetSize(),
                              destinationIndex=[x0_post,
                                                y0_post,
                                                z0_post])

# ==========================================================
# 12) Stage 3e: Local B-spline (Frontal Cortex / Olfactory ROI)
# ==========================================================
print("\n🎯 Stage 3e: Local B-spline refinement (Frontal Cortex / Olfactory ROI)")

roi_front_mask = sitk.Image(size, sitk.sitkUInt8)
roi_front_mask.CopyInformation(maldi_sitk)

x0_front = int(0.00 * size[0]); x1_front = int(0.30 * size[0])
y0_front = int(0.15 * size[1]); y1_front = int(0.80 * size[1])
z0_front = int(0.10 * size[2]); z1_front = int(0.90 * size[2])

roi_front = sitk.RegionOfInterest(sitk.Image(size, sitk.sitkUInt8) + 1,
                                  size=[x1_front - x0_front,
                                        y1_front - y0_front,
                                        z1_front - z0_front],
                                  index=[x0_front, y0_front, z0_front])
roi_front_mask = sitk.Paste(roi_front_mask, roi_front, roi_front.GetSize(),
                            destinationIndex=[x0_front, y0_front, z0_front])

# ==========================================================
# 12) Stage 3e\f: Local B-spline(Dorsal Cortex ROI)
# ==========================================================
print("\n🎯 Stage 3f: Define Dorsal Cortex ROI")

roi_dors_mask = sitk.Image(size, sitk.sitkUInt8)
roi_dors_mask.CopyInformation(maldi_sitk)

x0_dors = int(0.10 * size[0]);  x1_dors = int(0.90 * size[0])
y0_dors = int(0.00 * size[1]);  y1_dors = int(0.35 * size[1])
z0_dors = int(0.10 * size[2]);  z1_dors = int(0.90 * size[2])

roi_dors = sitk.RegionOfInterest(sitk.Image(size, sitk.sitkUInt8) + 1,
                                 size=[x1_dors - x0_dors,
                                       y1_dors - y0_dors,
                                       z1_dors - z0_dors],
                                 index=[x0_dors, y0_dors, z0_dors])

roi_dors_mask = sitk.Paste(roi_dors_mask, roi_dors, roi_dors.GetSize(),
                           destinationIndex=[x0_dors, y0_dors, z0_dors])


# ==========================================================
# 🔍 VISUALIZE MANUAL ROIs BEFORE REGISTRATION
# ==========================================================
print("\n🔍 Previewing manual ROIs in Napari...")

maldi_preview = sitk.GetArrayFromImage(maldi_sitk)
cb_mask_np    = sitk.GetArrayFromImage(roi_cb_mask)
hip_mask_np   = sitk.GetArrayFromImage(roi_hip_mask)
post_cb_mask_np = sitk.GetArrayFromImage(roi_post_cb_mask)
front_mask_np = sitk.GetArrayFromImage(roi_front_mask)


viewer_roi = napari.Viewer()

# Base image
viewer_roi.add_image(
    maldi_preview,
    name="MALDI",
    colormap="gray",
    opacity=0.7
)

# Cerebellum label layer
cb_layer = viewer_roi.add_labels(
    cb_mask_np.astype(np.uint8),
    name="Cerebellum ROI",
    opacity=0.4
)
cb_layer.color = {1: "cyan"}   # <-- Works in all versions

# Hippocampus label layer
hip_layer = viewer_roi.add_labels(
    hip_mask_np.astype(np.uint8),
    name="Hippocampus ROI",
    opacity=0.4
)
hip_layer.color = {1: "yellow"}   # <-- Same fix


# Post cb label layer
post_cb_layer = viewer_roi.add_labels(
    post_cb_mask_np.astype(np.uint8),
    name="Posterior Cerebellar ROI",
    opacity=0.40
)
post_cb_layer.color = {1: "green"}

# Frontal Cortex ROI (ROI4)
front_layer = viewer_roi.add_labels(
    front_mask_np.astype(np.uint8),
    name="Frontal Cortex ROI",
    opacity=0.40
)
front_layer.color = {1: "red"}   # NEW ROI4 color

# Dorsal Cortex ROI
dors_mask_np = sitk.GetArrayFromImage(roi_dors_mask)
dors_layer = viewer_roi.add_labels(
    dors_mask_np.astype(np.uint8),
    name="Dorsal Cortex ROI",
    opacity=0.40
)
dors_layer.color = {1: "blue"}   # suggested unique color


viewer_roi.scale_bar.visible = True
viewer_roi.scale_bar.unit = "µm"

napari.run()
# ==========================================================



grid_mm_hip = [25, 25, 25]  # finer grid for curved hippocampus
mesh_hip = [max(1, int(phys[i]/grid_mm_hip[i] + 0.5)) for i in range(3)]
bs_hip_init = sitk.BSplineTransformInitializer(maldi_sitk, mesh_hip, order=3)

nl_hip = sitk.ImageRegistrationMethod()
nl_hip.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
nl_hip.SetMetricFixedMask(roi_hip_mask)
nl_hip.SetInterpolator(sitk.sitkLinear)
nl_hip.SetOptimizerAsGradientDescent(learningRate=0.25,
                                     numberOfIterations=120,
                                     convergenceMinimumValue=1e-6,
                                     convergenceWindowSize=10)
nl_hip.SetOptimizerScalesFromPhysicalShift()
nl_hip.SetShrinkFactorsPerLevel([2,1])
nl_hip.SetSmoothingSigmasPerLevel([1,0])
nl_hip.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
nl_hip.SetMovingInitialTransform(df_hip_tfm)
nl_hip.SetInitialTransform(bs_hip_init, inPlace=False)

def _iter_hip():
    print(f"[HIP] iter={nl_hip.GetOptimizerIteration():4d}  "
          f"metric={nl_hip.GetMetricValue():.6f}")
nl_hip.AddCommand(sitk.sitkIterationEvent, _iter_hip)

bs_hip = nl_hip.Execute(fixed=maldi_sitk, moving=atlas_sitk)
print("✅ Local B-spline (hippocampus) done")

# ==========================================================
# 12) Stage 3d: Local B-spline (Posterior Cerebellar Cortex)
# ==========================================================
print("\n🚀 Stage 3d: Local B-spline refinement (Posterior Cerebellar Cortex)")

# chain so far: affine + global + cerebellum + hippocampus
chain_post = sitk.CompositeTransform([aff_tfm, bs_global, bs_cb, bs_hip])
df_post = bake_to_displacement_field(chain_post, maldi_sitk)
df_post_tfm = sitk.DisplacementFieldTransform(df_post)

# very fine warp for cerebellar folia
grid_mm_post = [18, 18, 18]
mesh_post = [max(1, int(phys[i] / grid_mm_post[i] + 0.5)) for i in range(3)]
bs_post_init = sitk.BSplineTransformInitializer(maldi_sitk, mesh_post, order=3)

nl_post = sitk.ImageRegistrationMethod()
nl_post.SetMetricAsMattesMutualInformation(60)
nl_post.SetMetricFixedMask(roi_post_cb_mask)
nl_post.SetInterpolator(sitk.sitkLinear)
nl_post.SetOptimizerAsGradientDescent(
    learningRate=0.20,
    numberOfIterations=110,
    convergenceMinimumValue=1e-6,
    convergenceWindowSize=10
)
nl_post.SetOptimizerScalesFromPhysicalShift()
nl_post.SetShrinkFactorsPerLevel([2, 1])
nl_post.SetSmoothingSigmasPerLevel([1, 0])
nl_post.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
nl_post.SetMovingInitialTransform(df_post_tfm)
nl_post.SetInitialTransform(bs_post_init, inPlace=False)

def _iter_post():
    print(f"[POST_CB] iter={nl_post.GetOptimizerIteration():4d}  "
          f"metric={nl_post.GetMetricValue():.7f}")
nl_post.AddCommand(sitk.sitkIterationEvent, _iter_post)

# RUN optimizer
bs_post_cb = nl_post.Execute(fixed=maldi_sitk, moving=atlas_sitk)
print("✅ Local B-spline (Posterior Cerebellar Cortex) done")

# ==========================================================
# 12) Stage 3e: Local B-spline (Frontal Cortex / Olfactory Region)
# ==========================================================
print("\n🚀 Stage 3e: Local B-spline refinement (Frontal Cortex / Olfactory)")

# chain so far + all previous refinements
chain_front = sitk.CompositeTransform([aff_tfm, bs_global, bs_cb, bs_hip, bs_post_cb])
df_front = bake_to_displacement_field(chain_front, maldi_sitk)
df_front_tfm = sitk.DisplacementFieldTransform(df_front)

# Finer deformation grid for frontal cortex
grid_mm_front = [20, 20, 20]
mesh_front = [max(1, int(phys[i]/grid_mm_front[i] + 0.5)) for i in range(3)]
bs_front_init = sitk.BSplineTransformInitializer(maldi_sitk, mesh_front, order=3)

nl_front = sitk.ImageRegistrationMethod()
nl_front.SetMetricAsMattesMutualInformation(60)
nl_front.SetMetricFixedMask(roi_front_mask)
nl_front.SetInterpolator(sitk.sitkLinear)
nl_front.SetOptimizerAsGradientDescent(
    learningRate=0.22,
    numberOfIterations=110,
    convergenceMinimumValue=1e-6,
    convergenceWindowSize=10
)
nl_front.SetOptimizerScalesFromPhysicalShift()
nl_front.SetShrinkFactorsPerLevel([2, 1])
nl_front.SetSmoothingSigmasPerLevel([1, 0])
nl_front.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
nl_front.SetMovingInitialTransform(df_front_tfm)
nl_front.SetInitialTransform(bs_front_init, inPlace=False)

def _iter_front():
    print(f"[FRONTAL] iter={nl_front.GetOptimizerIteration():4d}  "
          f"metric={nl_front.GetMetricValue():.7f}")
nl_front.AddCommand(sitk.sitkIterationEvent, _iter_front)

bs_front = nl_front.Execute(fixed=maldi_sitk, moving=atlas_sitk)
print("✅ Local B-spline (Frontal Cortex / Olfactory) done")


# ==========================================================
# 12) Stage 3f: Local B-spline (Dorsal Cortex)
# ==========================================================
print("\n🚀 Stage 3f: Local B-spline refinement (Dorsal Cortex)")

chain_dors = sitk.CompositeTransform([aff_tfm, bs_global, bs_cb, bs_hip, bs_post_cb, bs_front])
df_dors = bake_to_displacement_field(chain_dors, maldi_sitk)
df_dors_tfm = sitk.DisplacementFieldTransform(df_dors)

grid_mm_dors = [22, 22, 22]   # moderately fine network
mesh_dors = [max(1, int(phys[i]/grid_mm_dors[i] + 0.5)) for i in range(3)]
bs_dors_init = sitk.BSplineTransformInitializer(maldi_sitk, mesh_dors, order=3)

nl_dors = sitk.ImageRegistrationMethod()
nl_dors.SetMetricAsMattesMutualInformation(60)
nl_dors.SetMetricFixedMask(roi_dors_mask)
nl_dors.SetInterpolator(sitk.sitkLinear)
nl_dors.SetOptimizerAsGradientDescent(
    learningRate=0.22,
    numberOfIterations=110,
    convergenceMinimumValue=1e-6,
    convergenceWindowSize=10
)
nl_dors.SetOptimizerScalesFromPhysicalShift()
nl_dors.SetShrinkFactorsPerLevel([2,1])
nl_dors.SetSmoothingSigmasPerLevel([1,0])
nl_dors.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
nl_dors.SetMovingInitialTransform(df_dors_tfm)
nl_dors.SetInitialTransform(bs_dors_init, inPlace=False)

def _iter_dors():
    print(f"[DORSAL] iter={nl_dors.GetOptimizerIteration():4d}  "
          f"metric={nl_dors.GetMetricValue():.7f}")
nl_dors.AddCommand(sitk.sitkIterationEvent, _iter_dors)

bs_dors = nl_dors.Execute(fixed=maldi_sitk, moving=atlas_sitk)
print("✅ Local B-spline (Dorsal Cortex) done")


# ==========================================================
# 13) Final compose + resample
# ==========================================================
# final_tfm = sitk.CompositeTransform([aff_tfm, bs_global, bs_cb, bs_hip, bs_post_cb, bs_front, bs_dors, bs_vent])
final_tfm = sitk.CompositeTransform([aff_tfm, bs_global, bs_cb, bs_hip, bs_post_cb, bs_front, bs_dors])
final_res = sitk.Resample(atlas_sitk, maldi_sitk, final_tfm,
                          sitk.sitkLinear, 0.0)
sitk.WriteImage(final_res, os.path.join(out_dir, "atlas_warped_final.nii.gz"))
print("\n✅ Done. Registered output saved in:", out_dir)

# ==========================================================
# 14) Final Napari visualization
# ==========================================================
maldi_final_np = sitk.GetArrayFromImage(maldi_sitk)
atlas_final_np = sitk.GetArrayFromImage(final_res)

def normalize_for_view(a):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, (1, 99))
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)

maldi_final_np = normalize_for_view(maldi_final_np)
atlas_final_np = normalize_for_view(atlas_final_np)

viewer_final = napari.Viewer()
viewer_final.add_image(maldi_final_np, name="MALDI (Fixed, sagittal)",
                       colormap="gray", blending="additive", opacity=0.7)
viewer_final.add_image(atlas_final_np, name="Atlas Warped (Moving)",
                       colormap="magenta", blending="additive", opacity=0.5)
viewer_final.scale_bar.visible = True
viewer_final.scale_bar.unit = "µm"
napari.run()
