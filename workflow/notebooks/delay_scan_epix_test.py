import xcs_code_stockpile as xcs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
proposal_number = 101927227   # noqa
run = 97  # noqa

# %%
h5 = xcs.load_smalldata(run, proposal_number)
h5dict = xcs.H5Dictionary(
    stack='epix_6/ROI_area',
    i0=lambda h5: h5['diodeGon/channels'][..., 0],
    laser_on='evr/code_41',
    delay='scan/awg_delay_ns_delay',
    measured_voltage='ai/ch04'
)

# %%
metadata = xcs.create_metadata(h5, run, h5dict, roi=None)
valid = xcs.sensible_filter(metadata, 0.5, 99.5, 0.5, 99.5)

# %%
on_meta = metadata[metadata['laser_on']]
off_meta = metadata[~metadata['laser_on']]
fig, axs = plt.subplots(1, 2)
for ax in axs:
    ax.plot(on_meta['measured_voltage'], label='on', color='red')
    ax.plot(off_meta['measured_voltage'], label='off', color='blue')
    ax.set_xlabel('frame')
    ax.set_ylabel('measured_voltage')
axs[1].set_ylim(bottom=0.0, top=0.3)
plt.legend()


# %%
rois = [
    (slice(160, 170), slice(168, 178)) # whole ROI
]

# %%
xcs.plot_detector_image(h5, h5dict, samplerate=100)
roi_colors = ['black', 'red', 'green', 'blue', 'purple', 'brown']
for i, roi in enumerate(rois):
    plt.plot([roi[1].start, roi[1].stop], [roi[0].start, roi[0].start], color=roi_colors[i])
    plt.plot([roi[1].start, roi[1].stop], [roi[0].stop, roi[0].stop], color=roi_colors[i])
    plt.plot([roi[1].start, roi[1].start], [roi[0].start, roi[0].stop], color=roi_colors[i])
    plt.plot([roi[1].stop, roi[1].stop], [roi[0].start, roi[0].stop], color=roi_colors[i])

# %%
xcs.plot_i0_vs_time(metadata, valid).show()

# %%
xcs.plot_intensity_vs_i0(metadata, valid).show()

# %%
xcs.plot_intensity_vs_time(metadata, valid).show()

# %%
delays = h5[h5dict.delay][...]
diff = np.diff(delays)
u_diffs, counts = np.unique(diff, return_counts=True)
typical_dt = u_diffs[u_diffs > 0][np.argmax(counts[u_diffs > 0])]
print(f'{typical_dt=:0.2f}')

# %%

fig, axd = plt.subplot_mosaic([['intensity_mean', 'com_i', 'com_j'], ['fwhm_i', 'fwhm_j', '_skip']])
for i, (tag, ax) in enumerate(axd.items()):
    if tag == '_skip':
        continue
    for j, (roi, color) in enumerate(zip(rois, roi_colors)):
        trxrd = xcs.calc_trxrd_stats(h5, h5dict, metadata, valid, typical_dt, roi=roi)
        label = f'on/off (roi {j})' if i == 0 else None
        ax.plot(trxrd.index, trxrd[f'on_{tag}']/trxrd[f'off_{tag}'], color=color, linestyle='-', label=label)
    ax.set_title(tag)
fig.legend(loc='lower right')
fig.suptitle(f'Run {run}')
fig.tight_layout()


# %%


def normalize(a):
    return a / np.amax(a)


fig, axd = plt.subplot_mosaic([['intensity_mean', 'com_i', 'com_j'], ['fwhm_i', 'fwhm_j', '_skip']])
for i, (tag, ax) in enumerate(axd.items()):
    if tag == '_skip':
        continue
    for j, (roi, color) in enumerate(zip(rois, roi_colors)):
        trxrd = xcs.calc_trxrd_stats(h5, h5dict, metadata, valid, typical_dt, roi=roi)
        label = f'off (roi {j})' if j == i else None
        ax.plot(trxrd.index, normalize(trxrd[f'off_{tag}']), color=color, linestyle=':', label=label)
    ax.set_title(tag)
fig.legend(loc='lower right')
fig.suptitle(f'Run {run}')
fig.tight_layout()


# %%

breakpoint()
off_filter = valid[~metadata['laser_on']]
off_metadata = metadata[~metadata['laser_on']]
for j, (roi, color) in enumerate(zip(rois, roi_colors)):
    off_st_corr, off_st_twotime = xcs.calc_static_correlation(h5, h5dict, off_metadata, off_filter, roi=roi, denominator_type='nanmean', ret_twotime=True)
    fig, axs = plt.subplots(1, 2)
    axs[0].semilogx(off_st_corr, label=f'roi {j}', color=color)
    axs[1].imshow(off_st_twotime, origin='lower')
    axs[0].set_xlabel(r'$\Delta$ frame#')
    axs[0].set_ylabel(r'$g_2(frame_1, frame_2) - 1$')
    axs[1].set_xlabel(r'frame$_1$')
    axs[1].set_ylabel(r'frame$_2$')
    axs[0].set_ylim(bottom=1.0)
    fig.suptitle(f'Run {run:04d}, roi {j}')
    fig.tight_layout()
    fig.show()
