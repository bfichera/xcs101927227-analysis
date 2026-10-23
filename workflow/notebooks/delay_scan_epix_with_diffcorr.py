import xcs_code_stockpile as xcs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# %%
def slow_mean(stack, valid, roi=None):
    if roi is None:
        roi = (slice(None), slice(None))
    n = 0
    out = np.zeros(stack[0, *roi].shape)
    for i in tqdm(list(range(stack.shape[0]))):
        if valid[i]:
            out += stack[i, *roi]
            n += 1
    return out / n


def slow_sum(stack, valid, roi=None, ret_n=False):
    if roi is None:
        roi = (slice(None), slice(None))
    n = 0
    out = np.zeros(stack[0, *roi].shape)
    for i in tqdm(list(range(stack.shape[0]))):
        if valid[i]:
            out += stack[i, *roi]
            n += 1
    if ret_n:
        return out, n
    return out

# %%
proposal_number = {{proposal_number}}  # noqa
run = {{run}}  # noqa

# %%
h5 = xcs.load_smalldata(run, proposal_number)
h5dict = xcs.H5Dictionary(
    stack='epix_5/ROI_area',
    i0=lambda h5: h5['diodeGon/channels'][..., 0],
    laser_on='evr/code_41',
    delay='scan/awg_delay_ns_delay',
    measured_voltage='ai/ch04'
)

# %%
rois = [
    (slice(50, 200), slice(100, 300)),   # The whole film peak
    (slice(130, 170), slice(180, 220)),  # Somewhere on the film peak
    (slice(490, 530), slice(280, 320))   # Superlattice peak
]
roi_colors = ['red', 'black', 'green']

# %%
metadata = xcs.create_metadata(h5, run, h5dict, roi=rois[0])
valid = xcs.no_filter(metadata)

# %%
xcs.plot_i0_vs_time(metadata, valid).show()

# %%
xcs.plot_intensity_vs_i0(metadata, valid).show()

# %%
xcs.plot_intensity_vs_time(metadata, valid).show()

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
plt.show()

# %%
voltage_high = metadata['measured_voltage'] > 1.0
laser_on = metadata['laser_on']
high_and_on = (voltage_high) & (laser_on)
off = (~voltage_high) & (~laser_on)
stack = h5[h5dict.stack]

# %%
mean_on = slow_mean(stack, high_and_on)
mean_off = slow_mean(stack, off)
diff = mean_on / mean_off - 1.0

# %%
fig, axd = plt.subplot_mosaic([['diff', 'on', 'off']], figsize=(12, 12))
axd['diff'].imshow(diff, cmap='RdBu_r', vmin=np.nanpercentile(diff, 1), vmax=np.nanpercentile(diff, 99))
axd['on'].imshow(mean_on, vmin=np.nanpercentile(mean_on, 1), vmax=np.nanpercentile(mean_on, 80))
axd['off'].imshow(mean_off, vmin=0.0, vmax=3.0)

for i, roi in enumerate(rois):
    axd['diff'].plot([roi[1].start, roi[1].stop], [roi[0].start, roi[0].start], color=roi_colors[i])
    axd['diff'].plot([roi[1].start, roi[1].stop], [roi[0].stop, roi[0].stop], color=roi_colors[i])
    axd['diff'].plot([roi[1].start, roi[1].start], [roi[0].start, roi[0].stop], color=roi_colors[i])
    axd['diff'].plot([roi[1].stop, roi[1].stop], [roi[0].start, roi[0].stop], color=roi_colors[i])
fig.show()

# %%
fig, axs = plt.subplots(len(rois), 1, sharex=True)
for i, (roi, ax) in enumerate(zip(rois, axs)):
    dist = np.ravel(h5[h5dict.stack][..., *roi])
    hist, bin_edges = np.histogram(dist, bins = np.arange(dist.min(), dist.max()), density=True)
    ax.semilogy(bin_edges[:-1], hist, label=f'roi {i}', color=roi_colors[i])
    ax.legend()
    ax.minorticks_on()
axs[-1].set_xlabel('ADU')
axs[1].set_ylabel('density')
fig.tight_layout()
fig.show()

# %%
for i, roi in enumerate(rois):
    shape = h5[h5dict.stack][0, *roi].shape
    sum_on_shot, n_shots = slow_sum(stack, high_and_on, roi=roi, ret_n=True)
    adu_per_pixel_per_shot = np.sum(sum_on_shot) / n_shots / shape[0] / shape[1]
    ph_per_pixel_per_shot = adu_per_pixel_per_shot / 9
    print(roi)
    print(ph_per_pixel_per_shot)

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
plt.show()

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
plt.show()

# %%

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
    fig.suptitle(f'Run {run:04d}, roi {j}, off correlation')
    fig.tight_layout()
    fig.show()

# %%
# TODO
# Try other reversibility conditions?
comparable_labtime_frames = 500
per_frame_time = (metadata['event_time'].max() - metadata['event_time'].min())/len(metadata)
labtime_bin_interval = comparable_labtime_frames * per_frame_time
for j, (roi, color) in enumerate(zip(rois, roi_colors)):
    diffcorr = xcs.differential_correlation('strong',
                                            h5,
                                            h5dict,
                                            metadata,
                                            valid,
                                            typical_dt,
                                            labtime_bin_interval,
                                            roi=roi,
                                            denominator_type='nanmean')
    fig, ax = plt.subplots()
    ax.plot(diffcorr.index, diffcorr)
    ax.set_xlabel('delay')
    ax.set_ylabel('diffcorr')
    fig.suptitle(f'Run {run:04d}, roi {j}, diffcorr')
    fig.tight_layout()
    fig.show()

# %%
comparable_labtime_frames = metadata.shape[0]
per_frame_time = (metadata['event_time'].max() - metadata['event_time'].min())/len(metadata)
labtime_bin_interval = comparable_labtime_frames * per_frame_time
twotimes = []
for j, (roi, color) in enumerate(zip(rois, roi_colors)):
    twotime = xcs.differential_twotime('strong', h5, h5dict, metadata, valid, typical_dt, labtime_bin_interval, roi=roi, denominator_type='nanmean')
    twotimes.append(twotime)

# %%
for j, (roi, color, twotime) in enumerate(zip(rois, roi_colors, twotimes)):
    fig, ax = plt.subplots()
    ax.imshow(twotime, vmin=0.0, vmax=np.max(twotime), origin='lower')
    ax.set_xlabel('delay$_1$')
    ax.set_ylabel('delay$_2$')
    fig.suptitle(f'Run {run:04d}, roi {j}, twodelay')
    fig.tight_layout()
    fig.show()
