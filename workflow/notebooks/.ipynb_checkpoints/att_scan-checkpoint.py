import xcs_code_stockpile as xcs
import numpy as np

# %%
proposal_number = {{proposal_number}}  # noqa
run = {{run}}  # noqa

# %%
h5 = xcs.load_smalldata(run, proposal_number)
h5dict = xcs.H5Dictionary(
    stack='/jungfrau1M/ROI_area',
    i0=lambda h5: h5['diodeGon/channels'][..., 0],
    laser_on=lambda h5: np.zeros(h5['event_time'].shape, dtype=bool),
    att='epics/att_T'
)

# %%
metadata = xcs.create_metadata(h5, run, h5dict, roi=None)
valid = xcs.sensible_filter(metadata, 0.5, 99.5, 0.5, 99.5)

# %%
xcs.plot_detector_image(h5, h5dict, samplerate=100).show()

# %%
xcs.plot_i0_vs_time(metadata, valid).show()

# %%
xcs.plot_intensity_vs_i0(metadata, valid).show()

# %%
xcs.plot_intensity_vs_time(metadata, valid).show()

# %%
metadata['att_round'] = np.exp(np.log(metadata['att']).round(decimals=1))
rocking = xcs.calc_rocking_stats(h5, h5dict, 'att_round', metadata, valid, roi=None)
plt.errorbar(rocking.index.to_numpy(), rocking['intensity_mean'].to_numpy(), rocking['intensity_stderr'].to_numpy())
plt.xlabel('att')
plt.ylabel('intensity / i0')
plt.show()
