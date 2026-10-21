import xcs_code_stockpile as xcs

# %%
proposal_number = 101927227  # noqa
run = 27  # noqa

# %%
h5 = xcs.load_smalldata(run, proposal_number)
h5dict = xcs.H5Dictionary(
    stack='/jungfrau1M/ROI_area',
    i0=lambda h5: h5['diodeGon/channels'][..., 0]
)

# %%
metadata = xcs.create_metadata(h5, run, h5dict, roi=None)
metadata['laser_on'] = [False for i in metadata['laser_on']]
valid = xcs.sensible_filter(metadata, 0.5, 99.5, 0.5, 99.5)

# %%
xcs.plot_detector_image(h5, h5dict, samplerate=100).show()

# %%
xcs.plot_i0_vs_time(metadata, valid).show()

# %%
xcs.plot_intensity_vs_i0(metadata, valid).show()

# %%
xcs.plot_intensity_vs_time(metadata, valid).show()
