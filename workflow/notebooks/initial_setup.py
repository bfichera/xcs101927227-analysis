import xcs_code_stockpile as xcs

# %%
proposal_number = {{proposal_number}}  # noqa
run = {{run}}  # noqa

# %%
h5 = xcs.load_smalldata(run, proposal_number)
h5dict = xcs.H5Dictionary(stack='/jungfrau1M/ROI_area')

# %%
metadata = xcs.create_metadata(h5, run, h5dict, roi=None)
valid = xcs.sensible_filter(metadata, 0.5, 99.5, 0.5, 99.5)

# %%
xcs.plot_i0_vs_time(metadata, valid)

# %%
xcs.plot_intensity_vs_i0(metadata, valid)

# %%
xcs.plot_intensity_vs_time(metadata, valid)
