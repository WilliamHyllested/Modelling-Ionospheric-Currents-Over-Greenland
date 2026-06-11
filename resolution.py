#%% Import
import numpy as np
import scipy
#from secsy import get_SECS_B_G_matrices, get_SECS_J_G_matrices, CSgrid, CSprojection
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
import copy
#from scipy.optimize import curve_fit
#from scipy.interpolate import splrep, BSpline

def add_qd_gridlines_res(ax, apex_obj, projection,
                         qlat_lines=range(-10, 90, 10),
                         qlon_lines=range(-180, 181, 20),
                         n_points=500,
                         lat_color='blue', lon_color='blue',
                         linewidth=0.5, alpha=0.5, linestyle='--',
                         label_gridlines=True):
    """
    QD grid lines for plots that use grid.projection.geo2cube() (xi/eta space).
    Pass ax and grid.projection from plot_map.
    """

    # ── QD latitude lines ───────────────────────────────────────────────────
    qlon_sweep = np.linspace(-180, 180, n_points)

    for qlat in qlat_lines:
        qlat_arr = np.full(n_points, float(qlat))
        geo_lat, geo_lon = apex_obj.convert(qlat_arr, qlon_sweep, 'qd', 'geo')

        # Convert to xi/eta
        xi, eta = projection.geo2cube(geo_lon, geo_lat)

        # Split on large jumps in xi or eta (dateline / projection discontinuities)
        jumps = np.where(
            (np.abs(np.diff(xi)) > 500) | (np.abs(np.diff(eta)) > 500)
        )[0] + 1

        for seg_xi, seg_eta in zip(np.split(xi, jumps), np.split(eta, jumps)):
            if len(seg_xi) < 2:
                continue
            ax.plot(seg_xi, seg_eta,
                    color=lat_color, linewidth=linewidth,
                    alpha=alpha, linestyle=linestyle, zorder=2)

        if label_gridlines:
            # Label at the first non-NaN point
            valid = np.where(np.isfinite(xi) & np.isfinite(eta))[0]
            if len(valid):
                idx = valid[0]
                ax.text(xi[idx], eta[idx], f'{int(qlat):+d}°',
                        fontsize=7, color=lat_color, alpha=0.8,
                        ha='left', va='bottom', zorder=4)

    # ── QD longitude lines ──────────────────────────────────────────────────
    qlat_sweep = np.linspace(-90, 90, n_points)

    for qlon in qlon_lines:
        qlon_arr = np.full(n_points, float(qlon))
        geo_lat, geo_lon = apex_obj.convert(qlat_sweep, qlon_arr, 'qd', 'geo')

        xi, eta = projection.geo2cube(geo_lon, geo_lat)

        jumps = np.where(
            (np.abs(np.diff(xi)) > 500) | (np.abs(np.diff(eta)) > 500)
        )[0] + 1

        for seg_xi, seg_eta in zip(np.split(xi, jumps), np.split(eta, jumps)):
            if len(seg_xi) < 2:
                continue
            ax.plot(seg_xi, seg_eta,
                    color=lon_color, linewidth=linewidth,
                    alpha=alpha, linestyle=linestyle, zorder=2)

        if label_gridlines:
            valid = np.where(np.isfinite(xi) & np.isfinite(eta))[0]
            if len(valid):
                idx = valid[0]
                ax.text(xi[idx], eta[idx], f'{int(qlon):+d}°',
                        fontsize=7, color=lon_color, alpha=0.8,
                        ha='center', va='top', zorder=4)


def plot_map(ax, xiv, etav, var, obs, grid, RI, cmap, clevels, label, mask=-1, PD=[],
             apex_obj=None, qlat_lines=range(50, 90, 5), qlon_lines=range(0, 360, 10)):
    
    ximin = np.min(xiv)
    ximax = np.max(xiv)
    etamin = np.min(etav)
    etamax = np.max(etav)
    
    fill = False
    if np.all(mask != -1):
        fill = True
        var = np.ma.array(var, mask=mask < 0.6)

    # Check if Prediction Domain is used
    pe1 = [mpe.Stroke(linewidth=6, foreground='white',alpha=1), mpe.Normal()]
    if len(PD) != 0:
        ax.plot([PD[0][0], PD[0][0], PD[0][1], PD[0][1], PD[0][0]],
                [PD[1][0], PD[1][1], PD[1][1], PD[1][0], PD[1][0]],
                color='k', linewidth=5, path_effects=pe1)

    # plot the data tracks:
    #for i in range(4):
    #    lon = obs['lon_' + str(i+1)]
    #    lat = obs['lat_' + str(i+1)]
    #    xi, eta = grid.projection.geo2cube(lon, lat)
    #    ax.plot(xi, eta, color = 'C' + str(i), linewidth = 5, path_effects=pe1)

    # plot map
    if fill:
        cc = ax.contourf(xiv, etav, var, 
                         levels=clevels, cmap=cmap, zorder=0, extend='both')
    else:
        cc = ax.tricontourf(xiv, etav, var,
                            levels=clevels, cmap=cmap, zorder=0, extend='both')
    
    # plot coordinate grids, fix aspect ratio and axes in each panel
    if apex_obj is not None:
        add_qd_gridlines_res(ax, apex_obj, grid.projection,
                            qlat_lines=qlat_lines, qlon_lines=qlon_lines,
                            lat_color='grey', lon_color='grey',
                            linewidth=0.5, alpha=0.6, linestyle='-',
                            label_gridlines=False)
    else:
        # fallback to original geographic grid lines if no apex_obj provided
        for l in np.r_[60:90:5]:
            xi, eta = grid.projection.geo2cube(np.linspace(0, 360, 360), np.ones(360)*l)
            ax.plot(xi, eta, color='lightgrey', linewidth=.5, zorder=1)
        for l in np.r_[0:360:15]:
            xi, eta = grid.projection.geo2cube(np.ones(360)*l, np.linspace(50, 90, 360))
            ax.plot(xi, eta, color='lightgrey', linewidth=.5, zorder=1)

    ax.axis('off')

    # Write labels:
    ax.text(ximin - 25/(RI * 1e-3), etamax - 25/(RI * 1e-3), label, va = 'top', ha = 'left', bbox = dict(facecolor='white', alpha=1), zorder = 101, size = 14)

    # set plot limits and write label:
    ax.set_xlim(ximin, ximax)
    ax.set_ylim(etamin, etamax)
    ax.text(ximin - 25/(RI * 1e-3), etamax - 25/(RI * 1e-3), label, va = 'top', ha = 'left', bbox = dict(facecolor='white', alpha=1), zorder = 101, size = 14)
        
    ax.set_adjustable('datalim') 
    ax.set_aspect('equal')

    return cc



def basic_plot_resolution(xi_FWHM, eta_FWHM, ef, grid, obs, RI, PD=[], apex_obj=None, title='Spatial resolution', vmax=None, diff=False):
    
    # Start figure
    fig, axs = plt.subplots(1, 2, figsize=(10.88, 7))
    cax = fig.add_axes([0.1, 0.1, 0.8, 0.03]) 
    
    mask = np.zeros(grid.shape)
    mask[ef] = 1
    
    # colorbar range
    cmap='Reds'
    if vmax == None:
        vmax = np.max([np.max(xi_FWHM[ef]), np.max(eta_FWHM[ef])])
    clevels = np.linspace(0, vmax, 40)

    if diff == True:
        vmax_abs = np.max([np.max(np.abs(xi_FWHM[ef])), np.max(np.abs(eta_FWHM[ef]))])
        clevels = np.linspace(-vmax_abs, vmax_abs, 80)
        cmap=plt.cm.bwr

    # plot magnetic field in upward direction (MHD and retrieved)
    for (ax, var, label) in zip(axs, [xi_FWHM, eta_FWHM], 
                               ['East (geo)', 'North (geo)']):
        cc = plot_map(ax, grid.xi, grid.eta, var, obs, grid, RI, cmap, clevels, label, mask=mask, PD=PD,
              apex_obj=apex_obj)
        
    # plot colorbar:
    cax.contourf(np.vstack((cc.levels, cc.levels)), np.vstack((np.zeros(cc.levels.size), np.ones(cc.levels.size))), np.vstack((cc.levels, cc.levels)), cmap=cmap, levels=cc.levels)
    cax.set_xlabel('[km]')
    cax.set_yticks([])

    # remove whitespace
    fig.suptitle(title, fontsize=18)
    plt.subplots_adjust(bottom=.05, top=.92, left=.01, right=.99)

    return axs, cax


def left_right(PSF_i, fraq=0.5, inside=False, x='', x_min='', x_max=''):
    if inside:
        PSF_ii = copy.deepcopy(PSF_i)
        valid = False
        while not valid:
            i_max = np.argmax(PSF_ii)
            
            x_i = x[i_max]
            if (x_i >= x_min) and (x_i <= x_max):
                valid = True
            else:
                PSF_ii[i_max] = np.min(PSF_i)            
                        
    else:
        i_max = np.argmax(PSF_i)    

    PSF_max = PSF_i[i_max]
        
    j = 0
    i_left = 0
    left_edge = True
    while (i_max - j) >= 0:
        if PSF_i[i_max - j] < fraq*PSF_max:
            
            dPSF = PSF_i[i_max - j + 1] - PSF_i[i_max - j]
            dx = (fraq*PSF_max - PSF_i[i_max - j]) / dPSF
            i_left = i_max - j + dx
            
            left_edge = False
            
            break
        else:
            j += 1

    j = 0
    i_right = len(PSF_i) - 1
    right_edge = True
    while (i_max + j) < len(PSF_i):
        if PSF_i[i_max + j] < fraq*PSF_max:
            
            dPSF = PSF_i[i_max + j] - PSF_i[i_max + j - 1]
            dx = (fraq*PSF_max - PSF_i[i_max + j - 1]) / dPSF
            i_right = i_max + j - 1 + dx 
            
            right_edge = False
            
            break
        else:
            j += 1

    flag = True
    if left_edge and right_edge:
        print('I think something is wrong')
        flag = False
    elif left_edge:
        i_left = i_max - (i_right - i_max)
        flag = False
    elif right_edge:
        i_right = i_max + (i_max - i_left)
        flag = False

    return i_left, i_right, flag


def get_resolution(R, grid):
    xi_FWHM  = np.zeros(grid.size)
    eta_FWHM = np.zeros(grid.size)
    xi_flag  = np.zeros(grid.size).astype(int)
    eta_flag = np.zeros(grid.size).astype(int)
    for i in range(R.shape[0]):
        
        #PSF = abs(R[:, i].reshape(grid.shape))
        PSF = R[:, i].reshape(grid.shape)
        
        PSF_xi = np.sum(PSF, axis=0)
        i_left, i_right, flag = left_right(PSF_xi)
        xi_FWHM[i] = (i_right-i_left)*grid.Lres/1000 #Assuming Lres is cell size and not number of cells. Also assumes size in metres, hence the conversion to km
        xi_flag[i] = flag
        
        PSF_eta = np.sum(PSF, axis=1)
        i_left, i_right, flag = left_right(PSF_eta)
        eta_FWHM[i] = (i_right-i_left)*grid.Wres/1000 #Assuming Wres is cell size and not number of cells. Also assumes size in metres, hence the conversion to km
        eta_flag[i] = flag
        
    xi_FWHM = xi_FWHM.reshape(grid.shape)
    eta_FWHM = eta_FWHM.reshape(grid.shape)
    xi_flag = xi_flag.reshape(grid.shape)
    eta_flag = eta_flag.reshape(grid.shape)
        
    return xi_FWHM, eta_FWHM, xi_flag, eta_flag


                    
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
def add_coastlines_resolution(axs, grid, linewidth=1, color='black', zorder=5):
    """
    Add coastlines to axes that use grid.projection.geo2cube coordinate system.
    
    Parameters:
        axs    : list of matplotlib axes (or single ax)
        grid   : grid object with projection.geo2cube method
        linewidth, color, zorder : styling options
    """
    shpfilename = shpreader.natural_earth(resolution='110m',
                                          category='physical',
                                          name='coastline')
    reader = shpreader.Reader(shpfilename)
    coastlines = reader.geometries()

    # Flatten to a 1D list of axes regardless of input shape
    if isinstance(axs, np.ndarray):
        ax_list = axs.flatten().tolist()
    elif hasattr(axs, '__iter__'):
        ax_list = list(axs)
    else:
        ax_list = [axs]

    for geom in coastlines:
        if geom.geom_type == 'MultiLineString':
            segments = geom.geoms
        else:
            segments = [geom]

        for seg in segments:
            lons, lats = zip(*seg.coords)
            lons = np.array(lons)
            lats = np.array(lats)

            mask = lats >= 50
            if not np.any(mask):
                continue

            indices = np.where(mask)[0]
            breaks = np.where(np.diff(indices) > 1)[0] + 1
            chunks = np.split(indices, breaks)

            for chunk in chunks:
                if len(chunk) < 2:
                    continue
                xi, eta = grid.projection.geo2cube(lons[chunk], lats[chunk])
                for ax in ax_list:
                    ax.plot(xi, eta, color=color, linewidth=linewidth,
                            zorder=zorder, transform=ax.transData)


def add_stations(axs, xi_stations, eta_stations, zorder=10, **scatter_kwargs):
    """
    Add station scatter plot to axes using xi/eta coordinates.
    
    Parameters:
        axs            : list of axes or single ax
        xi_stations    : array of xi coordinates
        eta_stations   : array of eta coordinates
        zorder         : drawing order (default 10, above coastlines)
        **scatter_kwargs: passed to ax.scatter (s, color, marker, edgecolors, etc.)
    """
    if isinstance(axs, np.ndarray):
        ax_list = axs.flatten().tolist()
    elif hasattr(axs, '__iter__'):
        ax_list = list(axs)
    else:
        ax_list = [axs]

    defaults = dict(s=30, color='blue', marker='^', edgecolors='black', linewidths=0.5)
    defaults.update(scatter_kwargs)

    for ax in ax_list:
        ax.scatter(xi_stations, eta_stations, zorder=zorder, **defaults)


def comparison_plot_resolution(xi_FWHM_1, eta_FWHM_1, xi_FWHM_2, eta_FWHM_2,
                                ef, ef2, grid, obs, RI, PD=[], apex_obj=None,
                                title='Spatial resolution comparison',
                                vmax=None, vmax_diff=None,
                                labels=('Run 1', 'Run 2')):

    fig = plt.figure(figsize=(16, 9), layout='constrained')
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.08], hspace=0.05, wspace=0.05)

    axs = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])
    cax_normal = fig.add_subplot(gs[2, :2])
    cax_diff   = fig.add_subplot(gs[2, 2])

    mask = np.zeros(grid.shape)
    mask[ef] = 1
    mask2 = np.zeros(grid.shape)
    mask2[ef2] = 1

    # --- Normal colorbar range ---
    cmap_normal = 'Reds'
    if vmax is None:
        vmax = np.max([
            np.max(xi_FWHM_1[ef]),  np.max(eta_FWHM_1[ef]),
            np.max(xi_FWHM_2[ef2]),  np.max(eta_FWHM_2[ef2]),
        ])
    clevels_normal = np.linspace(0, vmax, 40)

    # --- Diff colorbar range ---
    cmap_diff = plt.cm.bwr
    xi_diff  = xi_FWHM_1  - xi_FWHM_2
    eta_diff = eta_FWHM_1 - eta_FWHM_2
    if vmax_diff is None:
        vmax_diff = np.max([
            np.max(np.abs(xi_diff[ef])),
            np.max(np.abs(eta_diff[ef])),
        ])
    clevels_diff = np.linspace(-vmax_diff, vmax_diff, 80)

    # --- Plot all 6 panels ---
    plot_specs = [
        (axs[0, 0], xi_FWHM_1, mask, clevels_normal, cmap_normal, 'East (QD)'),
        (axs[0, 1], xi_FWHM_2, mask2, clevels_normal, cmap_normal, 'East (QD)'),
        (axs[0, 2], xi_diff, mask2,   clevels_diff,   cmap_diff,   'East (QD)'),
        (axs[1, 0], eta_FWHM_1, mask, clevels_normal, cmap_normal, 'North (QD)'),
        (axs[1, 1], eta_FWHM_2, mask2, clevels_normal, cmap_normal, 'North (QD)'),
        (axs[1, 2], eta_diff, mask2,  clevels_diff,   cmap_diff,   'North (QD)'),
    ]

    cc_normal = None
    cc_diff   = None
    for ax, var, mask, clevels, cmap, row_label in plot_specs:
        cc = plot_map(ax, grid.xi, grid.eta, var, obs, grid, RI,
                      cmap, clevels, row_label, mask=mask, PD=PD, apex_obj=apex_obj)
        if cmap == cmap_normal:
            cc_normal = cc
        else:
            cc_diff = cc

    # --- Column titles ---
    col_titles = [labels[0], labels[1], 'Difference']
    for ax, col_title in zip(axs[0], col_titles):
        ax.set_title(col_title, fontsize=18, pad=13)

    # --- Colorbars ---
    def _draw_cbar(cax, cc, cmap, label='[km]', fontsize=14):
        cax.contourf(
            np.vstack((cc.levels, cc.levels)),
            np.vstack((np.zeros(cc.levels.size), np.ones(cc.levels.size))),
            np.vstack((cc.levels, cc.levels)),
            cmap=cmap, levels=cc.levels,
        )
        cax.set_xlabel(label, fontsize=fontsize)
        cax.tick_params(axis='x', labelsize=fontsize)
        cax.set_yticks([])
        

    _draw_cbar(cax_normal, cc_normal, cmap_normal)
    _draw_cbar(cax_diff,   cc_diff,   cmap_diff)

    #fig.suptitle(title, fontsize=18)

    return axs, cax_normal, cax_diff
